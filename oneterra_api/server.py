#!/usr/bin/env python
"""独立 HTTP 推理服务：只调用现有 XSamDemo，不修改 xsam/xsam/demo 下的 Gradio / demo 代码。

给后续新 Gradio 或 Agent 使用。任务与 app_021 一致：imgconv / ovseg / refseg / reaseg。
"""

from __future__ import annotations

import argparse
import base64
import datetime
import os
import os.path as osp
import re
import threading
import time
import traceback
import warnings
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource

from xsam.demo.demo import XSamDemo
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

set_default_logging_format()
warnings.filterwarnings("ignore")

SUPPORTED_TASKS = ("imgconv", "ovseg", "refseg", "reaseg")
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_MASK_THRESHOLD = 0.5
TASK_HINTS = {
    "imgconv": "根据图像回答自然语言问题。",
    "ovseg": "开放词汇全景分割。推荐用表单字段 thing= / stuff=（避免 curl -F 把分号截断）；也可用 prompt。",
    "refseg": "指代表达分割：用自然语言描述目标。",
    "reaseg": "推理分割：根据推理类问题定位并分割。",
}


def _split_class_list(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_ovseg_prompt(
    prompt: Optional[str] = None,
    thing: Optional[str] = None,
    stuff: Optional[str] = None,
) -> str:
    """组装 ovseg 提示。优先用独立 thing/stuff 字段，避免 curl -F 把 ';' 当表单选项截断。"""
    thing_classes = _split_class_list(thing)
    stuff_classes = _split_class_list(stuff)
    if thing_classes or stuff_classes:
        parts = []
        if thing_classes:
            parts.append("thing: " + ", ".join(thing_classes))
        if stuff_classes:
            parts.append("stuff: " + ", ".join(stuff_classes))
        return "; ".join(parts)

    text = (prompt or "").strip()
    if not text:
        raise ValueError(
            "ovseg 需要 prompt，或提供 thing= / stuff= 字段。"
            "注意：curl 的 -F 会把 ';' 后面截掉，请用 -F thing=... -F stuff=...，"
            "或改用 --form-string 'prompt=thing: a; stuff: b'"
        )
    # 常见踩坑：curl -F 'prompt=thing: a; stuff: b' → 服务端只收到 thing 段
    if re.search(r"(?i)\bthing\s*:", text) and not re.search(r"(?i)\bstuff\s*:", text):
        print_log(
            "[API] ovseg prompt 只有 thing、没有 stuff。"
            "若你本意带 stuff，多半是 curl -F 把 ';' 截断了；"
            "请改用 -F thing=... -F stuff=... 或 --form-string。",
            logger="current",
        )
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneTerra-5B 独立 HTTP API（不改动原 Gradio/评测）")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./api_work_021",
        help="API 结果落盘目录（与 demo_work_021 分开）",
    )
    parser.add_argument("--pth_model", type=str, default=None, help="checkpoint 路径或 latest")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--log-dir", type=str, default="./api_logs_021", help="API 日志目录")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument("--port", type=int, default=7871, help="API 端口（默认 7871，避免与 Gradio 7870 冲突）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind host")
    return parser.parse_args()


def setup_cfg(args):
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.seed is not None:
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    register_function(cfg._cfg_dict)

    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if args.work_dir and osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        elif args.work_dir:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        else:
            raise ValueError("work_dir must be specified when using 'latest' checkpoint")
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    return args, cfg


def _pil_from_ndarray_or_image(seg_output):
    if seg_output is None:
        return None
    if isinstance(seg_output, Image.Image):
        return seg_output.convert("RGB")
    arr = np.asarray(seg_output)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[-1] == 4:
        return Image.fromarray(arr[..., :3])
    return Image.fromarray(arr)


def create_app(demo: XSamDemo, work_dir: str, log_dir: str) -> FastAPI:
    infer_lock = threading.Lock()
    out_dir = osp.join(work_dir, "api_outputs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    app = FastAPI(
        title="OneTerra-5B Inference API",
        description="独立封装 XSamDemo.run_on_image，供新 Gradio / Agent 调用。单卡串行。",
        version="0.1.0",
    )

    def _save_vis(seg_output, request_id: str) -> Optional[str]:
        vis = _pil_from_ndarray_or_image(seg_output)
        if vis is None:
            return None
        rel_name = f"{request_id}.png"
        abs_path = osp.join(out_dir, rel_name)
        vis.save(abs_path)
        return rel_name

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "tasks": list(SUPPORTED_TASKS)}

    @app.get("/v1/tasks")
    def list_tasks():
        return {
            "tasks": [
                {"name": name, "description": TASK_HINTS[name]}
                for name in SUPPORTED_TASKS
            ]
        }

    @app.get("/v1/results/{filename}")
    def get_result(filename: str):
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        path = osp.join(out_dir, filename)
        if not osp.isfile(path):
            raise HTTPException(status_code=404, detail="result not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/v1/infer")
    async def infer(
        task: str = Form(..., description="imgconv | ovseg | refseg | reaseg"),
        prompt: Optional[str] = Form(
            None,
            description="任务提示词。ovseg 若含 thing:...; stuff:...，请勿用 curl -F（分号会被截断），改用 thing/stuff 字段或 --form-string",
        ),
        thing: Optional[str] = Form(
            None,
            description="仅 ovseg：thing 类别，逗号分隔，如 building, car, bridge",
        ),
        stuff: Optional[str] = Form(
            None,
            description="仅 ovseg：stuff 类别，逗号分隔，如 road, grassland, river",
        ),
        image: Optional[UploadFile] = File(None, description="输入图像文件"),
        image_path: Optional[str] = Form(None, description="服务器本地图像路径（与 image 二选一）"),
        score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
        mask_threshold: float = Form(DEFAULT_MASK_THRESHOLD),
        include_base64: bool = Form(False, description="是否在 JSON 中附带可视化图 base64"),
    ):
        task_name = (task or "").strip().lower()
        if task_name not in SUPPORTED_TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported task: {task}. allowed: {list(SUPPORTED_TASKS)}",
            )

        if task_name == "ovseg":
            try:
                final_prompt = build_ovseg_prompt(prompt=prompt, thing=thing, stuff=stuff)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            final_prompt = (prompt or "").strip()
            if not final_prompt:
                raise HTTPException(status_code=400, detail="prompt is required")

        if image is not None and image.filename:
            raw = await image.read()
            if not raw:
                raise HTTPException(status_code=400, detail="empty image upload")
            try:
                from io import BytesIO

                pil_image = Image.open(BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"cannot decode image: {e}")
        elif image_path:
            if not osp.isfile(image_path):
                raise HTTPException(status_code=400, detail=f"image_path not found: {image_path}")
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"cannot open image_path: {e}")
        else:
            raise HTTPException(status_code=400, detail="provide either image file or image_path")

        run_kwargs = {}
        if task_name == "ovseg":
            run_kwargs["threshold"] = float(score_threshold)
            run_kwargs["mask_threshold"] = float(mask_threshold)
        elif task_name in ("refseg", "reaseg"):
            run_kwargs["mask_threshold"] = float(mask_threshold)

        request_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print_log(
            f"[API] 收到请求 id={request_id} task={task_name} prompt={final_prompt!r}",
            logger="current",
        )

        acquired = infer_lock.acquire(timeout=600)
        if not acquired:
            raise HTTPException(status_code=503, detail="inference queue timeout, retry later")
        start = time.time()
        try:
            llm_input, llm_output, seg_output = demo.run_on_image(
                pil_image,
                final_prompt,
                task_name,
                **run_kwargs,
            )
        except Exception as e:
            print_log(f"[API] error: {e}\n{traceback.format_exc()}", logger="current")
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")
        finally:
            infer_lock.release()

        elapsed = time.time() - start
        vis_name = _save_vis(seg_output, request_id)
        vis_url = f"/v1/results/{vis_name}" if vis_name else None
        vis_abs = osp.join(out_dir, vis_name) if vis_name else None

        payload = {
            "ok": llm_output is not None,
            "request_id": request_id,
            "task": task_name,
            "prompt": final_prompt,
            "llm_input": llm_input,
            "llm_output": llm_output,
            "inference_time": round(elapsed, 3),
            "seg_success": vis_name is not None,
            "vis_filename": vis_name,
            "vis_url": vis_url,
            "vis_path": vis_abs,
        }
        if include_base64 and vis_abs and osp.isfile(vis_abs):
            with open(vis_abs, "rb") as f:
                payload["vis_base64"] = base64.b64encode(f.read()).decode("ascii")

        if task_name != "imgconv" and vis_name is None:
            payload["warning"] = "segmentation visualization is empty"

        return JSONResponse(payload)

    return app


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    args, cfg = setup_cfg(args)

    print_log(f"work-dir: {osp.abspath(args.work_dir)}", logger="current")
    print_log("Initializing OneTerra API (XSamDemo)...", logger="current")
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)
    print_log("Model loaded.", logger="current")

    app = create_app(demo, args.work_dir, args.log_dir)
    print_log(f"API: http://{args.host}:{args.port}/docs", logger="current")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
