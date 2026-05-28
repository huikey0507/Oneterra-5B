#!/usr/bin/env python
"""RS-Xsam 本地推理界面（与仓库根目录 xsam_eval_021.sh 对齐：该脚本调用 eval_ori.py；config / checkpoint / work-dir 与评测一致）。"""

import argparse
import datetime
import os
import os.path as osp
import time
import traceback
import warnings

import cv2
import gradio as gr
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource

from xsam.demo.demo import XSamDemo
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

this_dir = osp.dirname(osp.abspath(__file__))

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# 遥感 / 工作台风格：深蓝底 + 半透明渐变标题 + 白字
custom_css = """
.gradio-container {
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    background: linear-gradient(165deg, #061326 0%, #0b1a38 46%, #081224 100%);
    min-height: 100vh;
}
.main {
    background: rgba(248, 250, 252, 0.97);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    margin: 12px;
    padding: 18px;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
}
.main-header {
    text-align: center;
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.14) 0%,
        rgba(148, 163, 184, 0.08) 48%,
        rgba(59, 130, 246, 0.12) 100%
    );
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.16);
    color: #ffffff;
    padding: 2.5rem 1.75rem;
    border-radius: 16px;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 44px rgba(0, 0, 0, 0.28);
    position: relative;
    overflow: hidden;
}
.main-header h1 {
    font-size: 2.45rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
    letter-spacing: 0.02em;
    position: relative;
    z-index: 1;
    color: #ffffff;
}
.main-header h2 {
    font-size: 1.18rem;
    font-weight: 400;
    margin: 0;
    position: relative;
    z-index: 1;
    color: rgba(255, 255, 255, 0.9);
    line-height: 1.58;
}
.running-info {
    padding: 12px 14px;
    border-radius: 8px;
    border-left: 4px solid #64748b;
    background: #f1f5f9;
}

.input-section, .output-section {
    display: flex;
    flex-direction: column;
}

.input-section > div, .output-section > div {
    flex-grow: 1;
}

/* 输入区域样式优化 */
.input-section {
    padding-right: 10px;
}

.output-section {
    padding-left: 10px;
}

/* Video instruction spacing tweaks */
.video-instruction {
    margin: 3px 0 3px 0 !important;
    padding: 0 !important;
}

/* Reduce the gap before the main row below the video instruction */
.main-row {
    margin-top: 6px !important;
}

/* Usage instructions styling */
.usage-instructions {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #28a745;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.usage-instructions h3 {
    color: #2c3e50;
    margin-top: 0;
    margin-bottom: 15px;
    font-weight: 600;
}

.usage-instructions ul {
    margin: 0;
    padding-left: 0;
}

.usage-instructions li {
    margin-bottom: 8px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.usage-instructions li:last-child {
    border-bottom: none;
}

.usage-instructions strong {
    color: #495057;
}

.task-description {
    border-left: 4px solid #475569;
}

/* 单图上传：按内容比例缩放，避免固定高度造成左右大块黑边 */
.image-upload, .image-upload > div, .image-upload canvas, .image-upload img {
    width: 100% !important;
    height: auto !important;
    max-height: min(72vh, 680px) !important;
    max-width: 100% !important;
    min-height: unset !important;
    object-fit: contain !important;
    display: block !important;
}

.seg-output, .seg-output > div, .seg-output img {
    width: 100% !important;
    height: auto !important;
    max-height: min(78vh, 760px) !important;
    object-fit: contain !important;
}
"""

TASK_DESCRIPTION = {
    "imgconv": "根据图像回答自然语言问题。",
    "genseg": "全景分割（genseg）：与 xsam_predict_genseg_pano_021.sh 相同通路，固定使用 assets/annotations_val.json 全量 pano 类别（tensor）；界面提示词仅作展示，可留空。",
    "ovseg": "开集全景（ovseg）：由用户在提示中自行定义类别；推荐 thing: 实例类; stuff: 背景类，逗号分隔时全部按 stuff 处理。",
    "refseg": "指代表达分割：用自然语言描述目标物体。",
    "reaseg": "推理分割：根据推理类问题定位并分割相关区域。",
}

SUPPORTED_TASKS = list(TASK_DESCRIPTION.keys())

# Examples with proper image paths
EXAMPLES = {
    "imgconv": [
        (
            osp.join(this_dir, "./images/imgconv.jpg")
            if osp.exists(osp.join(this_dir, "./images/imgconv.jpg"))
            else None
        ),
        "Can you describe this image briefly? Please elaborate on your response.",
        "imgconv",
    ],
    "genseg": [
        (osp.join(this_dir, "./images/genseg.jpg") if osp.exists(osp.join(this_dir, "./images/genseg.jpg")) else None),
        "__DEFAULT_GENSEG_PROMPT__",
        "genseg",
    ],
    "ovseg": [
        (osp.join(this_dir, "./images/genseg.jpg") if osp.exists(osp.join(this_dir, "./images/genseg.jpg")) else None),
        "thing: person, car; stuff: road, sky, tree",
        "ovseg",
    ],
    "refseg": [
        (osp.join(this_dir, "./images/refseg.jpg") if osp.exists(osp.join(this_dir, "./images/refseg.jpg")) else None),
        "the white tshirt kid",
        "refseg",
    ],
    "reaseg": [
        (osp.join(this_dir, "./images/reaseg.jpg") if osp.exists(osp.join(this_dir, "./images/reaseg.jpg")) else None),
        "What object can be put into dog food?",
        "reaseg",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RS-Xsam Gradio 推理台（与 xsam_eval_021.sh 一致，该脚本调用 xsam/xsam/tools/eval_ori.py；work-dir/vis 与评测习惯一致）"
    )
    parser.add_argument("config", help="config file name or path")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./demo_work_021",
        help="与 xsam_eval_021.sh（内部 eval_ori.py）一致：每次分割结果额外保存为 vis/<时间戳>.png，并可用于 --pth_model latest",
    )
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint or 'latest' to use the latest checkpoint in work_dir",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="directory to save logs")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument("--port", type=int, default=7860, help="port for gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host for gradio server")
    parser.add_argument("--share", action="store_true", help="share gradio app")
    return parser.parse_args()


class GradioApp:
    def __init__(self, demo: XSamDemo, log_dir: str, work_dir: str, default_genseg_prompt: str):
        self.demo = demo
        self.log_dir = log_dir
        self.work_dir = work_dir
        self.default_genseg_prompt = default_genseg_prompt or ""
        self.processing_status = "Ready"

    def gradio_predict_with_progress(self, data, prompt, task_name="imgconv", score_thr=0.5, progress=gr.Progress()):
        """Enhanced prediction function with progress tracking and better error handling"""
        if data is None:
            return "未提供图像", "", "", None

        try:
            print_log(f"[Gradio] 收到请求 task={task_name}", logger="current")
            progress(0.1, desc="初始化…")

            # Validate inputs（genseg 允许空提示词，与评测一致表示全类别）
            if not prompt or prompt.strip() == "":
                if task_name != "genseg":
                    return "当前任务需要输入提示词", "", "", None

            # Logging setup
            day_timestamp = datetime.datetime.now().strftime("%Y%m%d")
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            day_log_dir = osp.join(self.log_dir, day_timestamp)
            log_file = osp.join(day_log_dir, f"{day_timestamp}.log")
            img_log_dir = osp.join(day_log_dir, "image")
            out_log_dir = osp.join(day_log_dir, "output")

            os.makedirs(day_log_dir, exist_ok=True)
            os.makedirs(img_log_dir, exist_ok=True)
            os.makedirs(out_log_dir, exist_ok=True)

            progress(0.3, desc="处理图像…")

            # Convert PIL image to format expected by demo
            vprompt_masks = None
            if isinstance(data, Image.Image):
                pil_image = data
                array_image = np.array(pil_image)
            elif isinstance(data, np.ndarray):
                pil_image = Image.fromarray(data)
                array_image = data
            elif isinstance(data, dict):
                pil_image = data["background"].convert("RGB")
                array_image = np.array(pil_image)
                vprompt_masks = [np.array(layer)[..., -1] for layer in data["layers"]]
                vprompt_masks = [mask for mask in vprompt_masks if mask.sum() > 0]
                vprompt_masks = None if len(vprompt_masks) == 0 else vprompt_masks
            else:
                raise ValueError(f"Unsupported image type: {type(data)}")

            progress(0.5, desc="模型推理中…")

            # Run prediction using custom logic
            start_time = time.time()

            # genseg/ovseg 与 eval 一致用 threshold=0；过高会滤掉全部 mask，可视化只剩原图
            seg_threshold = 0.0 if task_name in ("genseg", "ovseg") else score_thr
            llm_input, llm_output, seg_output = self.demo.run_on_image(
                pil_image, prompt, task_name, vprompt_masks=vprompt_masks, threshold=seg_threshold
            )

            llm_success = llm_output is not None
            seg_success = seg_output is not None

            inference_time = time.time() - start_time

            progress(0.9, desc="保存结果…")
            # Save input image and output image
            cv2.imwrite(f"{img_log_dir}/{file_timestamp}.png", cv2.cvtColor(array_image, cv2.COLOR_RGB2BGR))
            if seg_success:
                cv2.imwrite(f"{out_log_dir}/{file_timestamp}.png", cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR))
                vis_eval_dir = osp.join(self.work_dir, "vis")
                os.makedirs(vis_eval_dir, exist_ok=True)
                cv2.imwrite(
                    osp.join(vis_eval_dir, f"{file_timestamp}.png"),
                    cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR),
                )

            # Log to file
            if not osp.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("timestamp\timage\tprompt\ttask_name\tinference_time\tllm_success\tseg_success\n")
            with open(log_file, "a") as f:
                f.write(
                    f"{file_timestamp}\t{file_timestamp}.png\t{prompt}\t{task_name}\t{inference_time:.3f}\t{llm_success}\t{seg_success}\n"
                )

            progress(1.0, desc="完成")

            if llm_success or seg_success:
                status_message = f"完成，耗时 {inference_time:.2f}s。"
            else:
                status_message = f"未得到有效输出，耗时 {inference_time:.2f}s。"

            return (
                status_message,
                llm_input,
                llm_output,
                (gr.update(value=seg_output) if seg_output is not None else None),
            )

        except Exception as e:
            error_msg = f"错误: {str(e)}"
            print(f"Error in gradio_predict: {traceback.format_exc()}")
            return error_msg, "", "", None

    def create_interface(self):
        with gr.Blocks(title="遥感基础模型OneTerra-5B", css=custom_css, theme=gr.themes.Soft()) as app:
            gr.HTML(
                """
                <div class="main-header">
                    <h1>遥感基础模型OneTerra-5B</h1>
                    <h2>支持光学和SAR双模态，在统一框架下支持图像对话与问答理解（imgconv）、闭集全景/语义分割（genseg）、开放词汇分割（ovseg）、指代分割（refseg）与推理分割（reaseg），实现图像级语义理解、实例级目标识别、像素级精细分割与语言驱动可控推理的一体化解译能力</h2>
                </div>
            """
            )

            with gr.Row(elem_classes="main-row"):
                with gr.Column(scale=5, elem_classes="input-section"):
                    image_input = gr.Image(
                        type="pil",
                        label="输入图像",
                        elem_classes="image-upload",
                        sources=["upload", "webcam", "clipboard"],
                    )

                    # Enhanced text input with suggestions
                    with gr.Group(elem_classes="prompt-group"):
                        # Task selection
                        task_name = gr.Dropdown(
                            choices=SUPPORTED_TASKS,
                            value="imgconv",
                            label="任务类型",
                            elem_classes="task-dropdown",
                        )

                        task_description = gr.Textbox(
                            value=TASK_DESCRIPTION["imgconv"],
                            label="任务说明",
                            interactive=False,
                            lines=2,
                            elem_classes="task-description",
                        )

                        suggestions_btn = gr.Button(
                            "载入该任务示例（图+提示词）", size="sm", elem_classes="btn-secondary example-btn"
                        )
                        score_thr = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.0,
                            step=0.01,
                            interactive=True,
                            label="分数阈值（refseg/reaseg 等；genseg/ovseg 固定为 0）",
                            elem_classes="score-threshold",
                        )

                        text_input = gr.Textbox(
                            lines=2,
                            label="用户提示词",
                            placeholder="imgconv/ovseg/refseg/reaseg 需填写；genseg 可留空（全 pano 类别）或使用 ins:/sem:。",
                            value="",
                            elem_id="user-prompt-input",
                            elem_classes="prompt-input",
                        )

                    with gr.Row(elem_classes="action-buttons"):
                        submit_btn = gr.Button(
                            "运行推理", variant="primary", size="lg", elem_classes="btn-primary run-btn"
                        )
                        clear_btn = gr.Button(
                            "清空", variant="secondary", elem_classes="btn-secondary clear-btn"
                        )

                with gr.Column(scale=6, elem_classes="output-section"):
                    status_display = gr.Textbox(
                        value="就绪：上传图像并填写提示词后点击「运行推理」。",
                        label="运行状态",
                        interactive=False,
                        elem_classes="running-info status-display",
                        lines=1,
                    )

                    with gr.Group(elem_classes="llm-section"):
                        gr.HTML("<h3 style='margin: 0 0 12px 0; color: #0f172a;'>语言模型</h3>")

                        llm_input = gr.Textbox(
                            value="",
                            label="送入模型的指令（解码）",
                            placeholder="推理后显示。",
                            lines=2,
                            elem_classes="llm-input",
                            interactive=False,
                        )
                        llm_output = gr.Textbox(
                            value="",
                            label="模型回复",
                            placeholder="推理后显示。",
                            lines=4,
                            elem_classes="llm-output",
                            interactive=False,
                        )

                    with gr.Group(elem_classes="seg-section"):
                        seg_output = gr.Image(
                            type="pil",
                            label="分割可视化",
                            elem_classes="seg-output",
                        )

            # Event handlers
            submit_btn.click(
                fn=self.gradio_predict_with_progress,
                inputs=[image_input, text_input, task_name, score_thr],
                outputs=[status_display, llm_input, llm_output, seg_output],
                show_progress=True,
            )

            clear_btn.click(
                fn=lambda: [
                    None,
                    "",
                    "imgconv",
                    "",
                    None,
                    gr.update(value=None),
                    0.5,
                    "🧹 已清空，可重新上传图像。",
                ],
                outputs=[
                    image_input,
                    text_input,
                    task_name,
                    llm_input,
                    llm_output,
                    seg_output,
                    score_thr,
                    status_display,
                ],
            )

            suggestions_btn.click(fn=self.get_examples, inputs=[task_name], outputs=[image_input, text_input])

            def _on_task_change(task, cur_prompt):
                desc = TASK_DESCRIPTION.get(task, "")
                if task == "genseg" and self.default_genseg_prompt and (
                    cur_prompt is None or str(cur_prompt).strip() == ""
                ):
                    return desc, self.default_genseg_prompt
                return desc, gr.update()

            task_name.change(fn=_on_task_change, inputs=[task_name, text_input], outputs=[task_description, text_input])

            # Auto-update status when image is uploaded
            image_input.change(
                fn=lambda data: (
                    "图像已载入，请填写提示词并点击「运行推理」。"
                    if data is not None
                    else "就绪：请上传图像并填写提示词后点击「运行推理」。"
                ),
                inputs=[image_input],
                outputs=[status_display],
            )

        return app

    def get_examples(self, task_name):
        """Get examples for the given task - returns image and text prompt"""
        example = EXAMPLES.get(task_name, None)
        if not example:
            return None, ""
        try:
            image_path = example[0]
            text_prompt = example[1]
            if text_prompt == "__DEFAULT_GENSEG_PROMPT__":
                text_prompt = self.default_genseg_prompt

            # Load image if path exists
            if image_path and osp.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    return image, text_prompt
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}\n{traceback.format_exc()}")
                    return None, text_prompt
            else:
                return None, text_prompt

        except Exception as e:
            print(f"Error processing example for task {task_name}: {e}\n{traceback.format_exc()}")
            return None, ""


def setup_cfg(args):
    """Setup configuration from arguments."""
    # Load and process config
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

    # Handle latest checkpoint
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


def main():
    """启动 RS-Xsam Gradio 推理台。"""
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup configuration
    args, cfg = setup_cfg(args)

    print_log(f"work-dir（评测对齐可视化输出）: {osp.abspath(args.work_dir)}", logger="current")
    print_log("Initializing RS-Xsam demo...", logger="current")
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)
    print_log("Model loaded.", logger="current")

    default_genseg = demo.default_genseg_prompt()
    gradio_app = GradioApp(demo, args.log_dir, args.work_dir, default_genseg)
    app = gradio_app.create_interface()

    print_log(f"Gradio: http://{args.host}:{args.port}", logger="current")
    # max_file_size 限制过大上传，减轻「点了很久后端才收到」的体感（多为大图/队列等待）
    launch_kw = dict(
        show_error=True,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
    )
    try:
        app.launch(**launch_kw, max_file_size="80mb")
    except TypeError:
        app.launch(**launch_kw)


if __name__ == "__main__":
    main()
