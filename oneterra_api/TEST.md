# OneTerra-5B 独立 API 测试说明
## 1. 启动服务

```bash
bash xsam_api_021.sh
```

参数（可选）：`$1` checkpoint，`$2` work-dir，`$3` 端口，`$4` host，`$5` log-dir。

默认：

- checkpoint：`/mnt_llm_A100_V1/shui/iter_44000.pth`
- work-dir：`./api_work_021`（和 `demo_work_021` 分开）
- 端口：`7871`

**起来算通过：** 日志出现 `Model loaded.` 和 `API: http://0.0.0.0:7871/docs`，且进程一直挂着。缺 `fastapi` / `uvicorn` 时脚本会尝试 `pip install`。

另开一个终端做下面的 curl（服务终端不要停）。

---

## 2. 探活（不走模型）

```bash
curl -s "$API/health"
curl -s "$API/v1/tasks"
```

**通过标准**

- `/health` 含 `"status":"ok"`，`"tasks"` 含 `imgconv`、`ovseg`、`refseg`、`reaseg`
- `/v1/tasks` 返回四个任务及中文说明

---

## 3. 四个任务（走模型）

统一接口：`POST /v1/infer`（multipart）。  
二选一给图：`-F image=@$IMG` 或 `-F image_path=/服务器上的路径`。

成功时 JSON 常见字段：

| 字段 | 含义 |
|---|---|
| `ok` | `llm_output` 非空则为 true |
| `task` | 实际任务名 |
| `llm_input` / `llm_output` | 模型侧文本 |
| `inference_time` | 秒 |
| `seg_success` | 是否写出可视化图 |
| `vis_url` | 如图 `/v1/results/20260824_....png` |
| `vis_path` | 磁盘绝对路径，在 `api_work_021/api_outputs/` |
| `warning` | 分割任务没有 vis 时会出现 |

取可视化图：

```bash
# 把 vis_url 换成返回值里的路径
curl -o /tmp/oneterra_vis.png "$API/v1/results/文件名.png"
```

图很大时不要加 `include_base64=true`，除非你要测 base64。

---

### 3.1 imgconv（图像问答）

```bash
curl -s --max-time 300 \
  -F task=imgconv \
  -F prompt='Can you describe this image briefly? Please elaborate on your response.' \
  -F image=@"$IMG" \
  "$API/v1/infer"
```

**通过：** `ok=true`，`llm_output` 有自然语言描述；`seg_success` 可为 false，`vis_url` 可为 null（对话任务本来可以没有分割图）。

**失败：** `ok=false`，或 `llm_output` 为空。

---

### 3.2 ovseg（开放词汇全景分割）

**重要：不要用 `curl -F prompt='thing: a; stuff: b'`。**  
curl 的 `-F` 会把 `;` 后面当成表单选项（`type=` / `filename=`）截掉，服务端只能收到 `thing: a`。这是 curl 行为，不是模型 bug。

推荐写法：拆成两个字段（无分号）：

```bash
curl -sS --max-time 300 \
  -F task=ovseg \
  -F thing='building, car, bridge, ship' \
  -F stuff='road, grassland, river, farmland' \
  -F score_threshold=0.0 \
  -F mask_threshold=0.5 \
  -F image=@"$IMG" \
  "$API/v1/infer"
```
例子：
"
curl -sS --max-time 300 \
  -F task=ovseg \
  -F thing='building, car, bridge, ship' \
  -F stuff='road, grassland, river, farmland' \
  -F score_threshold=0.0 \
  -F mask_threshold=0.5 \
  -F image=@/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/test_images/bridge_229_GSD_0.5.png \
  http://127.0.0.1:7871/v1/infer
"
---

用其他机器访问的时候，相应修改API地址即可

### 3.3 refseg（指代分割）

prompt 用短指代表达，不要整段闲聊。

```bash
curl -s --max-time 300 \
  -F task=refseg \
  -F prompt='the road on the bottom right' \
  -F mask_threshold=0.5 \
  -F image=@"$IMG" \
  "$API/v1/infer"
```

例子：
“
curl -sS --max-time 300 \
  -F task=refseg \
  -F prompt='parking lot' \
  -F mask_threshold=0.5 \
  -F image=@/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/test_images/fast_train_904_0000.png \
  http://127.0.0.1:7871/v1/infer
”

用其他机器访问的时候，相应修改API地址即可



---

### 3.4 reaseg（推理分割）

prompt 用问句/推理句，不是单纯物体名。

```bash
curl -s --max-time 300 \
  -F task=reaseg \
  -F prompt='What region is likely used for transportation?' \
  -F mask_threshold=0.5 \
  -F image=@"$IMG" \
  "$API/v1/infer"
```
例子：
“
curl -s --max-time 300 \
  -F task=reaseg \
  -F prompt='What region is likely used for seafood farming ?' \
  -F mask_threshold=0.5 \
  -F image=@/mnt_llm_A100_V1/shui/LAE/RS-Xsam-main-old/test_images/bridge_1477_GSD_2.png \
  http://127.0.0.1:7871/v1/infer
”

用其他机器访问的时候，相应修改API地址即可

---

### 3.5 本机路径（不上传文件）

图已经在这台机器上时：

```bash
curl -s --max-time 300 \
  -F task=imgconv \
  -F prompt='Can you describe this image briefly?' \
  -F image_path="$IMG" \
  "$API/v1/infer"
```

**通过：** 与 `-F image=@` 同样能出 `llm_output`。

---
