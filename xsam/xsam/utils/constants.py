from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

DEFAULT_SEG_TOKEN = "<SEG>"
DEFAULT_CLS_TOKEN = "<CLS>"
DEFAULT_PSTART_TOKEN = "<p>"
DEFAULT_PEND_TOKEN = "</p>"
DEFAULT_REGION_TOKEN = "<region>"
REGION_TOKEN_INDEX = -300

DEFAULT_TASKS = ["imgconv", "genseg", "refseg", "reaseg", "gcgseg", "ovseg", "interseg", "vgdseg"]
# 需要分割输出的任务（除 imgconv 外）。训练与推理一致：question 模板末尾含 <SEG>，mask 触发以问题里的 <SEG> 为准；若无则用回答中的 <SEG> 或词表兜底。
SEG_REQUIRED_TASKS = ["genseg", "refseg", "reaseg", "gcgseg", "ovseg", "interseg", "vgdseg"]
TOKEN2INDEX = {
    DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX,
    DEFAULT_REGION_TOKEN: REGION_TOKEN_INDEX,
}
INDEX2TOKEN = {v: k for k, v in TOKEN2INDEX.items()}
