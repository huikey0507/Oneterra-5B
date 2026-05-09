from .gcg_seg_evaluator import GCGSegEvaluator
from .generic_seg_evaluator import GenericSegEvaluator
from .imgconv_evaluator import ImgConvEvaluator
from .imgconv_cc_evaluator import ImgConvCCEvaluator
from .inter_seg_evaluator import InterSegEvaluator
from .ov_seg_evaluator import OVSegEvaluator
from .reason_seg_evaluator import ReasonSegEvaluator
from .refer_seg_evaluator import ReferSegEvaluator
from .vgd_seg_evaluator import VGDSegEvaluator
from .imgconv_mlsc_evaluator import ImgConvMLSCEvaluator

# 添加别名以支持配置文件中的简短名称
GenSegEvaluator = GenericSegEvaluator
RefSegEvaluator = ReferSegEvaluator

__all__ = [
    "GenericSegEvaluator",
    "GenSegEvaluator",  # 别名
    "ReferSegEvaluator",
    "RefSegEvaluator",  # 别名
    "ReasonSegEvaluator",
    "GCGSegEvaluator",
    "VGDSegEvaluator",
    "InterSegEvaluator",
    "OVSegEvaluator",
    "ImgConvEvaluator",
    "ImgConvCCEvaluator",
    "ImgConvMLSCEvaluator",
]