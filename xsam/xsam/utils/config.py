from typing import Optional, Tuple

from mmengine.config import Config
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList, StoppingCriteria
from xtuner.utils import StopWordStoppingCriteria


class DelayedStopWordStoppingCriteria(StoppingCriteria):
    """延迟停止条件：只有在生成了分割token之后才允许停止"""
    def __init__(self, tokenizer, stop_word, seg_token_idx=None, min_tokens_after_seg=10, eos_token_id=None):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.seg_token_idx = seg_token_idx
        self.min_tokens_after_seg = min_tokens_after_seg
        self.eos_token_id = eos_token_id
        # 如果stop_word是None，说明这是EOS token停止条件
        if stop_word is None and eos_token_id is not None:
            self.stop_word_ids = [eos_token_id]
        else:
            self.stop_word_ids = tokenizer(stop_word, add_special_tokens=False)["input_ids"]
        self.seg_token_found = False
        self.tokens_after_seg = 0
        
    def __call__(self, input_ids, scores, **kwargs):
        # 检查是否找到了分割token（检查整个序列，不只是最后一个token）
        if self.seg_token_idx is not None and not self.seg_token_found:
            # 检查整个序列中是否有分割token
            if (input_ids[0] == self.seg_token_idx).any():
                self.seg_token_found = True
                # 找到分割token的位置
                seg_positions = (input_ids[0] == self.seg_token_idx).nonzero(as_tuple=True)[0]
                if len(seg_positions) > 0:
                    # 计算从最后一个分割token到序列末尾的token数
                    last_seg_pos = seg_positions[-1].item()
                    self.tokens_after_seg = input_ids[0].shape[0] - last_seg_pos - 1
        
        # 如果找到了分割token，更新计数（从最后一个分割token到当前末尾）
        if self.seg_token_found and self.seg_token_idx is not None:
            seg_positions = (input_ids[0] == self.seg_token_idx).nonzero(as_tuple=True)[0]
            if len(seg_positions) > 0:
                last_seg_pos = seg_positions[-1].item()
                self.tokens_after_seg = input_ids[0].shape[0] - last_seg_pos - 1
        
        # 检查是否遇到了停止词或EOS token
        # 对于EOS token，检查最后一个token
        if self.eos_token_id is not None:
            if input_ids[0, -1].item() == self.eos_token_id:
                # 如果还没找到分割token，不允许停止
                if not self.seg_token_found:
                    return False
                # 如果找到后生成的token太少，不允许停止
                if self.tokens_after_seg < self.min_tokens_after_seg:
                    # 添加调试信息
                    from mmengine.logging import print_log
                    print_log(
                        f"DelayedStopWord: Found EOS token but only {self.tokens_after_seg} tokens after <SEG> "
                        f"(required: {self.min_tokens_after_seg}), preventing stop",
                        logger="current"
                    )
                    return False
                return True
        
        # 对于停止词，检查序列末尾
        if len(input_ids[0]) >= len(self.stop_word_ids):
            # 检查序列末尾是否匹配停止词
            sequence_end = input_ids[0, -len(self.stop_word_ids):].tolist()
            if sequence_end == self.stop_word_ids:
                # 如果还没找到分割token，不允许停止
                if not self.seg_token_found:
                    return False
                # 如果找到后生成的token太少，不允许停止
                if self.tokens_after_seg < self.min_tokens_after_seg:
                    # 添加调试信息
                    from mmengine.logging import print_log
                    print_log(
                        f"DelayedStopWord: Found stop word '{self.stop_word}' but only {self.tokens_after_seg} tokens after <SEG> "
                        f"(required: {self.min_tokens_after_seg}), preventing stop",
                        logger="current"
                    )
                    return False
                return True
        return False


def setup_model_config(model, cfg: Config) -> Tuple[Optional[StoppingCriteriaList], Optional[GenerationConfig]]:
    """Setup model configuration for generation."""
    stop_criteria = None
    generation_config = None

    if model.llm is not None:
        prompt_template = cfg.prompt_template
        stop_words = []
        if isinstance(prompt_template, str):
            prompt_template = get_object_from_string(prompt_template)
        stop_words += prompt_template.get("STOP_WORDS", [])

        stop_criteria = StoppingCriteriaList()
        # 获取分割token的索引（如果存在）
        # 注意：模型初始化时seg_token_idx默认为-1，需要检查是否>=0
        seg_token_idx = None
        if hasattr(model, 'seg_token_idx') and model.seg_token_idx is not None and model.seg_token_idx >= 0:
            seg_token_idx = model.seg_token_idx
        
        # 对于分割任务，使用延迟停止条件
        # 如果找到了有效的seg_token_idx（>=0），对所有停止词都使用延迟停止条件
        use_delayed_stop = seg_token_idx is not None and seg_token_idx >= 0
        
        # 调试信息
        if stop_words:
            from mmengine.logging import print_log
            if use_delayed_stop:
                print_log(f"Using delayed stop criteria with seg_token_idx={seg_token_idx}, min_tokens_after_seg=50", logger="current")
            else:
                print_log(f"Using normal stop criteria (seg_token_idx={model.seg_token_idx if hasattr(model, 'seg_token_idx') else 'N/A'})", logger="current")
        for word in stop_words:
            print_log(f"Adding stop word: {word}", logger="current")
            if use_delayed_stop:
                # 使用延迟停止条件，确保在生成分割token后才允许停止
                stop_criteria.append(DelayedStopWordStoppingCriteria(
                    model.tokenizer, word, seg_token_idx=seg_token_idx, min_tokens_after_seg=50  # 增加到50个token，确保有足够空间生成完整的分割结果
                ))
            else:
                # 如果没有分割token，使用普通停止条件
                stop_criteria.append(StopWordStoppingCriteria(model.tokenizer, word))

        # 如果有分割token，禁用GenerationConfig的eos_token_id，完全依赖延迟停止条件
        eos_token_id_for_config = None if use_delayed_stop else model.tokenizer.eos_token_id
        
        generation_config = GenerationConfig(
            max_new_tokens=1024,  # 增加最大生成token数，确保有足够空间生成分割结果
            do_sample=False,
            num_beams=1,
            temperature=1,
            top_p=None,
            bos_token_id=model.tokenizer.bos_token_id,
            eos_token_id=eos_token_id_for_config,  # 如果有延迟停止条件，禁用EOS token
            pad_token_id=(
                model.tokenizer.pad_token_id
                if model.tokenizer.pad_token_id is not None
                else model.tokenizer.eos_token_id
            ),
        )
        
        # 如果有分割token，也需要对EOS token使用延迟停止条件
        if use_delayed_stop and model.tokenizer.eos_token_id is not None:
            # 为EOS token创建延迟停止条件（使用eos_token_id而不是字符串）
            stop_criteria.append(DelayedStopWordStoppingCriteria(
                model.tokenizer, None, seg_token_idx=seg_token_idx, min_tokens_after_seg=50,
                eos_token_id=model.tokenizer.eos_token_id
            ))

    return stop_criteria, generation_config
