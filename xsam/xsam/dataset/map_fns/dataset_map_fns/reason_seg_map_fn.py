import random

from xtuner.utils import DEFAULT_IMAGE_TOKEN

from ....utils.constants import DEFAULT_CLS_TOKEN, DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN

SHORT_QUESTIONS = [
    f"Can you segment {{sent}} in this image? Please output the segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"Can you give the segmentation mask for {{sent}} in this image? Please respond with the segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"Can you highlight {{sent}} and output the corresponding segmentation mask? Please provide the segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"What is {{sent}} in this image? Please output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"What is {{sent}} in this image? Please generate a segmentation mask for this image. {DEFAULT_SEG_TOKEN}",
]

LONG_QUESTIONS = [
    f"{{sent}} Please output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please generate a segmentation mask for this image. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please extract the segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please return the segmentation predictions as masks. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please segment the image. {DEFAULT_SEG_TOKEN}",
]

EXPLANATORY_QUESTIONS = [
    f"{{sent}} Please explain why and output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please explain the reason and output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please give some explanation and output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please explain the reason and output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
    f"{{sent}} Please give detailed explanation and output the corresponding segmentation mask. {DEFAULT_SEG_TOKEN}",
]

# 回答不含 <SEG>，训练/推理一致：mask 触发只用问题里的 <SEG>
ANSWER_LIST = [
    "Sure.",
    "OK.",
    "Done.",
    "Here is the mask.",
    "Done.",
]

ANSWER_LIST_WITH_EXPLANATION = [
    "Sure.",
    "OK.",
    "And done.",
    "Here is the mask.",
    "Done.",
]

P_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN
C_FORMAT = "{} " + DEFAULT_CLS_TOKEN
P_C_FORMAT = DEFAULT_PSTART_TOKEN + "{}" + DEFAULT_PEND_TOKEN + DEFAULT_CLS_TOKEN

FORMAT_DICT = {
    "phrase": (
        P_FORMAT,
        SHORT_QUESTIONS,
        LONG_QUESTIONS,
        EXPLANATORY_QUESTIONS,
        ANSWER_LIST,
        ANSWER_LIST_WITH_EXPLANATION,
    ),
    "cls": (
        C_FORMAT,
        SHORT_QUESTIONS,
        LONG_QUESTIONS,
        EXPLANATORY_QUESTIONS,
        ANSWER_LIST,
        ANSWER_LIST_WITH_EXPLANATION,
    ),
    "all": (
        P_C_FORMAT,
        SHORT_QUESTIONS,
        LONG_QUESTIONS,
        EXPLANATORY_QUESTIONS,
        ANSWER_LIST,
        ANSWER_LIST_WITH_EXPLANATION,
    ),
}


def reason_seg_conversations(labels, explain, is_sentence=True, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []
    sent_format, short_questions, long_questions, explanatory_questions, answer_list, explain_answer_list = (
        FORMAT_DICT[cond_type]
    )

    for i, label in enumerate(labels):
        question_template = random.choice(long_questions) if is_sentence else random.choice(short_questions)
        answer = random.choice(answer_list) if output_ids_with_output else ""
        if explain and random.random() < 0.5:
            question_template = random.choice(explanatory_questions)
            answer = (explain + " " + random.choice(ANSWER_LIST_WITH_EXPLANATION)) if output_ids_with_output else ""

        question = question_template.format(sent=sent_format.format(label.strip()))
        questions.append(question)
        answers.append(answer)

    rets = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            rets.append({"from": "human", "value": DEFAULT_IMAGE_TOKEN + question})
        else:
            rets.append({"from": "human", "value": question})
        rets.append({"from": "gpt", "value": answer})
    return rets


def reason_seg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    messages = reason_seg_conversations(
        example["sampled_sents"],
        example.get("explain", None),
        example.get("is_sentence", False),
        output_ids_with_output,
        cond_type,
    )
    input = ""
    conversation = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            if DEFAULT_IMAGE_TOKEN in msg["value"]:
                msg["value"] = msg["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                msg["value"] = DEFAULT_IMAGE_TOKEN + "\n" + msg["value"]
                msg["value"] = msg["value"].strip()
            input += msg["value"]

        elif msg["from"] == "gpt":
            conversation.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError
    example.update({"conversation": conversation})
    return example
