import json
import os

import pandas as pd


def get_mmlu_dataset(data_dir: str, ):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    k = 5
    dataset = []
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[: k]
        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]
            dataset.append(
                {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
            )
    return dataset


def get_bbh_dataset(data_dir: str):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = []

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i + 1:]
            icl = form_icl(other_exes)
            question, answer = target_ex.split("\nA:")

            dataset.append(
                {
                    "messages": [
                        {"role": "user", "content": f"{task_prompt.strip()}\n\n{icl}{question}\nA:"},
                        {"role": "assistant", "content": f"{answer}"}
                    ]
                }
            )

    return dataset


def get_tydiqa_dataset(data_dir: str, zh: bool = False):
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": (
            "Answer the following question based on the information in the given passage.", "Passage:", "Question:",
            "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": (
            "প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": (
            "Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": (
            "Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:",
            "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": (
            "Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": (
            "Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa", file_name)

    examples = json.load(open(file, "r"))
    dataset = []

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
                  format(example["context"]) + "\n" + q_template + \
                  " " + format(example["question"]) + "\n"
        answer = " " + format(example["answers"][0]["text"])
        dataset.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]})
    return dataset


def get_gsm8k_dataset(data_dir: str):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/test.jsonl"
    data = [json.loads(line) for line in open(file, "r")]
    dataset = [{"messages": [{"role": "user", "content": example["context"]},
                             {"role": "assistant", "content": example["response"]}]} for example in data]

    return dataset


if __name__ == "__main__":
    # data = get_mmlu_dataset("/net/nfs.cirrascale/mosaic/seungjuh/LESS/data")
    # with open("/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/mmlu.jsonl", "w") as f:
    #     for example in data:
    #         f.write(json.dumps(example) + "\n")

    # data = get_bbh_dataset("/net/nfs.cirrascale/mosaic/seungjuh/LESS/data")
    # with open("/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/bbh.jsonl", "w") as f:
    #     for example in data:
    #         f.write(json.dumps(example) + "\n")

    # data = get_tydiqa_dataset("/net/nfs.cirrascale/mosaic/seungjuh/LESS/data")
    # with open("/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/tydiqa.jsonl",
    #           "w") as f:
    #     for example in data:
    #         f.write(json.dumps(example) + "\n")

    data = get_gsm8k_dataset("/net/nfs.cirrascale/mosaic/seungjuh/OE-Safety/evaluation/tasks/generation/gsm8k")
    with open("/net/nfs.cirrascale/mosaic/seungjuh/open-instruct-general/open_instruct/gradient/gsm8k.jsonl", "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")
