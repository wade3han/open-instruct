import json
import random

# datapath = "./megamixv2_dedup_tulu2mix-cot_train10k.jsonl"
datapath = "./megamixv2_dedup_wildchat-gpt-4-0125-preview_train10k.jsonl"
with open(datapath, "r") as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

while True:
    # randomly sample 16 from the dataset.
    # sample = random.sample(data, 16)
    sample = random.sample(data, 4)
    # convert the messages into the following format;
    # USER: <message>
    # ASSISTANT: <message>
    # USER: <message>
    # ASSISTANT: <message> ...

    example = "Here are the example of the conversations. Craft the single conversation similar to the given conversations.\n\n"
    for s in sample:
        messages = s["messages"]
        example += "### CONVERSATION\n"
        for m in messages:
            if m["role"] == "user":
                example += f"USER: {m['content']}\n"
            else:
                example += f"ASSISTANT: {m['content']}\n"
            example += "\n"
    example += "### Now, craft the single conversation similar to the given conversations, using the same format."

    with open("icl_examples.txt", "w") as f:
        f.write(example)

    input("Press Enter to continue...")
