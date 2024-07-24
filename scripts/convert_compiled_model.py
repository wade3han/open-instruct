import argparse

from transformers import AutoModelForCausalLM


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = model.state_dict()

    import ipdb;
    ipdb.set_trace();

    # Remove prefix "_orig_mod." from the keys.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = remove_prefix(key, "_orig_mod.")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.save_pretrained(model_name)
