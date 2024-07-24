import argparse
import os

import torch


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    # search *.bin file in model_name directory
    for file in os.listdir(model_name):
        if file.endswith(".bin"):
            bin_path = os.path.join(model_name, file)
            new_state_dict = {}
            state_dict = torch.load(bin_path)
            import ipdb;
            ipdb.set_trace();
            for key, value in state_dict.items():
                new_key = remove_prefix(key, "_orig_mod.model.")
                new_state_dict[new_key] = value
            torch.save(new_state_dict, bin_path)
