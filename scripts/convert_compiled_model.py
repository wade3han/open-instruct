import argparse
import json
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
            for key, value in state_dict.items():
                new_key = remove_prefix(key, "_orig_mod.model.")
                new_key = remove_prefix(new_key, "_orig_mod.")
                new_state_dict[new_key] = value
            torch.save(new_state_dict, bin_path)
            print(f"Converted {bin_path}")
        elif file.endswith("index.json"):
            index_path = os.path.join(model_name, file)
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            new_weight_map = {}
            for key, value in weight_map.items():
                new_key = remove_prefix(key, "_orig_mod.model.")
                new_key = remove_prefix(new_key, "_orig_mod.")
                new_weight_map[new_key] = value
            index["weight_map"] = new_weight_map

            with open(index_path, "w") as f:
                json.dump(index, f)
                print(f"Converted {index_path}")
