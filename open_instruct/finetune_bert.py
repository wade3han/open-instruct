import json
import random
import fire
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate.utils import set_seed
from datasets import Dataset

# Load the tokenizer and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load the IMDb dataset
def load_data(dataset_path: str) -> Dataset:
    # dataset_path is jsonl file.
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    formatted_train_data = []
    formatted_test_data = []
    for item in data:
        # each item have statement, document, label.
        formatted_train_data.append(
            {
                "statement": item["statement"],
                "document": item["document"],
                "label": item["label"],
            }
        )

    return Dataset.from_list(formatted_train_data)


def train(
    dataset_path: str,
    model_name: str,
    lr: float = 1e-5,
    batch_size: int = 8,
    num_epochs: int = 2,
    use_lora: bool = False,
    use_debug: bool = False,
):
    set_seed(42)
    train_dataset = load_data(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=128,
            task_type="SEQ_CLS",
            lora_alpha=512,
            lora_dropout=0.1,
            target_modules=[
                "query_proj",
                "key_proj",
                "value_proj",
                "attention.output.dense",
            ],  # for deberta-v3-large
            modules_to_save=["classifier", "pooler"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large", num_labels=2
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large", num_labels=2
        )

    # Tokenization function
    def tokenize_function(example):
        statement = example["statement"]
        document = example["document"]
        text = tokenizer.eos_token.join([document, statement])
        return tokenizer(text, padding="max_length", truncation=True, max_length=512)

    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function)

    # Set the format for PyTorch
    tokenized_train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )  # label 1 is SUPPORTED, label 0 is NOT_SUPPORTED

    # Create DataLoaders
    train_loader = DataLoader(
        tokenized_train_dataset, batch_size=batch_size, shuffle=True
    )

    # Set up the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )  # 1e-5 for the RoBERTa-large
    # 5e-5 for the Deberta-v3-large
    epochs = num_epochs
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wandb.init(project="fact_verifier_small", entity="seungjuhan3", name=model_name)

    # Training loop
    training_step = 0
    accumulated_loss = 0
    loss_count = 0
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            model.train()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            training_step += 1
            loss_count += 1
            accumulated_loss += loss.item()
            optimizer.step()
            scheduler.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=accumulated_loss / loss_count)

            # log the loss to wandb
            if training_step % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "loss": accumulated_loss / loss_count,
                        "lr": lr,
                        "grad_norm": grad_norm,
                    },
                    step=training_step,
                )
                accumulated_loss = 0
                loss_count = 0
                
            if use_debug:
                break

    # Save the fine-tuned model and tokenizer
    output_dir = f"./finetuned_deberta_{model_name}"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # # save optimizer state
    optimizer_state_dict = optimizer.state_dict()
    torch.save(optimizer_state_dict, f"{output_dir}/optimizer.pt")
    print(f"Optimizer state saved to {output_dir}/optimizer.pt")

    # save the optimizer_idx_to_param_name.
    param_id_name_dict = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    optimizer_params = [p for p_group in optimizer.param_groups for p in p_group['params']]

    optimizer_model_param_dict = {}
    for idx, param in enumerate(optimizer_params):
        optimizer_model_param_dict[idx] = param_id_name_dict[id(param)]

    with open(f"{output_dir}/optimizer_map.json", "w") as f:
        json.dump(optimizer_model_param_dict, f)

    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
