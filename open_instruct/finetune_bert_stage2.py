import json
import random
import fire
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset, Dataset


# Load the IMDb dataset
def load_data(dataset_path: str) -> tuple[Dataset, Dataset]:
    # dataset_path is jsonl file.
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    # split the data into train and test
    random.seed(42)
    random.shuffle(data)
    train_data = data[: int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)) :]

    formatted_train_data = []
    formatted_test_data = []
    for item in train_data:
        # each item have statement, document, label.
        formatted_train_data.append(
            {
                "statement": item["statement"],
                "document": item["document"],
                "label": item["label"],
            }
        )
    for item in test_data:
        formatted_test_data.append(
            {
                "statement": item["statement"],
                "document": item["document"],
                "label": item["label"],
            }
        )

    return Dataset.from_list(formatted_train_data), Dataset.from_list(
        formatted_test_data
    )


def train(dataset_path: str, model_name: str, model_path: str):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    train_dataset, test_dataset = load_data(dataset_path)

    # Tokenization function
    def tokenize_function(example):
        statement = example["statement"]
        document = example["document"]
        text = tokenizer.eos_token.join([statement, document])
        return tokenizer(text, padding="max_length", truncation=True, max_length=512)

    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function)
    tokenized_test_dataset = test_dataset.map(tokenize_function)

    # Set the format for PyTorch
    tokenized_train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )  # label 1 is SUPPORTED, label 0 is NOT_SUPPORTED
    tokenized_test_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Create DataLoaders
    train_loader = DataLoader(tokenized_train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(tokenized_test_dataset, batch_size=8, shuffle=False)

    # Set up the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5
    )  # 1e-5 for the RoBERTa-large
    epochs = 1
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.03 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wandb.init(
        project="fact_verifier_small", entity="seungjuhan3", name=model_name
    )

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

            if training_step % 1000 == 0:
                # evaluate the model
                model.eval()
                accumulated_eval_loss = 0
                eval_loss_count = 0
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                        accumulated_eval_loss += loss.item()
                        eval_loss_count += 1

                # log the loss to wandb
                wandb.log(
                    {"eval_loss": accumulated_eval_loss / eval_loss_count},
                    step=training_step,
                )
                accumulated_eval_loss = 0
                eval_loss_count = 0

    # Save the fine-tuned model and tokenizer
    output_dir = f"./finetuned_roberta_{model_name}"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
