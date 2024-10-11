import json
import fire
import torch
import wandb
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset, Dataset

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForSequenceClassification.from_pretrained("roberta-large")


# Load the IMDb dataset
def load_data(dataset_path: str) -> Dataset:
    # dataset_path is jsonl file.
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    formatted_data = []
    for item in data:
        # each item have statement, document, label.
        formatted_data.append(
            {
                "text": f"{item['statement']}\n\n{item['document']}",
                "label": item["label"],
            }
        )

    return Dataset.from_list(formatted_data)


def train(dataset_path: str):
    dataset = load_data(dataset_path)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    # Tokenize the datasets
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set the format for PyTorch
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Create DataLoaders
    train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

    # Set up the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5
    )  # 1e-5 for the RoBERTa-large
    epochs = 2
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.03 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wandb.init(
        project="fact_verifier_small", entity="seungjuhan3", name="roberta-large"
    )

    # Training loop
    training_step = 0
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            training_step += 1
            optimizer.step()
            scheduler.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

            # log the loss to wandb
            wandb.log({"loss": loss.item()}, step=training_step)

    # Save the fine-tuned model and tokenizer
    output_dir = "./finetuned_roberta"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
