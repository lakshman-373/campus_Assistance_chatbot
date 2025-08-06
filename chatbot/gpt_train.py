import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import pandas as pd
from tqdm import tqdm

class GPTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = self.samples.iloc[idx]['prompt']
        response = self.samples.iloc[idx]['response']
        full_text = f"{prompt} {self.tokenizer.eos_token} {response}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def compute_accuracy(preds, labels, ignore_token_id):
    correct = (preds == labels) & (labels != ignore_token_id)
    total = (labels != ignore_token_id).sum()
    acc = correct.sum().item() / total.item() if total.item() > 0 else 0
    return acc

def train_gpt(train_file, val_file, model_name='gpt2', epochs=10, batch_size=28, lr=3e-5, max_length=64):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    train_dataset = GPTDataset(train_data, tokenizer, max_length)
    val_dataset = GPTDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    ignore_token_id = -100  # GPT2 does not use this, but placeholder if used later

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0

        print(f"\n🚀 Epoch {epoch + 1}/{epochs} — Training")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != tokenizer.pad_token_id
            correct_tokens += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
        print(f"✅ Train Loss: {avg_train_loss:.4f} | 🎯 Train Accuracy: {train_acc:.4f}")

        # 🔍 Validation
        model.eval()
        val_loss = 0
        val_correct_tokens = 0
        val_total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="🔎 Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                val_loss += outputs.loss.item()

                predictions = torch.argmax(logits, dim=-1)
                mask = labels != tokenizer.pad_token_id
                val_correct_tokens += ((predictions == labels) & mask).sum().item()
                val_total_tokens += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct_tokens / val_total_tokens if val_total_tokens > 0 else 0
        print(f"📉 Val Loss: {avg_val_loss:.4f} | 🎯 Val Accuracy: {val_acc:.4f}")

    # ✅ Save final model
    model.save_pretrained("/content/drive/MyDrive/chatbot1/gpt_finetuned")
    tokenizer.save_pretrained("/content/drive/MyDrive/chatbot1/gpt_finetuned")
    print("✅ GPT model and tokenizer saved to 'gpt_finetuned/'")

# ---- Run ---- #
if __name__ == "__main__":
    train_gpt(
        train_file="/content/drive/MyDrive/chatbot1/gpt_dataset_train.csv",
        val_file="/content/drive/MyDrive/chatbot1/gpt_dataset_val.csv"
    )