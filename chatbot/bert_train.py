import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import os

# Config
MAX_LENGTH = 128
BATCH_SIZE = 28
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):  # ✅ Correct init
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # ✅ Correct len
        return len(self.texts)

    def __getitem__(self, idx):  # ✅ Correct getitem
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = BertIntentDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)


def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(data_loader), acc


def eval_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(data_loader), acc


def main():
    # Load data
    train_df = pd.read_csv("/content/drive/MyDrive/chatbot1/bert_dataset_train.csv")
    val_df = pd.read_csv("/content/drive/MyDrive/chatbot1/bert_dataset_val.csv")

    # Load label map
    with open("/content/drive/MyDrive/chatbot1/bert_dataset_label_map.json", "r") as f:
        label_map = json.load(f)

    num_labels = len(label_map)

    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.to(DEVICE)

    # Dataloaders
    train_loader = create_data_loader(train_df, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_loader = create_data_loader(val_df, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n🧪 Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = eval_model(model, val_loader)
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    model.save_pretrained("/content/drive/MyDrive/chatbot1/bert_intent_model")
    tokenizer.save_pretrained("/content/drive/MyDrive/chatbot1/bert_intent_model")

    print("\n✅ Model and tokenizer saved to bert_intent_model/")