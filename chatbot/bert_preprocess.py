import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def preprocess_bert_data(input_file, output_file_prefix):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    labels = []
    label_map = {}
    label_counter = 0

    for item in data.get("intents", []):
        intent = item.get("intent")
        examples = item.get("examples") or item.get("patterns", [])
        if not intent or not isinstance(examples, list):
            print(f"[!] Skipping invalid entry: {item}")
            continue

        if intent not in label_map:
            label_map[intent] = label_counter
            label_counter += 1

        for example in examples:
            cleaned = clean_text(example)
            sentences.append(cleaned)
            labels.append(label_map[intent])

    df = pd.DataFrame({"text": sentences, "label": labels})

    # Check if stratified split is possible
    num_classes = len(label_map)
    test_size = 0.2
    min_required_test_samples = num_classes
    total_samples = len(df)
    test_samples = int(test_size * total_samples)

    if test_samples >= min_required_test_samples:
        stratify_option = df["label"]
    else:
        stratify_option = None
        print(f"[⚠] Not enough samples to stratify across {num_classes} classes. Proceeding without stratification.")

    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=stratify_option
    )

    # Save CSVs
    train_df.to_csv(f"{output_file_prefix}_train.csv", index=False)
    val_df.to_csv(f"{output_file_prefix}_val.csv", index=False)

    # Save label map
    with open(f"{output_file_prefix}_label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    print("[✅] BERT preprocessing complete!")
    print(f"[📄] Train data saved to: {output_file_prefix}_train.csv")
    print(f"[📄] Validation data saved to: {output_file_prefix}_val.csv")
    print(f"[🧾] Label map saved to: {output_file_prefix}_label_map.json")

# ---- Run via CLI ---- #
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bert_preprocess.py <input_json> <output_prefix>")
        sys.exit(1)

    input_file = "/content/drive/MyDrive/chatbot1/data/intents_paraphrased.json"
    output_prefix = "/content/drive/MyDrive/chatbot1/bert_dataset"
    preprocess_bert_data(input_file, output_prefix)