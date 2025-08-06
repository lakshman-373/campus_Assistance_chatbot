import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_gpt_data(input_file, output_file_prefix):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    responses = []

    for item in data.get("intents", []):
        intent = item.get("intent", "").strip()
        all_examples = item.get("examples", [])
        all_responses = item.get("responses", [])

        # Clean and validate
        examples = [clean_text(ex) for ex in all_examples if isinstance(ex, str) and ex.strip()]
        responses_list = [clean_text(r) for r in all_responses if isinstance(r, str) and r.strip()]

        if not intent or not examples or not responses_list:
            print(f"[!] Skipping intent '{intent}' due to missing data.")
            continue

        for example in examples:
            for response in responses_list:
                prompts.append(example)
                responses.append(response)

        print(f"[✓] Intent: {intent} | Examples: {len(examples)} | Responses: {len(responses_list)}")

    if not prompts or not responses:
        print("❌ No valid prompt-response pairs found.")
        return

    df = pd.DataFrame({"prompt": prompts, "response": responses})

    if len(df) < 2:
        print("❌ Not enough data to split.")
        return

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = f"{output_file_prefix}_train.csv"
    val_path = f"{output_file_prefix}_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("✅ GPT data preprocessing complete.")
    print(f"📄 Train file: {train_path}")
    print(f"📄 Val file:   {val_path}")

# ---- Entry Point ----
if __name__ == "__main__":
    input_file = "/content/drive/MyDrive/chatbot1/data/intents_paraphrased.json"
    output_file_prefix = "/content/drive/MyDrive/chatbot1/gpt_dataset"
    preprocess_gpt_data(input_file, output_file_prefix)

if __name__ == "__main__":
    main()    