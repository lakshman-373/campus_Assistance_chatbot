import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
import json
import random
import re
import csv
import os
import pandas as pd
from datetime import datetime
from googlesearch import search

# ----------- Load Models ----------- #
bert_model_path = "/content/drive/MyDrive/cbt/chatbot1/bert_intent_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_model.eval()

gpt_model_path = "/content/drive/MyDrive/cbt/chatbot1/gpt_finetuned"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_path)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.eval()

with open("/content/drive/MyDrive/cbt/chatbot1/bert_dataset_label_map.json", "r") as f:
    label_map = json.load(f)
id_to_label = {v: k for k, v in label_map.items()}

with open("/content/drive/MyDrive/cbt/chatbot1/data/intents_expanded.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)
intent_response_map = {
    item["intent"]: item.get("responses", []) for item in intents_data.get("intents", [])
}

chat_log_file = "/content/drive/MyDrive/cbt/chatbot1/chat_logs.csv"
if not os.path.exists(chat_log_file):
    with open(chat_log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_input", "predicted_intent", "confidence", "response", "top_intents"])

# ----------- Load Student Database ----------- #
student_csv_path = "/content/drive/MyDrive/cbt/chatbot1/students_utf8.csv"
student_df = pd.read_csv(student_csv_path, encoding='utf-8')

# ----------- Helpers ----------- #
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()

def predict_intent(text, top_k=3):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        topk_confidences, topk_indices = torch.topk(probs, k=top_k, dim=1)
        results = [(id_to_label[idx.item()], conf.item()) for idx, conf in zip(topk_indices[0], topk_confidences[0])]
        return results[0][0], results[0][1], results

def generate_gpt_response(user_input, intent="unknown"):
    prompt = f"[Intent: {intent}]\nUser: {user_input}\nBot:"
    inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = gpt_model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=gpt_tokenizer.eos_token_id,
        )
    generated = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.split("Bot:")[-1].strip()

def log_chat(user_input, intent, confidence, response, top_intents=None):
    with open(chat_log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        top_intent_str = "; ".join([f"{i}:{c:.2f}" for i, c in top_intents]) if top_intents else ""
        writer.writerow([datetime.now().isoformat(), user_input, intent, f"{confidence:.2f}", response, top_intent_str])

# ----------- PDF Search Handler ----------- #
def search_pdf_link(query):
    search_query = f"{query} filetype:pdf"
    try:
        for url in search(search_query, num_results=5):
            if url.endswith(".pdf"):
                formatted_response = f"📄 Here's a PDF I found: <a href='{url}' target='_blank' rel='noopener noreferrer'>{url}</a>"
                print("DEBUG: PDF Response:", formatted_response)  # Debugging line
                return formatted_response
    except Exception as e:
        return f"⚠ Error fetching PDFs: {str(e)}"
    return "❌ Couldn't find a relevant PDF right now."


# ----------- Student Info Handler ----------- #
import os

def get_student_info(query_or_id, info_type=None):
    query = str(query_or_id).lower()

    for _, row in student_df.iterrows():
        name_match = str(row.get('Name', '')).lower() in query
        id_match = str(row.get('University No', '')).lower() in query

        if name_match or id_match:
            student_id = row.get('University No', 'N/A')
            student_name = row.get('Name', 'N/A')
            details = []

            # Student Photo Handling
            photo_filename = f"{student_id}.jpg"
            photo_path = f"static/student_photos/{photo_filename}"  # Local path
            web_photo_path = f"/static/student_photos/{photo_filename}"  # Web-accessible path

            if os.path.exists(photo_path):
                photo_html = f'<img src="{web_photo_path}" alt="{student_name}" width="150" height="150"><br>'
            else:
                photo_html = "🚫 No photo available.<br>"

            # Student Details
            details.append(f"👤 Name: {student_name}")
            details.append(f"🆔 University No: {student_id}")

            # Check for specific queries
            if any(keyword in query for keyword in ["name", "id", "university", "dob", "birth", "gender", "email", "contact", "phone", "branch", "cgpa", "gpa", "percentage", "aggregate", "city", "pincode", "father"]):
                if 'dob' in query or 'birth' in query:
                    details.append(f"🎂 Date of Birth: {row.get('Date of birth', 'N/A')}")
                if 'gender' in query:
                    details.append(f"⚧ Gender: {row.get('Gender', 'N/A')}")
                if 'email' in query:
                    details.append(f"📧 Email: {row.get('Email ID', 'N/A')}")
                if 'contact' in query or 'phone' in query:
                    details.append(f"📱 Mobile Number: {row.get('Mobile Number', 'N/A')}")
                if 'branch' in query:
                    details.append(f"📚 Branch: {row.get('Branch', 'N/A')}")
                if 'cgpa' in query or 'gpa' in query:
                    details.append(f"📊 Aggregate CGPA: {row.get('cgpa', 'N/A')}")
                if 'percentage' in query or 'aggregate' in query:
                    details.append(f"📈 Aggregate % (Graduation): {row.get('percentage', 'N/A')}")
                if 'city' in query:
                    details.append(f"🌍 City: {row.get('City', 'N/A')}")
                if 'pincode' in query:
                    details.append(f"📍 Pincode: {row.get('Pincode', 'N/A')}")
                if 'father' in query:
                    details.append(f"👨‍👦 Father's Name: {row.get('Father Name', 'N/A')}")
            else:
                # Return full details if no specific keyword is found
                details.extend([
                    f"🎂 Date of Birth: {row.get('Date of birth', 'N/A')}",
                    f"⚧ Gender: {row.get('Gender', 'N/A')}",
                    f"📧 Email: {row.get('Email ID', 'N/A')}",
                    f"📱 Mobile Number: {row.get('Mobile Number', 'N/A')}",
                    f"📚 Branch: {row.get('Branch', 'N/A')}",
                    f"📊 Aggregate CGPA: {row.get('cgpa', 'N/A')}",
                    f"📈 Aggregate % (Graduation): {row.get('percentage', 'N/A')}",
                    f"🌍 City: {row.get('City', 'N/A')}",
                    f"📍 Pincode: {row.get('Pincode', 'N/A')}",
                    f"👨‍👦 Father's Name: {row.get('Father Name', 'N/A')}"
                ])

            return photo_html + "<br>".join(details)  # Photo (or no-photo message) appears first!

    return "⚠ Sorry, I couldn't find that student or info."


# ----------- Main Chat Function ----------- #
def chatbot_response(user_input, confidence_threshold=0.5):
    cleaned_input = clean_text(user_input)

    # 🎯 PDF Request
    if "pdf" in cleaned_input:
        return search_pdf_link(cleaned_input)

    # 🎓 Student Info
    if "student" in cleaned_input :
        response = get_student_info(cleaned_input)
        log_chat(user_input, "student_info", 1.0, response, [("student_info", 1.0)])
        return response

    # 🤖 General Chat
    intent, confidence, top_intents = predict_intent(cleaned_input)
    fallback = (
        confidence < confidence_threshold or
        intent not in intent_response_map or
        not intent_response_map[intent]
    )

    if fallback:
        response = generate_gpt_response(cleaned_input, intent)
    else:
        response = random.choice(intent_response_map[intent])

    log_chat(user_input, intent, confidence, response, top_intents)
    return response
if __name__ == "_main_":
    user_input = input("You: ")  # Take user input
    response = chatbot_response(user_input)
    print("Bot:", response)