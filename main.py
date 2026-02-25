# -----------------------------
# FastAPI Mental Health API
# -----------------------------

# FastAPI framework
from fastapi import FastAPI

# Pydantic for request validation
from pydantic import BaseModel

# HuggingFace transformers for model + tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PyTorch for tensor operations
import torch
import torch.nn.functional as F

# OS + Google Drive downloader
import os
import gdown


# ----------------------------------
# 1️⃣ Create FastAPI App Instance
# ----------------------------------
app = FastAPI()


# ----------------------------------
# 2️⃣ Google Drive Model Setup
# ----------------------------------

# Google Drive File ID (ONLY for model.safetensors)
MODEL_FILE_ID = "18l_WN5jhZUvbVwFbsh7K1DFzjOicnQmL"

# Model file name
MODEL_FILE = "model.safetensors"

# If model file does NOT exist locally → download it
if not os.path.exists(MODEL_FILE):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)


# ----------------------------------
# 3️⃣ Load Tokenizer & Model
# ----------------------------------
# IMPORTANT:
# config.json, tokenizer.json, tokenizer_config.json
# must exist in the root directory (GitHub repo)

tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForSequenceClassification.from_pretrained(".")
model.eval()  # Set model to evaluation mode


# ----------------------------------
# 4️⃣ Request Schema
# ----------------------------------
class TextInput(BaseModel):
    text: str


# ----------------------------------
# 5️⃣ Root Endpoint
# ----------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Mental Health Detection API!"}


# ----------------------------------
# 6️⃣ Label Mapping
# ----------------------------------
label = {
    0: "Normal",
    1: "Depression",
    2: "Suicidal",
    3: "Anxiety"
}


# ----------------------------------
# 7️⃣ Prediction Endpoint
# ----------------------------------
@app.post("/predict")
def predict(data: TextInput):

    # Tokenize input text
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Disable gradients (faster inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits
    logits = outputs.logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Get predicted class index
    prediction = torch.argmax(probs, dim=1).item()

    # Get confidence score
    confidence = probs[0][prediction].item()

    # Convert class index to label
    prediction_label = label[prediction]

    # Build probability dictionary
    probabilities = {
        label[i]: round(probs[0][i].item(), 2)
        for i in label
    }

    # Alert logic
    alert = False
    if prediction_label == "Suicidal" and confidence > 0.70:
        alert = True

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "alert": alert
    }