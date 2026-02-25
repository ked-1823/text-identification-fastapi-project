🚀 NLP Text Classification API (DistilBERT Fine-Tuned)
📌 Project Overview

This project is an end-to-end NLP text classification system built using:

DistilBERT (Transformer model)

PyTorch

HuggingFace Transformers

FastAPI

Uvicorn

The model was fine-tuned on a labeled dataset and deployed as a REST API for real-time inference.

🧠 1️⃣ Model Training
🔹 Base Model

Pre-trained model:
Hugging Face
Model architecture:
DistilBERT

Why DistilBERT?

Smaller and faster than BERT

Good balance of performance and speed

Suitable for production APIs

🔹 Training Pipeline

Load dataset

Convert to HuggingFace Dataset

Tokenization using DistilBertTokenizerFast

Convert text to input IDs + attention masks

Fine-tune DistilBertForSequenceClassification

Evaluate using classification_report from
scikit-learn

🔹 Evaluation

Model performance evaluated using:

Precision

Recall

F1-score

Accuracy

Generated using:

classification_report()
⚙️ 2️⃣ API Development

Framework used:
FastAPI

Server:
Uvicorn

Endpoint
POST /predict
Request
{
  "text": "I feel amazing today"
}
Response
{
  "prediction": "Positive"
}
🌐 3️⃣ Running Locally

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn main:app --reload

Open Swagger docs:

http://127.0.0.1:8000/docs

☁️ 4️⃣ Deployment

Deployed using:
Render

Production command:

uvicorn main:app --host 0.0.0.0 --port 10000

After deployment, the API becomes publicly accessible.

📂 Project Structure
text_nlp/
│
├── train.py
├── main.py
├── requirements.txt
├── README.md
🔥 What This Project Demonstrates

✔ Fine-tuning transformer models
✔ Model evaluation and metrics
✔ Converting ML model into REST API
✔ Understanding ASGI architecture
✔ Cloud deployment workflow
✔ Real-world ML system design

🚀 Future Improvements

Add model versioning

Add Docker containerization

Add CI/CD pipeline

Add frontend interface (HTML / Streamlit)

Add logging and monitoring

Add batch prediction endpoint
