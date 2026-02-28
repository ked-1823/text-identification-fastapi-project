🚀 NLP Text Classification API (DistilBERT Fine-Tuned)

📌 Project Overview


This project is an end-to-end NLP text classification system built using:

DistilBERT (Transformer model)


# DistilBERT Text Classification App

🚀 Live Demo: https://huggingface.co/spaces/kedar18/distil-bert-model

This project deploys a DistilBERT model using Gradio on Hugging Face Spaces.

example prediction : 

<img width="1022" height="392" alt="image" src="https://github.com/user-attachments/assets/b8340a0c-5eaf-4e10-b4d8-270e2ae3a168" />

<img width="1275" height="467" alt="Screenshot 2026-02-28 102101" src="https://github.com/user-attachments/assets/e715e6e7-cc2b-4799-90e8-fa91a1be1ead" />
<img width="1044" height="461" alt="Screenshot 2026-02-28 102503" src="https://github.com/user-attachments/assets/548ca96c-0ba5-453a-81eb-951078a278ac" />
<img width="1232" height="507" alt="Screenshot 2026-02-28 102136" src="https://github.com/user-attachments/assets/6e128600-b57d-4a56-a96c-a65b87cab5ca" />



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

├── mental_health_text_detection.ipynb

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
