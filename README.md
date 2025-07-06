# 🧠 Mental Health Copilot (CBT-based AI Chatbot)

This is a GenAI-powered chatbot that supports mental health journaling and real-time emotional guidance using CBT techniques. Built with:

- 🧠 Mistral-7B-Instruct (via llama.cpp)
- 🧪 Sentiment analysis with RoBERTa
- ⚡ FastAPI backend
- 💬 React frontend

## 🚀 How to Run

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source vvenv\Scripts\activate 
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend (ReactJS)

```bash
cd frontend
npm install
npm start
