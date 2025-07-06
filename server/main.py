from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from llama_cpp import Llama

# Load once during app init
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    temperature=0.7,
    stop=["</s>", "User:", "Bot:"]
)



emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Store in memory (replace with DB later)
JOURNAL_ENTRIES = []

class JournalEntry(BaseModel):
    text: str

@app.post("/journal")
def analyze_journal(entry: JournalEntry):
    emotions = emotion_model(entry.text)
    embedding = embedding_model.encode(entry.text).tolist()
    JOURNAL_ENTRIES.append({
        "text": entry.text,
        "timestamp": str(datetime.datetime.now()),
        "emotions": emotions,
        "embedding": embedding
    })
    return {
        "message": "Journal received",
        "emotions": emotions
    }

@app.get("/journal")
def get_all_entries():
    return JOURNAL_ENTRIES

@app.post("/cbt-prompt")
def get_cbt_prompt(entry: JournalEntry):
    text = entry.text.lower()

    # Rule-based logic
    if "failure" in text or "useless" in text or "worthless" in text:
        return {"prompt": "What evidence supports this thought? Is there an alternative way to look at it?"}
    elif "anxious" in text or "overwhelmed" in text:
        return {"prompt": "What's one thing you can control right now? How can you ground yourself in the present?"}
    elif "angry" in text:
        return {"prompt": "What boundary do you feel has been crossed? Can you express this need assertively?"}
    else:
        return {"prompt": "What is one positive thing that happened today, even if it's small?"}

@app.post("/chat")
def chat_response(entry: JournalEntry):
    text = entry.text.lower()
    emotions = emotion_model(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])['label'].lower()

    # Optional: Collect journal history context
    history_context = ""
    if JOURNAL_ENTRIES:
        recent_entries = JOURNAL_ENTRIES[-3:]
        history_context = "\n".join([f"- {e['text']}" for e in recent_entries])

    # Save current entry
    JOURNAL_ENTRIES.append({
        "text": entry.text,
        "timestamp": str(datetime.datetime.now()),
        "emotions": emotions,
        "embedding": embedding_model.encode(text).tolist()
    })

    # Prepare prompt for Mistral
    prompt = f"""
    You are an empathetic AI mental health assistant trained in CBT (Cognitive Behavioral Therapy).
    Your job is to reply kindly and helpfully to the user using CBT strategies.
    The user seems to be feeling: {top_emotion}.

    Recent journal entries:
    {history_context if history_context else "No history yet."}

    Current user message:
    "{text}"

    Generate a therapeutic response. Avoid diagnosing. Be supportive, reflective, and emotionally intelligent.
    """

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
        stop=["<|user|>", "<|system|>", "<|assistant|>"]
    )
    response_text = response["choices"][0]["text"].strip()

    return {
        "emotion": top_emotion,
        "reply": response_text
    }
