"""
app.py
Terminal-based Gestational Diabetes Support Bot
with Indian Food Image Classification integrated
"""

import os
import pickle
import faiss
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from food_classifier import IndianFoodClassifier

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

# -----------------------------
# LOAD PDF CHUNKS + FAISS
# -----------------------------
with open("chunks.pkl", "rb") as f:
    pdf_chunks = pickle.load(f)

pdf_index = faiss.read_index("vector.index")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = ThreadPoolExecutor()

# -----------------------------
# LOAD FOOD CLASSIFIER
# -----------------------------
try:
    food_classifier = IndianFoodClassifier()
    CLASSIFIER_AVAILABLE = True
except FileNotFoundError as e:
    print(f"⚠️  Food classifier not available: {e}")
    print("   Image classification will be disabled.\n")
    food_classifier = None
    CLASSIFIER_AVAILABLE = False

# -----------------------------
# IMAGE INPUT HELPER
# -----------------------------
def try_classify_image(user_input: str):
    """
    If the user's message looks like a file path to an image,
    classify it and return the food label + confidence.
    Returns None if input is not an image path.
    """
    if not CLASSIFIER_AVAILABLE:
        return None

    stripped = user_input.strip().strip('"').strip("'")
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

    if not stripped.lower().endswith(image_extensions):
        return None

    if not os.path.exists(stripped):
        print(f"⚠️  Image file not found: {stripped}")
        return None

    result = food_classifier.predict_image(stripped)
    if result["success"]:
        return result
    else:
        print(f"⚠️  Could not classify image: {result.get('error')}")
        return None

# -----------------------------
# RAG RETRIEVAL
# -----------------------------
def get_relevant_chunks(query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_tensor=False)
    D, I = pdf_index.search(q_emb, top_k)
    results = [pdf_chunks[i] for i in I[0] if i < len(pdf_chunks)]

    if not results:
        q = query.lower()
        results = [c for c in pdf_chunks if q in c.lower()][:3]

    return "\n".join(results)

# -----------------------------
# BASE KNOWLEDGE
# -----------------------------
GD_DATASET = """
Gestational Diabetes Diet Guidance:

Carbohydrates:
- Brown rice (small portions)
- Whole wheat roti
- Millets
- Oats

Fruits:
- Fruits can be eaten in limited portions
- Prefer whole fruits instead of juices
- Mango, banana, chikoo should be eaten occasionally
- Pair fruits with nuts or curd

Lifestyle:
- 10–15 minute walk after meals helps control sugar spikes
"""

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = f"""
You are a caring and supportive assistant for women with Gestational Diabetes.

Answer clearly in short paragraphs.
Do not cut sentences.
Do not mention insulin, medicines, or sugar numbers.

Only give doctor warning if user asks about treatment or medication.

--- BASE KNOWLEDGE ---
{GD_DATASET}
"""

# -----------------------------
# TERMINAL CHAT LOOP
# -----------------------------
async def chat_loop():
    print("\n🤰 Gestational Diabetes Support Bot")
    if CLASSIFIER_AVAILABLE:
        print("📷  You can also type an image file path to analyse a food photo.")
    print("Ask food-related questions. Type 'exit' to quit.\n")

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT
    )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("\nTake care 🌸 Stay healthy.")
            break

        # ── Image path? Classify it first ──────────────────────────────
        image_result = try_classify_image(user_input)

        if image_result:
            top = image_result["top_prediction"]
            food_name = top["class"]
            confidence = top["confidence"]

            print(f"\n🍽️  Detected food: {food_name} ({confidence:.1f}% confidence)")

            # Show runner-up predictions for transparency
            others = image_result["top_predictions"][1:]
            if others:
                runner_ups = ", ".join(
                    f"{p['class']} ({p['confidence']:.1f}%)" for p in others
                )
                print(f"   Other possibilities: {runner_ups}")

            # Build a natural-language question from the classification result
            effective_query = (
                f"I have {food_name}. Is it safe for gestational diabetes? "
                f"How much can I eat and what precautions should I take?"
            )
            print(f"\n💬 Asking about: {effective_query}\n")

        else:
            # Plain text question
            effective_query = user_input

        # ── RAG retrieval ───────────────────────────────────────────────
        pdf_info = get_relevant_chunks(effective_query)

        final_prompt = f"""
Use the following PDF information if relevant.

--- PDF INFORMATION ---
{pdf_info}

Question:
{effective_query}
"""

        try:
            response = model.generate_content(
                final_prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 2048,
                }
            )

            final_text = response.text.strip()

            # Only append period if text ends mid-word (not mid-sentence)
            if final_text and not final_text.endswith((".", "!", "?", ":", "…")):
                # Find last complete sentence
                for punct in (".", "!", "?"):
                    last = final_text.rfind(punct)
                    if last != -1:
                        final_text = final_text[:last + 1]
                        break

            print("\nBot:\n")
            print(final_text)
            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(chat_loop())