"""
server.py
Flask server connecting frontend to GD bot + Gemini Vision food classifier.
"""

import os
import pickle
import faiss
from flask import Flask, request, Response, send_from_directory, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from food_classifier import IndianFoodClassifier

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
CHAT_MODEL = "gemini-2.5-flash"

# -----------------------------
# LOAD PDF CHUNKS + FAISS
# -----------------------------
with open("chunks.pkl", "rb") as f:
    pdf_chunks = pickle.load(f)

pdf_index = faiss.read_index("vector.index")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_relevant_chunks(query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_tensor=False)
    D, I = pdf_index.search(q_emb, top_k)
    results = [pdf_chunks[i] for i in I[0] if i < len(pdf_chunks)]
    if not results:
        q = query.lower()
        results = [c for c in pdf_chunks if q in c.lower()][:3]
    return "\n".join(results)


# -----------------------------
# LOAD FOOD CLASSIFIER
# -----------------------------
try:
    food_classifier = IndianFoodClassifier()
    CLASSIFIER_AVAILABLE = True
    print("✅ Food classifier ready")
except Exception as e:
    print(f"⚠️  Food classifier not available: {e}")
    food_classifier = None
    CLASSIFIER_AVAILABLE = False

# -----------------------------
# SYSTEM PROMPT
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
- 10-15 minute walk after meals helps control sugar spikes
"""

SYSTEM_PROMPT = f"""
You are a caring and supportive assistant for women with Gestational Diabetes.

Answer clearly in complete short paragraphs. Never cut a sentence mid-way.
Do not mention insulin, medicines, or sugar numbers.
Only give a doctor warning if user asks about treatment or medication.

--- BASE KNOWLEDGE ---
{GD_DATASET}
"""

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, static_folder="public")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(".", "style.css")


@app.route("/public/<path:filename>")
def static_files(filename):
    return send_from_directory("public", filename)


# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return Response("Empty message", status=400)

    pdf_info = get_relevant_chunks(user_message)

    final_prompt = f"""
Use the following PDF information if relevant.

--- PDF INFORMATION ---
{pdf_info}

Question:
{user_message}
"""

    # Build Gemini conversation history
    gemini_history = []
    for msg in history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [{"text": msg["text"]}]})

    def generate():
        try:
            model = genai.GenerativeModel(
                model_name=CHAT_MODEL,
                system_instruction=SYSTEM_PROMPT
            )
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(
                final_prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 2048,
                },
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n---STREAMING ERROR---\n{str(e)}"

    return Response(generate(), mimetype="text/plain")


# -----------------------------
# IMAGE CLASSIFY ENDPOINT
# -----------------------------
@app.route("/api/classify", methods=["POST"])
def classify_image():
    if not CLASSIFIER_AVAILABLE:
        return jsonify({"success": False, "error": "Classifier not available"}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = f"temp_{file.filename}"

    try:
        file.save(temp_path)
        result = food_classifier.predict_image(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    print("\n🚀 GD Support Bot starting...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)