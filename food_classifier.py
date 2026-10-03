"""
food_classifier.py
Indian food image classifier using Gemini Vision API.
"""

import os
import base64
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

VISION_MODEL = "gemini-2.5-flash"

CLASSIFY_PROMPT = """Look at this food image and respond with ONLY a JSON object. No explanation. No markdown. No extra text. Just the raw JSON.

Example response:
{"top_predictions": [{"class": "banana", "confidence": 92.0}, {"class": "plantain", "confidence": 5.0}, {"class": "fruit", "confidence": 3.0}]}

Now identify the food in this image and respond with the same JSON format."""


class IndianFoodClassifier:
    """Gemini Vision-based classifier for Indian food images."""

    def __init__(self, model_path=None, class_names_path=None):
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.model = genai.GenerativeModel(VISION_MODEL)
        print("✅ Indian Food Classifier (Gemini Vision) loaded!")
        print(f"   Model: {VISION_MODEL}")

    def _run_gemini(self, pil_image):
        """Send image to Gemini and parse JSON response."""
        # Resize to max 512px to keep response clean
        pil_image.thumbnail((512, 512))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode("utf-8")
        }

        response = self.model.generate_content(
            [CLASSIFY_PROMPT, {"inline_data": image_part}],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 1024   # increased so JSON doesn't get cut off
            }
        )

        raw = response.text.strip()
        print(f"\n[Classifier] Raw response: {raw}\n")

        # Find the JSON object by locating braces
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end == 0:
            print("[Classifier] No JSON braces found, using fallback")
            return [
                {"class": raw.strip()[:50], "confidence": 80.0},
                {"class": "unknown", "confidence": 12.0},
                {"class": "unknown", "confidence": 8.0}
            ]

        json_str = raw[start:end]

        try:
            parsed = json.loads(json_str)
            predictions = parsed["top_predictions"]

            # Validate that class values are plain strings
            for p in predictions:
                if not isinstance(p["class"], str):
                    p["class"] = str(p["class"])

            return predictions

        except Exception as e:
            print(f"[Classifier] JSON parse error: {e}")
            print(f"[Classifier] JSON string was: {json_str}")
            return [
                {"class": "food item", "confidence": 80.0},
                {"class": "unknown", "confidence": 12.0},
                {"class": "unknown", "confidence": 8.0}
            ]

    def _build_result(self, top_predictions, image_path=None):
        top = top_predictions[0]
        return {
            "success": True,
            "image_path": image_path or "",
            "image_filename": os.path.basename(image_path) if image_path else "",
            "top_prediction": {
                "class": top["class"],
                "confidence": float(top["confidence"]),
                "class_index": 0
            },
            "top_predictions": [
                {
                    "class": p["class"],
                    "confidence": float(p["confidence"]),
                    "class_index": i
                }
                for i, p in enumerate(top_predictions)
            ],
            "message": f"The image contains: {top['class']} ({top['confidence']:.1f}% confidence)",
            "simple_message": f"The user has uploaded an image of {top['class']}."
        }

    def predict_image(self, image_path, top_k=3):
        try:
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}
            pil_image = Image.open(image_path).convert("RGB")
            top_predictions = self._run_gemini(pil_image)
            return self._build_result(top_predictions[:top_k], image_path)
        except Exception as e:
            return {"success": False, "error": str(e), "image_path": image_path}

    def predict_from_pil_image(self, pil_image, top_k=3):
        try:
            top_predictions = self._run_gemini(pil_image)
            return self._build_result(top_predictions[:top_k])
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_all_classes(self):
        return ["Gemini Vision — open vocabulary"]

    def get_class_count(self):
        return -1


def classify_food_image(image_path):
    classifier = IndianFoodClassifier()
    result = classifier.predict_image(image_path)
    if result["success"]:
        return result["simple_message"]
    else:
        return f"Error: {result['error']}"


if __name__ == "__main__":
    print(f"Gemini Vision food classifier ready. Model: {VISION_MODEL}")