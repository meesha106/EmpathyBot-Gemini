import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import PyPDF2

# --------- Extract text from PDF ---------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Load your PDFs
pdf1 = "Gestational-Diabetes-Mellitus.pdf"
pdf2 = "Dietary Guidelines ICMR.pdf"

text_data = extract_text_from_pdf(pdf1) + extract_text_from_pdf(pdf2)

# --------- Split text into chunks ---------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text_data)

# --------- Create embeddings with SentenceTransformer ---------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_tensor=False)

# --------- Store embeddings in FAISS ---------
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save chunks alongside index
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

faiss.write_index(index, "vector.index")
print("✅ Database ready with", len(chunks), "chunks")
