# rag_engine.py
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/questions.csv"
INDEX_PATH = "vectorstore/faiss_index.bin"
EMB_PATH = "vectorstore/embeddings.npy"

# Embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding='latin-1', sep=';', on_bad_lines='skip')

    df["text"] = (
        df["Title"].fillna("") + " - " +
        df["Content"].fillna("") + " - " +
        df["Thématiques"].fillna("") + " - " +
        df["Écoles"].fillna("")
    )
    
    return df


def build_faiss_index():
    df = load_dataset()
    texts = df["text"].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embeddings)

    return index, embeddings, df


def load_faiss():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(EMB_PATH):
        print("Index or embeddings not found. Building from scratch...")
        return build_faiss_index()

    try:
        index = faiss.read_index(INDEX_PATH)
        embeddings = np.load(EMB_PATH)
        df = load_dataset()
        return index, embeddings, df
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Rebuilding index from scratch...")
        # Delete corrupted files
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(EMB_PATH):
            os.remove(EMB_PATH)
        return build_faiss_index()


def search(query, k=3):
    index, embeddings, df = load_faiss()

    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype("float32"), k)

    results = df.iloc[I[0]].to_dict(orient="records")
    return results
