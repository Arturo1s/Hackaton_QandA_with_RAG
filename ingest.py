# ingest.py
import os
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from config import EXCEL_PATH, DATA_DIR, INDEX_PATH, META_PATH, EMBEDDING_MODEL, COL_ID, COL_TITLE, COL_CONTENT, TOP_K

os.makedirs(DATA_DIR, exist_ok=True)


def load_excel(path):
    df = pd.read_excel(path, engine="openpyxl")
    # Basic cleanup: fillna
    df = df.fillna("")
    # Build a textual field for embedding:
    def doc_text(row):
        parts = []
        if row.get(COL_TITLE):
            parts.append(str(row[COL_TITLE]))
        if row.get(COL_CONTENT):
            parts.append(str(row[COL_CONTENT]))
        # add topics & meta
        if row.get("Thématiques"):
            parts.append(str(row["Thématiques"]))
        return "\n".join(parts)
    df["__text"] = df.apply(lambda r: doc_text(r), axis=1)
    return df


def build_embeddings(texts, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine search
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings, index_path=INDEX_PATH):
    dim = embeddings.shape[1]
    # use IndexFlatIP (inner product) on normalized vectors -> cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


def save_metadata(meta, meta_path=META_PATH):
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)


def main(args):
    df = load_excel(EXCEL_PATH)
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "id": row.get(COL_ID, ""),
            "title": row.get(COL_TITLE, ""),
            "content": row.get(COL_CONTENT, ""),
            "date": row.get("Date", ""),
            "lang": row.get(COL_LANG, ""),
            "topics": row.get("Thématiques", ""),
            "users": row.get("Utilisateurs", ""),
            "schools": row.get("Écoles", ""),
            "status": row.get("Status", ""),
            "text": row.get("__text", "")
        })
    texts = [d["text"] for d in docs]
    embeddings = build_embeddings(texts)
    build_faiss_index(embeddings)
    save_metadata(docs)
    print(f"Ingested {len(docs)} docs. Index saved to {INDEX_PATH}, meta to {META_PATH}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Reindex from EXCEL_PATH")
    args = parser.parse_args()
    main(args)