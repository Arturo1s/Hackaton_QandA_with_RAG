# config.py
import os

# Path to the Excel file you provided
EXCEL_PATH = os.getenv("EXCEL_PATH", "Questions-Export-2025-October-27-1237 (1).xlsx")

# Column names mapping (adjust if your sheet uses different names)
COL_ID = "id"
COL_TITLE = "Title"
COL_CONTENT = "Content"
COL_DATE = "Date"
COL_LANG = "Langues"
COL_TOPICS = "Thématiques"
COL_USERS = "Utilisateurs"
COL_SCHOOLS = "Écoles"
COL_STATUS = "Status"

# Where to store vector index and metadata
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")  # pickled metadata list
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLaMA / llama.cpp model path (you must provide a local model)
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-2-13b.ggmlv3.q4_0.bin")

# LLaMA generation params
LLAMA_PARAMS = {
    "n_ctx": 2048,
    "n_threads": 4,
    "temp": 0.2,
    "top_k": 40,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    "max_tokens": 512,
}

# Similarity threshold (cosine similarity). If top result < THRESH, we redirect.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))

# Logging
INTERACTION_LOG = os.path.join(DATA_DIR, "interactions.csv")

# Redirect email/form
CONTACT_EMAIL = "guillaume.douceron@devinci.fr"
REDIRECT_FORM_URL = "https://your-institution.edu/help-request-form"  # replace with real form

# Misc
TOP_K = int(os.getenv("TOP_K", 5))