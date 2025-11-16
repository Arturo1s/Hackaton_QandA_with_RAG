# app.py
import streamlit as st
from rag_engine import search
from llm_client import generate_answer
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "logs/interactions.csv"

st.set_page_config(page_title="PLV Intelligent Help Center", layout="wide")
st.title("PULV Intelligent Help Center")


def log_interaction(question, answer, source_found):
    os.makedirs("logs", exist_ok=True)

    row = {
        "timestamp": datetime.now(),
        "question": question,
        "answer": answer,
        "source_found": source_found,
    }

    if not os.path.exists(LOG_PATH):
        pd.DataFrame([row]).to_csv(LOG_PATH, index=False)
    else:
        try:
            df = pd.read_csv(LOG_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # If log file is corrupted or empty, create a new one
            df = pd.DataFrame([row])
        df.to_csv(LOG_PATH, index=False)


query = st.text_input("Pose ta question :", "")

if st.button("Rechercher"):
    if query.strip() == "":
        st.warning("Merci d’entrer une question.")
        st.stop()

    with st.spinner("Recherche des réponses…"):
        results = search(query, k=3)

        context_html = ""
        for r in results:
            context_html += f"<p><b>{r['Title']} :</b> {r['Content']}</p>\n"

        # RAG prompt for LLaMA
        prompt = f"""
You are an intelligent academic assistant for PLV students.
You ALWAYS answer in clean HTML.

### KNOWLEDGE BASE CONTEXT:
{context_html}

### STUDENT QUESTION:
{query}

### RULES:
- If the context contains a relevant answer, use it and rewrite it clearly in HTML.
- If the context does not provide an answer, respond exactly:

<p>Désolé, aucune réponse exacte n’a été trouvée. Merci de remplir le formulaire ou de contacter : 
<a href='mailto:guillaume.douceron@devinci.fr'>guillaume.douceron@devinci.fr</a></p>
"""

        answer = generate_answer(prompt)

    source_found = "aucune réponse exacte" not in answer.lower()
    log_interaction(query, answer, source_found)

    st.markdown(answer, unsafe_allow_html=True)

    with st.expander("🔍 Sources utilisées"):
        for r in results:
            st.markdown(f"### {r['Title']}")
            st.write(r["Content"])
            st.write("---")
