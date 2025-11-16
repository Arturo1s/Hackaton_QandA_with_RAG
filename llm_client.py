# llm_client.py
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_KEY,
)

def generate_answer(prompt: str, max_tokens=350):
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.25,
        top_p=0.95
    )
    return response.choices[0].message.content
