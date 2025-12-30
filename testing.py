import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="LLM Safety & Hallucination Testing", layout="wide")
st.title("🧪 LLM Safety & Hallucination Testing")

hf_token = st.text_input("Enter Hugging Face API Token", type="password")
if not hf_token:
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )

df = load_data()
st.success(f"Loaded {len(df)} support records")

# -------------------------------
# VECTOR STORE
# -------------------------------
@st.cache_resource
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(df["query"].tolist(), embeddings)

vectorstore = build_vectorstore()

# -------------------------------
# LLM (SAFE MODE)
# -------------------------------
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
    task="text2text-generation",
    temperature=0.2,
    max_new_tokens=256,
)

# -------------------------------
# QUERY LOGIC (NO RETRIEVALQA)
# -------------------------------
st.subheader("💬 Ask the Support Bot")
query = st.text_input("Your question")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        docs = vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful customer support assistant.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

        response = llm.invoke(prompt)
        st.success(response)

