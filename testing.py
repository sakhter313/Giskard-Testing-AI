import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.retrieval_qa.base import RetrievalQA

# --------------------------------
# STREAMLIT CONFIG
# --------------------------------
st.set_page_config(page_title="LLM Safety & Hallucination Testing", layout="wide")
st.title("🧪 LLM Safety & Hallucination Testing")

# --------------------------------
# HUGGING FACE TOKEN
# --------------------------------
hf_token = st.text_input("Enter Hugging Face API Token", type="password")
if not hf_token:
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# --------------------------------
# LOAD DATASET
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )

df = load_data()
st.success(f"Loaded {len(df)} customer support records")

# --------------------------------
# VECTOR STORE
# --------------------------------
@st.cache_resource
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(df["query"].tolist(), embeddings)

vectorstore = build_vectorstore()

# --------------------------------
# LLM — EXPLICIT ENDPOINT (CRITICAL FIX)
# --------------------------------
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
    task="text2text-generation",
    max_new_tokens=256,
    temperature=0.3,
)

# --------------------------------
# RETRIEVAL QA
# --------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
)

# --------------------------------
# UI
# --------------------------------
st.subheader("💬 Ask the Support Bot")

query = st.text_input("Your question", "My product stopped working")

if st.button("Ask"):
    with st.spinner("Generating answer..."):
        response = qa_chain.run(query)
    st.success(response)

