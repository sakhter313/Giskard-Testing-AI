import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA

import giskard
from giskard import Model

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="LLM Safety Testing", layout="wide")
st.title("🧪 LLM Safety & Hallucination Testing")

# -------------------------------
# HUGGING FACE TOKEN
# -------------------------------
hf_token = st.text_input("Enter Hugging Face API Token", type="password")
if not hf_token:
    st.warning("Please enter your Hugging Face token.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )

df = load_dataset()
st.success(f"Loaded {len(df)} customer support records")

# -------------------------------
# BUILD VECTOR STORE
# -------------------------------
@st.cache_resource
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(df["query"].tolist(), embeddings)

vectorstore = build_vectorstore()

# -------------------------------
# LOAD LLM (FREE MODEL)
# -------------------------------
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",   # ✅ FREE & STABLE
    huggingfacehub_api_token=hf_token,
    model_kwargs={
        "temperature": 0.3,
        "max_length": 256,
    },
)

# -------------------------------
# RETRIEVAL QA CHAIN
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# -------------------------------
# CHAT UI
# -------------------------------
st.subheader("💬 Ask the Support Bot")

query = st.text_input("Your question", "My product stopped working")

if st.button("Ask"):
    response = qa_chain.run(query)
    st.success(response)

# -------------------------------
# GISKARD SAFETY SCAN
# -------------------------------
st.subheader("🧪 Run Safety Evaluation")

if st.button("Run Scan"):
    model = Model(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Support Assistant",
        description="LLM for customer support QA",
        feature_names=["query", "response"],
    )

    results = giskard.scan(model)
    st.error("⚠️ Safety Issues Detected")
    st.write(results)

