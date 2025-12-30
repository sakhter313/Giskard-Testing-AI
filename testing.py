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
st.set_page_config(
    page_title="LLM Safety Evaluation",
    layout="wide"
)

st.title("🧪 LLM Safety & Hallucination Testing")

# -------------------------------
# HUGGINGFACE TOKEN
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
st.success(f"Loaded {len(df)} support records")

# -------------------------------
# BUILD VECTOR STORE
# -------------------------------
@st.cache_resource
def build_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(texts, embeddings)

vectorstore = build_vectorstore(df["query"].tolist())

# -------------------------------
# LOAD MODEL
# -------------------------------
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_token,
    model_kwargs={
        "temperature": 0.5,
        "max_length": 128,
    },
)

# -------------------------------
# RETRIEVAL QA
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# -------------------------------
# CHAT INTERFACE
# -------------------------------
st.subheader("💬 Ask the Support Bot")

query = st.text_input("Your question", "My product stopped working")

if st.button("Ask"):
    response = qa_chain.run(query)
    st.success(response)

# -------------------------------
# GISKARD EVALUATION
# -------------------------------
st.subheader("🧪 Run Giskard Safety Scan")

if st.button("Run Scan"):
    model = Model(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Support Assistant",
        description="LLM used for customer support Q&A",
        feature_names=["query", "response"]
    )

    results = giskard.scan(model)
    st.error("⚠️ Vulnerabilities Detected")
    st.write(results)
