import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA

import giskard
from giskard import Model

# ---------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------
st.set_page_config(page_title="Giskard LLM Safety", layout="wide")
st.title("🧪 LLM Safety Evaluation Dashboard")

# ---------------------------------------
# HUGGING FACE TOKEN
# ---------------------------------------
hf_token = st.text_input("HuggingFace API Token", type="password")

if not hf_token:
    st.warning("Enter HuggingFace API token to continue.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )

df = load_data()
st.success(f"Loaded {len(df)} records")

# ---------------------------------------
# EMBEDDINGS
# ---------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(df["query"].tolist(), embeddings)

vectorstore = load_vectorstore()

# ---------------------------------------
# LLM
# ---------------------------------------
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_token,
    model_kwargs={
        "temperature": 0.5,
        "max_length": 128
    }
)

# ---------------------------------------
# RETRIEVAL QA
# ---------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# ---------------------------------------
# CHAT
# ---------------------------------------
st.subheader("💬 Ask a question")

query = st.text_input("Your question", "My product stopped working")

if st.button("Ask"):
    answer = qa_chain.run(query)
    st.success(answer)

# ---------------------------------------
# GISKARD SCAN
# ---------------------------------------
st.subheader("🧪 Giskard Security Scan")

if st.button("Run Scan"):
    model = Model(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Support Assistant",
        description="LLM safety evaluation",
        feature_names=["query", "response"],
    )

    results = giskard.scan(model)
    st.error("⚠️ Vulnerabilities Found")
    st.write(results)

