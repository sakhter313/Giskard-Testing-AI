import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

import giskard
from giskard import Model

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="Giskard LLM Safety Test", layout="wide")
st.title("🧪 LLM Safety Testing with Giskard")

# -----------------------------------
# API KEY
# -----------------------------------
hf_token = st.text_input("Enter HuggingFace API Token", type="password")

if not hf_token:
    st.warning("Please enter your Hugging Face API token.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# -----------------------------------
# LOAD DATASET
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )

df = load_data()
st.success(f"Loaded {len(df)} customer queries")

# -----------------------------------
# EMBEDDINGS
# -----------------------------------
@st.cache_resource
def build_vector_store(texts):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(texts, embeddings)

vector_store = build_vector_store(df["query"].tolist())

# -----------------------------------
# LOAD LLM
# -----------------------------------
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.5, "max_length": 128},
)

# -----------------------------------
# RETRIEVAL CHAIN
# -----------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# -----------------------------------
# INTERACTIVE CHAT
# -----------------------------------
st.subheader("💬 Ask the Support Bot")
query = st.text_input("Ask a question:", "My product is broken")

if st.button("Run Query"):
    answer = qa_chain.run(query)
    st.success(answer)

# -----------------------------------
# GISKARD SAFETY TEST
# -----------------------------------
st.subheader("🧪 Run Giskard Safety Scan")

if st.button("Run Safety Evaluation"):

    giskard_model = Model(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Support Assistant",
        description="Customer support LLM powered by Falcon 7B",
        feature_names=["query", "response"],
    )

    scan_results = giskard.scan(giskard_model)

    st.error("⚠️ Issues Detected")
    st.write(scan_results)

