import os
import streamlit as st
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.retrieval_qa.base import RetrievalQA

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="LLM Safety & Hallucination Testing", layout="wide")

st.title("🧪 LLM Safety & Hallucination Testing")

# -------------------------------
# TOKEN
# -------------------------------
hf_token = st.text_input("Enter Hugging Face API Token", type="password")
if not hf_token:
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv")

df = load_data()

# -------------------------------
# VECTOR STORE
# -------------------------------
@st.cache_resource
def build_store():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(df["query"].tolist(), embeddings)

vectorstore = build_store()

# -------------------------------
# LLM (FIXED)
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=hf_token,
    task="text2text-generation",
    max_new_tokens=256,
    temperature=0.3,
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
# UI
# -------------------------------
st.subheader("💬 Ask the Support Bot")

query = st.text_input("Your question", "My product stopped working")

if st.button("Ask"):
    response = qa_chain.run(query)
    st.success(response)

