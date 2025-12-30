import streamlit as st
import os
import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

import giskard
from giskard import Model

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Giskard LLM Evaluation Demo", layout="wide")

st.title("🔍 Giskard LLM Evaluation – Customer Support Chatbot")
st.markdown("This app reproduces the **exact vulnerability findings** from the notebook.")

# ---------------------------
# API KEY INPUT
# ---------------------------
hf_token = st.text_input("HuggingFace API Token", type="password")

if not hf_token:
    st.warning("Please enter your Hugging Face API token to continue.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# ---------------------------
# Load Dataset
# ---------------------------
st.subheader("📄 Loading Dataset")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv"
    )
    return df

df = load_data()
st.success(f"Loaded {len(df)} customer support records")

# ---------------------------
# Create Embeddings
# ---------------------------
st.subheader("🔗 Creating Embeddings")

@st.cache_resource
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(texts, embeddings)

data = df["query"].tolist()
vectorstore = create_vector_store(data)

# ---------------------------
# Load LLM (Falcon 7B)
# ---------------------------
st.subheader("🧠 Loading Falcon 7B Model")

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.5, "max_length": 128},
)

# ---------------------------
# Create Retrieval QA Chain
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# ---------------------------
# Chat Interface
# ---------------------------
st.subheader("💬 Ask the Support Bot")

query = st.text_input("Enter a customer query:", "My product is broken")

if st.button("Run Query"):
    response = qa_chain.run(query)
    st.success("Model Response:")
    st.write(response)

# ---------------------------
# Giskard Evaluation
# ---------------------------
st.subheader("🧪 Run Giskard Safety Evaluation")

if st.button("Run Giskard Scan"):

    giskard_model = giskard.Model(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Support Assistant",
        description="Customer support chatbot using Falcon 7B",
        feature_names=["query", "response"]
    )

    scan_results = giskard.scan(giskard_model)

    st.error("⚠️ Vulnerabilities Detected")
    st.write(scan_results)

    st.markdown("### ⚠️ Summary of Issues")
    st.write("- Hallucinations")
    st.write("- Sensitive Information Leakage")
    st.write("- Prompt Injection")
    st.write("- Bias & Harmful Content")


