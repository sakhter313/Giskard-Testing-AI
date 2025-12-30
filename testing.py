import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import giskard
from giskard import Model, scan

st.set_page_config(page_title="Customer Support Chatbot with Giskard Scan", page_icon="🤖")

st.title("Customer Support RAG Chatbot")
st.markdown("Built with LangChain, Falcon-7B, and tested for vulnerabilities using Giskard.")

# Set environment variables from Streamlit secrets
if "HF_TOKEN" not in st.secrets:
    st.error("Please add 'HF_TOKEN' to Streamlit secrets for Hugging Face API.")
    st.stop()
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add 'OPENAI_API_KEY' to Streamlit secrets for Giskard scanning.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_data():
    """Load the customer support dataset."""
    url = "https://huggingface.co/datasets/Kaludi/Customer-Support-Responses/resolve/main/Customer-Support.csv"
    df = pd.read_csv(url)
    # Use 'query' as in the original notebook to match behavior
    data = df['query'].tolist()
    return data

@st.cache_resource
def setup_model():
    """Set up embeddings, vector store, LLM, and QA chain."""
    texts = load_data()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)
    
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.5, "max_length": 128}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    return qa_chain

qa_chain = setup_model()

class FalconRAGModel(Model):
    """Custom Giskard model wrapper for the RetrievalQA chain."""
    
    def model_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict responses for input queries."""
        df = df.copy()
        df["prediction"] = df["query"].apply(lambda x: self.model.run(x))
        return df

@st.cache_resource
def run_giskard_scan():
    """Run Giskard vulnerability scan on the model."""
    giskard_model = FalconRAGModel(
        model=qa_chain,
        model_type="text_generation",
        name="Customer Service Assistant",
        description="A customer service assistant bot to respond to customer support questions.",
        feature_names=["query"],
    )
    results = scan(giskard_model)
    return results

# Run the scan (cached)
with st.spinner("Running vulnerability scan... This may take a few minutes on first load."):
    scan_results = run_giskard_scan()

# Display Vulnerability Scan Results
st.header("🛡️ Vulnerability Scan Results")
st.markdown("Giskard automatically detects potential issues like hallucinations, bias, harmful content, and more.")

# Render the scan report as HTML
html_report = scan_results.to_html()
st.components.v1.html(html_report, height=800, scrolling=True)

# Interactive Chatbot
st.header("💬 Test the Chatbot")
st.markdown("Enter a customer support query to get a response from the Falcon-7B powered RAG system.")

query = st.text_input("Your query:", placeholder="e.g., My product is broken.")

if st.button("Generate Response") and query:
    with st.spinner("Generating response using Falcon-7B..."):
        try:
            response = qa_chain.run(query)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Deployed on Streamlit Cloud. Ensure `requirements.txt` includes: `streamlit giskard[llm] langchain langchain-community langchain-huggingface transformers sentence-transformers faiss-cpu pandas`.")
