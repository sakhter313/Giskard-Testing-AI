import streamlit as st
from huggingface_hub import InferenceClient
import giskard
from giskard import Model

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(
    page_title="LLM Safety & Hallucination Testing",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 LLM Safety & Hallucination Testing")

# ---------------------------
# INPUTS
# ---------------------------
hf_token = st.text_input(
    "Enter HuggingFace API Token",
    type="password"
)

user_question = st.text_input(
    "Ask the Support Bot",
    placeholder="My product stopped working"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model(token):
    return InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        token=token
    )

# ---------------------------
# MAIN ACTION
# ---------------------------
if st.button("Ask"):

    if not hf_token:
        st.error("Please enter your HuggingFace API token.")
        st.stop()

    with st.spinner("Loading model..."):
        client = load_model(hf_token)

    try:
        # Generate response
        response = client.text_generation(
            user_question,
            max_new_tokens=200,
            temperature=0.3
        )

        st.success("Response generated")
        st.write(response)

        # ---------------------------
        # GISKARD SAFETY CHECK
        # ---------------------------
        st.subheader("🔍 Safety Evaluation")

        llm = Model(
            model=client,
            model_type="text_generation",
            name="SupportBot",
            description="Customer support assistant"
        )

        report = giskard.scan(llm)
        st.success("Safety scan completed")
        st.write(report)

    except Exception as e:
        st.error("An error occurred")
        st.exception(e)

