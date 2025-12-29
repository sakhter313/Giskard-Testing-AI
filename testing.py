import streamlit as st
from datasets import load_dataset
import pandas as pd

# Page config
st.set_page_config(page_title="RealHarm Vulnerability Tester", layout="wide")

# Load dataset once
@st.cache_data
def load_realharm_dataset():
    dataset = load_dataset("giskardai/realharm")
    # Convert to DataFrames for easier manipulation
    safe_df = pd.DataFrame(dataset["safe"])
    unsafe_df = pd.DataFrame(dataset["unsafe"])
    return safe_df, unsafe_df

st.title("🛡️ RealHarm Dataset: Testing LLM Vulnerabilities")
st.markdown("Explore real-world AI failure cases from the [Giskard RealHarm dataset](https://huggingface.co/datasets/giskardai/realharm). Filter by harm category to see how prompts can elicit unsafe responses, and compare with safe alternatives.")

# Load data
safe_df, unsafe_df = load_realharm_dataset()

# Overview section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples (Safe)", len(safe_df))
with col2:
    st.metric("Total Samples (Unsafe)", len(unsafe_df))
with col3:
    st.metric("Unique Languages", safe_df["language"].nunique())

st.subheader("Harm Categories")
# Robust computation: skip None taxonomies
categories = sorted({cat for tax in safe_df["taxonomy"] if tax is not None for cat in tax})
st.write(", ".join(categories))

# Sidebar filters
st.sidebar.header("Filters")
selected_category = st.sidebar.multiselect(
    "Select Harm Categories",
    options=categories,
    default=[]  # No default selection - start broad, then filter
)
selected_language = st.sidebar.selectbox(
    "Select Language",
    options=sorted(safe_df["language"].unique()),
    index=0
)

if not selected_category:
    st.info("No categories selected - showing all samples for the selected language.")

# Filter data (robust: skip non-list taxonomies)
def has_category(tax, cats):
    if not isinstance(tax, (list, tuple)):
        return False
    return any(cat in cats for cat in tax)

filtered_safe = safe_df[
    (safe_df["language"] == selected_language) &
    (safe_df["taxonomy"].apply(lambda x: has_category(x, selected_category)))
]
filtered_unsafe = unsafe_df[
    (unsafe_df["language"] == selected_language) &
    (unsafe_df["taxonomy"].apply(lambda x: has_category(x, selected_category)))
]

# Match samples by sample_id (strip prefix for matching)
filtered_safe["base_id"] = filtered_safe["sample_id"].str.replace("safe_", "").str.replace("unsafe_", "")
filtered_unsafe["base_id"] = filtered_unsafe["sample_id"].str.replace("safe_", "").str.replace("unsafe_", "")
matched_pairs = pd.merge(filtered_safe, filtered_unsafe, on="base_id", suffixes=("_safe", "_unsafe"))

if len(matched_pairs) == 0:
    st.warning("No samples match the selected filters. Try broadening your selection.")
else:
    # Sample selector
    sample_options = matched_pairs["sample_id_safe"].tolist()
    selected_sample = st.selectbox("Select a Sample", sample_options)

    if selected_sample:
        row = matched_pairs[matched_pairs["sample_id_safe"] == selected_sample].iloc[0]

        # Display details
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🛑 Unsafe Interaction")
            st.write(f"**Sample ID:** {row['sample_id_unsafe']}")
            st.write(f"**Context:** {row['context_unsafe']}")
            st.write(f"**Source:** [{row['source_unsafe']}]({row['source_unsafe']})")
            unsafe_tax = row['taxonomy_unsafe'] if isinstance(row['taxonomy_unsafe'], (list, tuple)) else []
            st.write("**Harm Categories:**", ", ".join(unsafe_tax))

            for msg in row["conversation_unsafe"]:
                role_badge = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"**{role_badge} {msg['role'].title()}:** {msg['content']}")

        with col_right:
            st.subheader("✅ Safe Interaction")
            st.write(f"**Sample ID:** {row['sample_id_safe']}")
            st.write(f"**Context:** {row['context_safe']}")
            st.write(f"**Source:** [{row['source_safe']}]({row['source_safe']})")
            safe_tax = row['taxonomy_safe'] if isinstance(row['taxonomy_safe'], (list, tuple)) else []
            st.write("**Harm Categories:**", ", ".join(safe_tax))

            for msg in row["conversation_safe"]:
                role_badge = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"**{role_badge} {msg['role'].title()}:** {msg['content']}")

        st.markdown("---")
        st.caption("This app uses the RealHarm dataset to highlight vulnerabilities. For production testing, integrate with tools like Giskard's scanning library.")
