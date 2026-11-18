
import streamlit as st

from model_loader import load_model, generate_response
from formatter import format_response


# Page configuration
st.set_page_config(
    page_title="EduSupport AI",
    page_icon="🎓",
    layout="centered"
)


# Load model only once
@st.cache_resource
def initialize_model():
    return load_model()


# Application header
st.title(" EduSupport for Engineer ")

st.caption(
    "Your friendly AI assistant for engineering students."
)


# Load model
with st.spinner("Loading EduSupport AI..."):
    model, tokenizer = initialize_model()


# Branch selection
branches = [
    "AI & ML",
    "CSE",
    "ECE",
    "Mechanical",
    "Civil"
]

branch = st.selectbox(
    "Select your engineering branch",
    branches
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Clear conversation
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask your engineering question..."
)


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("EduSupport AI is thinking..."):

            answer = generate_response(
                model=model,
                tokenizer=tokenizer,
                branch=branch,
                question=question
            )

            formatted_answer = format_response(answer)

            st.markdown(formatted_answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": formatted_answer
    })
