"""Streamlit chat app for the HR Policy Assistant.

Run with:  streamlit run app.py
"""

import streamlit as st

from hr_assistant.pipeline import add_pdf_to_hr_assistant, ask, build_hr_assistant


st.set_page_config(page_title="HR Policy Assistant", page_icon="🤖")
st.title("🤖 HR Policy Assistant")
st.caption("Ask me anything about the company HR policy document.")


if "agent" not in st.session_state:
    with st.spinner("Setting up the assistant..."):
        st.session_state.agent = build_hr_assistant()

with st.sidebar:
    st.header("Knowledge base")
    uploaded_pdf = st.file_uploader("Upload an HR policy PDF", type=["pdf"])
    if uploaded_pdf and st.button("Add PDF to knowledge base", use_container_width=True):
        with st.spinner("Extracting text and updating the knowledge base..."):
            try:
                agent, chunk_count, already_indexed = add_pdf_to_hr_assistant(
                    uploaded_pdf.name, uploaded_pdf.getvalue()
                )
            except Exception as error:
                st.error(f"Could not add this PDF: {error}")
            else:
                st.session_state.agent = agent
                if already_indexed:
                    st.info("This PDF is already in the knowledge base.")
                else:
                    st.success(f"Added {chunk_count} searchable chunks from {uploaded_pdf.name}.")

agent = st.session_state.agent

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the past conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get a new question from the user
question = st.chat_input("Ask a question about HR policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(agent, question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
