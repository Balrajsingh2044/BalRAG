import os
os.environ["USER_AGENT"] = "BalRAG/1.0"

import streamlit as st
from src.document_loader import load_documents
from src.embedding import create_vector_store, rerank_documents
from src.rag_pipeline import create_rag_pipeline, get_answer
import time

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    
st.title('BalRAG')
st.subheader('Balraj Singh\'s RAG-based AI Assistant for Document Analysis')

# Document Input section
with st.sidebar:
    st.header("Document Input")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    url = st.text_input("Enter URL", placeholder="https://example.com") #not working right now
    raw_text = st.text_area("Enter Raw Text", height=100) #text of 100 lines can be inputed in the text section
    process_btn = st.button("Process Documents")

    if process_btn:
        all_docs = []
        
        if pdf_file:
            pdf_path = pdf_file.name
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            pdf_docs = load_documents("pdf", pdf_path)
            all_docs.extend(pdf_docs)
            
        if url:
            url_docs = load_documents("url", url)
            all_docs.extend(url_docs)
            
        if raw_text:
            text_docs = load_documents("text", raw_text)
            all_docs.extend(text_docs)

        if all_docs:
            st.session_state.vector_store = create_vector_store(all_docs)
            st.session_state.pipeline = create_rag_pipeline(st.session_state.vector_store)
            st.success("Documents processed successfully!")
            
        else:
            st.error("No valid input provided.")

# Query input Section
#it is customize to enter the query + context into the vectorDB to check it against the similar options 
question = st.text_input("Ask a Question", key="question_input", placeholder="Type your question here")

#Output generation section
answer = None
if st.button("Get Answer", key="get_answer_btn"):
    if st.session_state.pipeline and st.session_state.vector_store:
        st.write("Generating answer...")
        answer = get_answer(st.session_state.pipeline, st.session_state.vector_store, question)
        placeholder = st.empty()
        typed = ""
        for char in answer:
            typed += char
            placeholder.write(typed)
            time.sleep(0.02)
    else:
        st.error("Please process documents first.")

#history section
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if question and answer:
    st.session_state.chat_history.append({"question": question, "answer": answer})
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")