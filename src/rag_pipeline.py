import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

def create_rag_pipeline(vector_store):
    """
    Initializes the Groq LLM and the prompt template.
    """
    # Initialize Groq LLM with the currently supported, high-speed model
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",  # <-- UPDATED to Groq's current recommended model
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Strict prompt to prevent hallucinations
    prompt = PromptTemplate(
        template=(
            "You are a helpful AI assistant. Answer the question based *only* on the provided context. "
            "If the context does not contain the answer, strictly reply with 'I don't know based on the provided documents.'\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    # Modern LangChain (LCEL) syntax: connects prompt directly to the LLM
    chain = prompt | llm
    return chain


def get_answer(qa_chain, vector_store, question):
    """
    Fetches the best context using your custom reranker, 
    then passes it to the Groq LLM to generate the answer.
    """
    from src.embedding import rerank_documents
    
    # 1. Get the top 2 most relevant documents using your reranker
    ranked_docs = rerank_documents(vector_store, question, top_k=2)
    context = '\n'.join(doc.page_content for doc in ranked_docs)
    
    # 2. Safety check: if no context was found
    if not context.strip():
        return "No relevant information found in the documents to answer this question."

    # 3. Pass the context and question to the Groq chain
    response = qa_chain.invoke({"context": context, "question": question})
    
    # ChatGroq returns a message object; .content extracts the actual text string
    return response.content