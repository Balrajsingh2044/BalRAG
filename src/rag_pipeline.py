import torch
from transformers import pipeline

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# LangChain 0.3.x keeps RetrievalQA in langchain.chains.
# LangChain 1.x moved it into langchain_classic.chains.
# This try/except makes the file work with either version.
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

from src.embedding import rerank_documents


def create_rag_pipeline(vector_store):
    '''Create and return a RAG pipeline.'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hf_pipeline = pipeline(
        'text2text-generation',
        model='google/flan-t5-base',
        device=device,
        max_new_tokens=100,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate(
        template="Answer the question based on the context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        ),
        return_source_documents=True,
        input_key="question",
        output_key="answer",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def get_answer(qa_chain, vector_store, question):
    '''Get the answer to a question using the RAG pipeline.'''
    ranked_docs = rerank_documents(vector_store, question, top_k=2)
    context = '\n'.join(doc.page_content for doc in ranked_docs)
    result = qa_chain({'question': question, 'context': context})
    return result.get('answer', result.get('result', 'No answer found'))