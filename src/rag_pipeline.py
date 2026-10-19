import torch
from transformers import pipeline

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline


# This makes sure that the file work with either version of LC
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

from src.embedding import rerank_documents


def create_rag_pipeline(vector_store):
    #Create and return a RAG pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hf_pipeline = pipeline(
        'text2text-generation',
        model='google/flan-t5-base',
        device=device,
        max_new_tokens=100,
    )

    #connecting the rag to hf
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    #prompt template
    prompt = PromptTemplate(
        template="Answer the question based on the context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

    #concatenates all retrieved documents into a single prompt
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
    #Get the answer to a question using the RAG pipeline
    #Calls the rerank_document function to get the best possible context
    ranked_docs = rerank_documents(vector_store, question, top_k=2)
    
    #Joins the reranked documents into a single string
    context = '\n'.join(doc.page_content for doc in ranked_docs)
    
    #store the value of query and context
    result = qa_chain({'question': question, 'context': context})
    return result.get('answer', result.get('result', 'No answer found'))