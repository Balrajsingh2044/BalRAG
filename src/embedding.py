from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import util

def create_vector_store(documents):
    '''
    create and return a FAISS vector store from the given documents
    '''
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def rerank_documents(vector_store, query, top_k=2):
    '''Rerank retrieved documents based on cosine similarity.'''
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query_embedding = embeddings.embed_query(query)
    retrieved_docs = vector_store.similarity_search(query, k=top_k * 2) 
    doc_embeddings = [embeddings.embed_documents([doc.page_content])[0] for doc in retrieved_docs]
    similarities = util.cos_sim(query_embedding, doc_embeddings)
    ranked_indices = similarities[0].argsort(descending=True)
    return [retrieved_docs[i] for i in ranked_indices.tolist()]