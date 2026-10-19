from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import util

def create_vector_store(documents):
    #create and return a FAISS vector store from the given documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    #using liangchain model to change the text chunks into vectors value or embedded values
    vector_store = FAISS.from_documents(documents, embeddings)
    
    #Store the embedded values into the vectorStore
    return vector_store

def rerank_documents(vector_store, query, top_k=2):
    #Rerank retrieved documents based on cosine similarity
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    #the query which was turned into embedded value is also retrieve
    query_embedding = embeddings.embed_query(query)
    #retrieve the docs or chunks of text that matches the asked question
    retrieved_docs = vector_store.similarity_search(query, k=top_k * 2) 
    doc_embeddings = [embeddings.embed_documents([doc.page_content])[0] for doc in retrieved_docs]
    
    #checked if there are similarities within the query and the stored data embeddings
    similarities = util.cos_sim(query_embedding, doc_embeddings)
    
    #rank all the data that passes the threshold value of the data embeddings checker
    ranked_indices = similarities[0].argsort(descending=True)
    
    #return all the text that passes the threshold value for similarity between query and text
    return [retrieved_docs[i] for i in ranked_indices.tolist()] 
