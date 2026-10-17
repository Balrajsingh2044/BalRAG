from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader


def load_documents(source_type, source_content):
    """Load documents from different sources (PDF, URL, or raw text)."""
    if source_type == "pdf":
        if not isinstance(source_content, str):
            raise ValueError("For PDF, provide file path as string")
        loader = PyPDFLoader(source_content)
    elif source_type == "url":
        if not isinstance(source_content, str) or not source_content.startswith(('http://', 'https://')):
            raise ValueError("For URL, provide a valid URL string")
        loader = WebBaseLoader(source_content)
    elif source_type == "text":
        if not isinstance(source_content, str):
            raise ValueError("For text, provide a string")
        from langchain_core.documents import Document
        return [Document(page_content=source_content, metadata={"source": "user_input"})]
    else:
        raise ValueError("Unsupported source type. Use 'pdf', 'url', or 'text'")
    return loader.load()