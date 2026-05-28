from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, embeddings, db_type="faiss"):

    if db_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)

    elif db_type == "chroma":
        return Chroma.from_documents(chunks, embeddings)

    else:
        raise ValueError("Invalid DB type")