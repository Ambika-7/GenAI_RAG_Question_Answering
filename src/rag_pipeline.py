from loader import load_pdf
from chunking import split_documents
from embeddings import get_embeddings
from vectorstores import create_vector_store
from llm import get_llm


def build_pipeline(pdf_path, chunk_size, embedding_model, db_type):

    docs = load_pdf(pdf_path)

    chunks = split_documents(docs, chunk_size=chunk_size)

    embeddings = get_embeddings(embedding_model)

    vectorstore = create_vector_store(chunks, embeddings, db_type)

    return vectorstore


def generate_answer(vectorstore, query, llm_model="llama3"):

    docs = vectorstore.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    llm = get_llm(llm_model)

    prompt = f"""
    Answer the question using ONLY the context.
    Give a SHORT and PRECISE answer (1-2 lines).
    Do NOT add extra explanation.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    return response, docs