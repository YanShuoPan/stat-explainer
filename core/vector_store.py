# core/vector_store.py

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import os

CHROMA_PATH = "data/chroma_db"


def get_text_chunks(text: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def create_vectorstore_from_text(text: str, doc_id="notes"):
    chunks = get_text_chunks(text)
    docs = [
        Document(page_content=chunk, metadata={"source": doc_id}) for chunk in chunks
    ]
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    return vectordb


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def query_vectorstore(question: str, k=3):
    db = load_vectorstore()
    docs = db.similarity_search(question, k=k)
    return docs
