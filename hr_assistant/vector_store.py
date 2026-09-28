from hr_assistant import config
import json
import os
from langchain_community.vectorstores import FAISS

from hr_assistant.embeddings import get_embeddings_model

def build_vector_store(chunks):
    """Embed every chunk and build a searchable FAISS index in memory"""

    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)


#save vector store
def save_vector_store(vector_store, path:str=config.VECTOR_STORE_PATH)->None:
    vector_store.save_local(path)


def load_vector_store(path:str=config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk"""

    embedding_model = get_embeddings_model()

    return FAISS.load_local(path,embedding_model, allow_dangerous_deserialization=True)

def vector_store_exists(path:str=config.VECTOR_STORE_PATH) -> bool:
    """Check if the Vector Store already exists."""
    return os.path.exists(os.path.join(path,"index.faiss"))
    
def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that returns the top-k relevant chunks"""
    return vector_store.as_retriever(search_kwargs={"k":k})


def _uploaded_files_manifest_path(path: str = config.VECTOR_STORE_PATH) -> str:
    return os.path.join(path, "uploaded_files.json")


def uploaded_file_exists(file_hash: str, path: str = config.VECTOR_STORE_PATH) -> bool:
    """Return whether an uploaded file has already been added to this FAISS index."""
    manifest_path = _uploaded_files_manifest_path(path)
    if not os.path.exists(manifest_path):
        return False

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    return file_hash in manifest


def remember_uploaded_file(
    file_hash: str,
    file_name: str,
    chunk_count: int,
    path: str = config.VECTOR_STORE_PATH,
) -> None:
    """Persist upload metadata separately so an identical file is not indexed twice."""
    os.makedirs(path, exist_ok=True)
    manifest_path = _uploaded_files_manifest_path(path)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

    manifest[file_hash] = {"name": file_name, "chunks": chunk_count}
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
