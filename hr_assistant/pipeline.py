import hashlib

from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.splitter import split_into_chunks
from hr_assistant.document_loader import load_document, load_pdf
from hr_assistant.llm import get_llm
from hr_assistant.tools import create_search_tool
from hr_assistant.vector_store import(
    build_vector_store,
    vector_store_exists,
    load_vector_store,
    save_vector_store,
    get_retriever,
    uploaded_file_exists,
    remember_uploaded_file,
)


def build_vector_for_document(file_path:str=config.DATA_FILE_PATH):
    if vector_store_exists():
        print("store Already Exist, Load it")
        return load_vector_store()

    print("No store exist, build it from scratch")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    return vector_store

def build_hr_assistant(file_path:str=config.DATA_FILE_PATH):
    config.check_api_keys()

    vector_store = build_vector_for_document(file_path)
    return build_assistant_from_vector_store(vector_store)


def build_assistant_from_vector_store(vector_store):
    """Create an HR assistant that searches the supplied vector store."""
    retriever = get_retriever(vector_store)
    search_tool= create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm,[search_tool])

    return agent


def add_pdf_to_hr_assistant(file_name: str, file_bytes: bytes):
    """Parse an uploaded PDF, append its chunks to FAISS, and return a refreshed agent."""
    config.check_api_keys()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    vector_store = build_vector_for_document()
    if uploaded_file_exists(file_hash):
        return build_assistant_from_vector_store(vector_store), 0, True

    documents = load_pdf(file_name, file_bytes)
    chunks = split_into_chunks(documents)
    vector_store.add_documents(chunks)
    save_vector_store(vector_store)
    remember_uploaded_file(file_hash, file_name, len(chunks))

    return build_assistant_from_vector_store(vector_store), len(chunks), False


def ask(agent, question:str)->str:
    """Ask the agent a question and
    return its final answer as plain text."""
    response = agent.invoke({"messages":[{"role":"user","content":question}]})
    return response["messages"][-1].content
