from io import BytesIO

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader

from hr_assistant import config

def load_document(file_path:str = config.DATA_FILE_PATH):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_pdf(file_name: str, file_bytes: bytes) -> list[Document]:
    """Extract readable text from an uploaded PDF, preserving page metadata."""
    reader = PdfReader(BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_number},
                )
            )

    if not documents:
        raise ValueError(
            "No readable text was found in this PDF. Scanned PDFs need OCR before upload."
        )

    return documents
