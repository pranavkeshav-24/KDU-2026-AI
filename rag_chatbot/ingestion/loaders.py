"""
Document loaders for PDF files and web URLs.
Provides a unified List[Document] interface for the rest of the pipeline.
"""
import os
import tempfile
from typing import List

import bs4
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file using PyMuPDFLoader.
    Returns one Document per page with page metadata preserved.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List[Document] with page content and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Enrich metadata with doc title from filename
    title = os.path.splitext(os.path.basename(file_path))[0]
    for doc in docs:
        doc.metadata.setdefault('doc_title', title)
        doc.metadata.setdefault('source', file_path)

    return docs


def load_url(url: str) -> List[Document]:
    """
    Load a web page / blog article using WebBaseLoader + BeautifulSoup.
    Strips navigation, ads, and footer noise — keeps only semantic content tags.

    Args:
        url: Full URL including scheme (http/https).

    Returns:
        List[Document] with cleaned page content.
    """
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            'parse_only': bs4.SoupStrainer(
                ['article', 'main', 'div', 'p', 'h1', 'h2', 'h3', 'section']
            )
        },
    )
    docs = loader.load()

    # Enrich metadata
    for doc in docs:
        doc.metadata.setdefault('source', url)
        doc.metadata.setdefault('doc_title', url.split('/')[-1] or url)

    return docs


def save_uploaded_file(uploaded_file, upload_dir: str = './uploads') -> str:
    """
    Save a Streamlit UploadedFile object to disk.

    Args:
        uploaded_file: Streamlit UploadedFile object with .name and .read().
        upload_dir: Directory to save the file.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    return file_path
