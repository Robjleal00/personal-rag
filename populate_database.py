import argparse  # Import argparse for command-line argument parsing
import os  # Import os for operating system related operations
import shutil  # Import shutil for file operations
from typing import List  # Import List from typing for type hinting
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Import PDF document loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import text splitter
from langchain.schema.document import Document  # Import Document schema
from get_embedding_function import get_embedding_function  # Import custom embedding function
from langchain_community.vectorstores import Chroma  # Import Chroma for vector storage

# Define the paths for Chroma database and data directory
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    """
    Main function to manage the database population process.
    """
    # Check if the database should be cleared (using the --reset flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents() -> List[Document]:
    """
    Load documents from the specified data directory.

    Returns:
    List[Document]: A list of loaded documents.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.

    Args:
    documents (List[Document]): The list of documents to split.

    Returns:
    List[Document]: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    """
    Add or update documents in the Chroma database.

    Args:
    chunks (List[Document]): The list of document chunks to add to the database.
    """
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or update the documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for document chunks based on their source and page number.

    Args:
    chunks (List[Document]): The list of document chunks to process.

    Returns:
    List[Document]: The list of document chunks with updated IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the chunk meta-data
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """
    Clear the Chroma database by deleting the directory.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
