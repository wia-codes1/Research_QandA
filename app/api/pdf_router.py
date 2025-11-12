import os
import os
import tempfile
import logging
import fitz  # PyMuPDF
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.huggingface_service import chunk_pdf_huggingface, generate_huggingface_embeddings, create_and_save_faiss_index, improved_clean
from langchain_core.documents import Document


# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Define a permanent directory to save files (Change this to your desired path)
PERMANENT_DIR = "./data"  # Permanent directory for FAISS index and chunks
os.makedirs(PERMANENT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

pdf_router = APIRouter()

@pdf_router.post("/upload/")  # Endpoint for file upload
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Read the file content
        content = await file.read()
        logging.info(f"Received file with {len(content)} bytes")

        # Use PyMuPDF (fitz) to extract text from the PDF content
        with fitz.open(stream=content, filetype="pdf") as pdf_document:
            docs = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)  # Load a page
                page_text = page.get_text() # Extract text from the page
                doc = Document(page_content=page_text, metadata={"page": page_num})
                docs.append(doc)
    
        cleaned_docs = improved_clean(docs, file.filename)
 
        # Log the first 100 characters of the extracted text for verification
        logging.info(f"Decoded text preview: {cleaned_docs[:100]}")

        # Chunk the PDF content using Hugging Face-based method (using `RecursiveCharacterTextSplitter`)
        chunks = chunk_pdf_huggingface(cleaned_docs)
        logging.info(f"Number of chunks: {len(chunks)}")

        # Generate embeddings for the chunks using Hugging Face model
        embeddings = generate_huggingface_embeddings(chunks)
        logging.info(f"Generated embeddings: {embeddings[:5]}")  # Log first 5 embeddings for inspection

        # Create a permanent file for the FAISS index
        with tempfile.NamedTemporaryFile(delete=False, dir=PERMANENT_DIR) as temp_index_file:
            index_file_path = temp_index_file.name
            chunk_file_path = index_file_path + "_chunks.json"  # Create a separate file for chunks

            # Save FAISS index and chunks to the permanent files
            create_and_save_faiss_index(embeddings, chunks, index_file_path, chunk_file_path)

        # Return the paths of the saved files
        return {"message": "PDF processed successfully", "index_file": index_file_path, "chunk_file": chunk_file_path}

    except Exception as e:
        # Log the full stack trace for debugging
        logging.error(f"Error processing the file: {traceback.format_exc()}")  # Log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
