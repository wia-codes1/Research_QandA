import logging
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import json
from pydantic_settings import BaseSettings
import re
import unicodedata
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO)

# Pydantic Settings for HuggingFace API Key
class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str  # Define the environment variable for Hugging Face API token

    class Config:
        env_file = ".env"  # Automatically load environment variables from .env file

# Instantiate settings
settings = Settings()

# Verify Hugging Face API token
api_key = settings.HUGGINGFACEHUB_API_TOKEN
if api_key is None:
    raise EnvironmentError("HuggingFace API key is missing. Please set HUGGINGFACEHUB_API_TOKEN.")
else:
    logging.info(f"✅ HuggingFace API key loaded: {api_key}")

# Initialize HuggingFaceEmbeddings using the Hugging Face Hub API token
def get_embeddings_model():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Corrected initialization
    return embeddings

# Chunk PDF or Text content into smaller units (sentences or paragraphs)
def chunk_pdf_huggingface(documents: list) -> list:
    # Initialize the text splitter (you can adjust the chunk size and overlap as needed)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Extract and split the text from each document's page content
    chunks = []
    for doc in documents:
        # Ensure we're passing only the text (page_content) to the splitter
        content = doc.page_content
        chunks.extend(splitter.split_text(content))  # Split the text and add to the list

    logging.info(f"Number of chunks: {len(chunks)}")
    return chunks


# Generate embeddings for the text chunks
def generate_huggingface_embeddings(chunks: list) -> list:
    embeddings = get_embeddings_model()  # Get the HuggingFace embeddings model
    embeddings_list = [embeddings.embed_query(chunk) for chunk in chunks]  # Directly use the result
    return embeddings_list



# Store the embeddings in FAISS for similarity search
# Store the embeddings in FAISS for similarity search
def create_and_save_faiss_index(embeddings: list, chunks: list, index_file_path: str, chunk_file_path: str):
    # Convert the embeddings list to a numpy array
    embeddings = np.array(embeddings)  # Convert list to numpy array
    
    # Ensure embeddings is a 2D array (n_samples, n_features)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    
    # Create FAISS index (L2 distance)
    dim = embeddings.shape[1]  # Dimensionality of the embeddings
    faiss_index = faiss.IndexFlatL2(dim)  # Use L2 distance for the FAISS index
    faiss_index.add(embeddings)  # Add embeddings to FAISS index
    
    # Save the FAISS index to a file
    faiss.write_index(faiss_index, index_file_path)
    
    # Save the chunks to a file (this will allow us to load them later)
    with open(chunk_file_path, "w") as f:
        json.dump(chunks, f)
    
    logging.info(f"FAISS index and chunks saved to {index_file_path} and {chunk_file_path}")

def load_faiss_index(index_file_path: str, chunk_file_path: str):
    # Load the FAISS index from the file
    faiss_index = faiss.read_index(index_file_path)
    
    # Load the chunks from the file
    with open(chunk_file_path, "r") as f:
        chunks = json.load(f)
    
    logging.info(f"FAISS index and chunks loaded from {index_file_path} and {chunk_file_path}")
    
    return faiss_index, chunks

def retrieve_huggingface(query_embedding: list, index_file_path: str, chunk_file_path: str) -> list:
    """
    Retrieve relevant chunks from the FAISS index based on the query embedding.
    """
    try:
        # Ensure query_embedding is a numpy array with shape (1, embedding_size)
        query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape to (1, embedding_size)

        # Load the FAISS index and chunks
        faiss_index, chunks = load_faiss_index(index_file_path, chunk_file_path)
        
        # Ensure the query embedding dimension matches the FAISS index
        if query_embedding.shape[1] != faiss_index.d:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {faiss_index.d}.")
        
        # Perform search to get top-k most relevant chunks
        distances, indices = faiss_index.search(query_embedding, k=3)
        # Log first few chunks to inspect them
        logging.info(f"First few chunks: {chunks[:5]}")

        # Retrieve the most relevant chunks based on the indices
        relevant_chunks = [chunks[idx] for idx in indices[0]]
        return relevant_chunks
    except Exception as e:
        logging.error(f"Error in retrieving chunks: {e}")
        return f"Error processing the query: {str(e)}"


def improved_clean(documents, file_name):
    cleaned_docs = []
    for doc in documents:
        text = doc.page_content
        page_number = doc.metadata.get("page")  # Retrieve page number from metadata
        
        # Dummy metadata - replace with actual extraction logic
        title = "Sample Research Paper"  # This should be extracted from the PDF title or document header
        authors = "John Doe, Jane Smith"  # Extract authors from metadata or first pages
        publication_date = "2024-01-01"  # Extract publication date if available
        source_link = "https://arxiv.org/abs/123456"  # If available in the document metadata

        # 1) Unicode normalize (fix ligatures / odd widths)
        text = unicodedata.normalize("NFKC", text)

        # 2) Repair hyphenation across line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1-\2', text)

        # 3) Preserve paragraph breaks
        text = re.sub(r'\n{2,}', '<PAR>', text)  # mark paragraphs
        text = re.sub(r'[\r\n]+', ' ', text)     # flatten single newlines

        # 4) Remove bracketed numeric citations like [12]
        text = re.sub(r'\[\s*\d+\s*\]', '', text)

        # 5) Remove inline trailing citation digits glued to words (e.g., intelligence1.)
        text = re.sub(r'(?<=\w)(\d{1,3})(?=[\s\.,;:])', '', text)

        # 6) Remove long repeated digit runs (e.g., 1111, 1515151)
        text = re.sub(r'(\d)\1{3,}', '', text)

        # 7) Remove "Page 12" style markers
        text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)

        # 8) Strip control chars
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

        # 9) Normalize whitespace and restore paragraph breaks
        text = re.sub(r'[ \t\f\v]+', ' ', text)  # collapse horizontal whitespace
        text = text.replace('<PAR>', '\n\n')     # restore paragraphs
        text = re.sub(r' {2,}', ' ', text).strip()

        # 10) Targeted glyph fixes (extend if you see more)
        replacements = {
            'Trade-oƯ': 'Trade-off',
            'TradeoƯ': 'Trade-off',
            'oeƯ': 'oeff',
            'coeƯ': 'coeff',
            'Ư': 'f',  # keep last: broadest
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Assign metadata (file_name, page number, title, authors, date, source_link)
        doc.metadata["source"] = file_name
        doc.metadata["page"] = page_number
        doc.metadata["title"] = title
        doc.metadata["authors"] = authors
        doc.metadata["publication_date"] = publication_date
        doc.metadata["source_link"] = source_link

        # Determine section heading
        if page_number in ["0", "1"]:
            doc.metadata["section_heading"] = "authors"
        elif "references" in text.lower() or "bibliography" in text.lower():
            doc.metadata["section_heading"] = "references"
        else:
            doc.metadata["section_heading"] = "body"

        # Append the cleaned document with updated metadata
        cleaned_docs.append(Document(page_content=text, metadata=doc.metadata))

    return cleaned_docs

