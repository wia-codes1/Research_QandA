from fastapi import APIRouter
from app.services.huggingface_service import retrieve_huggingface, generate_huggingface_embeddings
from pydantic import BaseModel
from app.services.llm_service import generate_answer_from_prompt  # Assuming we have this function

# Define the query model which will accept the question
class QueryModel(BaseModel):
    question: str
    index_file_path: str  # Path to the FAISS index file
    chunk_file_path: str  # Path to the chunks file

# Initialize the FastAPI Router
query_router = APIRouter()

@query_router.post("/")
async def query(query: QueryModel):
    # Step 1: Generate query embedding using Hugging Face embeddings
    query_embedding = generate_huggingface_embeddings([query.question])  # Convert query to embeddings
    if not query_embedding:
        return {"error": "Failed to generate embeddings for the query."}

    # Step 2: Retrieve relevant chunks using FAISS based on the query embedding
    relevant_chunks = retrieve_huggingface(query_embedding[0], query.index_file_path, query.chunk_file_path)  # Use dynamic paths from query

    if not relevant_chunks:
        return {"error": "No relevant chunks found for the query."}

    # Step 3: Generate the answer from the LLM
    answer = await generate_answer_from_prompt(query.question, relevant_chunks)  # Only call once for answer

    if not answer:
        return {"error": "Failed to generate an answer from the LLM."}

    # Return the relevant chunk(s) and the answer as the response
    return {"response": relevant_chunks, "answer": answer}
