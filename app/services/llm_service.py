from huggingface_hub import InferenceClient
import os

# Async function to generate an answer using the inference client
async def generate_answer_from_prompt(question: str, relevant_chunks: list) -> str:
    # Concatenate the relevant chunks to form the context for the LLM
    context = "\n".join(relevant_chunks)

    # Formulate the prompt with context and question
    prompt = f"Given the following context, answer the question. The context may contain important information about the topic, so please refer to it to generate the most accurate answer.\n\nContext:\n{context}\n\nQuestion: {question}"

    # Initialize the InferenceClient with the specified model
    client = InferenceClient(token=os.environ["HUGGINGFACE_API_TOKEN"]) # Make sure the API token is set in the environment
    model = client.get_model("Mistralai/Mistral-7B-Instruct-v0.2")  # Specify the model

    # Use the appropriate method to call the model (predict is correct for generating output)
    response = await client.predict({"inputs": prompt})

    # Extract the generated answer from the response
    generated_answer = response.get("generated_text", "")

    # Return the generated answer
    return generated_answer
