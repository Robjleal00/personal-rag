from langchain_community.embeddings.ollama import OllamaEmbeddings  # Import OllamaEmbeddings from langchain_community for embedding functions
from langchain_community.embeddings.bedrock import BedrockEmbeddings  # Import BedrockEmbeddings from langchain_community for embedding functions

def get_embedding_function():
    """
    Function to get the embedding function for vector representations.

    Returns:
    embeddings: An instance of the embedding function.
    """
    # Uncomment the following lines to use BedrockEmbeddings with specific AWS credentials and region
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )

    # Use OllamaEmbeddings with the specified model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
