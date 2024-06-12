import streamlit as st  # Import Streamlit for creating the web interface
from langchain_community.vectorstores import Chroma  # Import Chroma for vector storage
from langchain.prompts import ChatPromptTemplate  # Import ChatPromptTemplate for creating prompt templates
from langchain_community.llms.ollama import Ollama  # Import Ollama for language model interaction
from get_embedding_function import get_embedding_function  # Import custom embedding function

# Define the path for Chroma database
CHROMA_PATH = "chroma"

# Define the template for the prompt
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    """
    Function to query the RAG (Retrieve and Generate) model.
    
    Args:
    query_text (str): The query text input by the user.

    Returns:
    str: The formatted response from the model.
    """
    # Prepare the embedding function and initialize the Chroma database
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database for similar documents
    results = db.similarity_search_with_score(query_text, k=12)

    # Create the context text by joining the contents of the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt using the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the language model and get the response
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Extract the source IDs from the results
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    # Update the conversation history in the session state
    st.session_state.conversation_history.append({"role": "user", "content": query_text})
    st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
    
    return formatted_response

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set the title of the Streamlit app
    st.title("Robert Leal RAG")

    # Define the path to the image file
    image_path = 'robert.jpg'

    # Display the image
    st.image(image_path, use_column_width=True)
    
    # Initialize conversation history in session state if it doesn't exist
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Create a text input box for the query
    query_text = st.text_input("Enter your query:")
    
    # Create a submit button for the query
    if st.button("Submit"):
        if query_text:
            # Process the query and display the response
            response = query_rag(query_text)
            st.write(response)
        else:
            st.write("Please enter a query.")

    # Display the conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for entry in st.session_state.conversation_history:
            if entry['role'] == 'user':
                st.text(f"User: {entry['content']}")
            elif entry['role'] == 'assistant':
                st.text(f"Assistant: {entry['content']}")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
