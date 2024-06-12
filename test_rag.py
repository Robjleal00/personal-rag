import pytest # Importing the pytest library for unit testing
from unittest.mock import patch, MagicMock # Importing patch and MagicMock from unittest.mock for creating mock objects
import streamlit as st # Importing Streamlit for web app development
from query_data import query_rag # Importing the query_rag function from the query_data module
from langchain_community.llms.ollama import Ollama # Importing Ollama from langchain_community.llms for language model interactions


# Evaluation prompt template for comparing expected and actual responses
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# Function to mock Streamlit's session_state for testing
def mock_session_state():
    state = MagicMock()  # Create a mock object
    state.conversation_history = []  # Initialize conversation history
    return state

# Pytest fixture to setup and teardown mock session state before and after each test
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup before each test
    with patch('streamlit.session_state', mock_session_state()):
        yield
    # Teardown after each test
    

# Function to query and validate the response against the expected response
def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)  # Query the RAG model
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3")  # Initialize the model
    evaluation_results_str = model.invoke(prompt)  # Get evaluation result
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()  # Clean the result

    print(prompt)  # Print the prompt for debugging

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

# Test cases for validating the RAG model responses

def test_how_many_brothers():
    assert query_and_validate(
        question="How many brothers does Robert have?",
        expected_response="Three",
    )

def test_brothers_names():
    assert query_and_validate(
        question="What are Robert's brother's names?",
        expected_response="Ray, Ryan, and Rowan",
    )

def test_favorite_cuisine():
    assert query_and_validate(
        question="What is Robert's favorite cuisine?",
        expected_response="Texas brisket, loaded mashed potatoes, asparagus, mac and cheese, coleslaw",
    )

def test_karate_duration():
    assert query_and_validate(
        question="How long has Robert practiced Karate for and what rank is he?",
        expected_response="13 years and he is a third degree blackbelt",
    )

def test_taught_salsa():
    assert query_and_validate(
        question="Has Robert taught salsa?",
        expected_response="Robert taught salsa for one year",
    )

def test_has_girlfriend():
    assert query_and_validate(
        question="Does Robert have a girlfriend?",
        expected_response="Yes, Robert has a girlfriend named Rhona he met while dancing",
    )

def test_is_single():
    assert query_and_validate(
        question="Is Robert single?",
        expected_response="No, Robert is not single",
    )

def test_age():
    assert query_and_validate(
        question="When is Robert's birthday?",
        expected_response="February 19, 2000",
    )

def test_dogs_name():
    assert not query_and_validate(
        question="What is Robert's dog's name?",
        expected_response="Rowan",
    )

def test_arrived_in_netherlands(): 
    assert query_and_validate(
        question="At what age did Robert leave the United States?",
        expected_response="At 18 years old",
    )

if __name__ == "__main__":
    pytest.main()  