# Robert Leal's RAG (Retrieval-Augmented Generation) Project

Welcome to Robert Leal's Retrieval-Augmented Generation (RAG) project. This project utilizes Streamlit, Langchain, Chroma, and Ollama machine learning models to create an interactive query interface.

## Table of Contents
- Introduction
- Features
- Installation
- Usage
- Populating the Database
- Unit Testing
- Project Structure
- Contributing

## Introduction

This project leverages the power of vector databases, machine learning, and natural language processing to provide an interactive interface for querying a personal dataset (PDFs on information about Robert!). The interface is built with Streamlit, and the backend utilizes Ollama, Langchain, and Chroma for data retrieval, data storage, and response generation.

## Features

- **Interactive Query Interface**: Built with Streamlit for user-friendly interactions.
- **Vector Database Integration**: Utilizes Chroma for efficient data retrieval.
- **Machine Learning Models**: Incorporates advanced models for response generation from Ollama.
- **Memory Functionality**: Maintains conversation history for context-aware interactions.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Robjleal00/personal-rag.git
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### To run the Streamlit application, use the following command:
```bash
streamlit run query_data.py
```

## Populating the Database

### Before running the application, ensure that the database is populated with the necessary data. You can use the populate_database.py script for this purpose

1.  **Clear the Database**:
    ```bash
    python populate_database.py --reset
    ```
2. **Populate the Database**:
    ```bash
    python populate_database.py
    ```

## Running the Tests

### To run the unit tests, use the following command 
```bash
pytest test_rag.py
```

## Project Structure
    .
    ├── data                    # Directory containing the PDF documents
    ├── chroma                  # Directory for the Chroma vector database
    ├── query_data.py           # Main Streamlit application
    ├── populate_database.py    # Script to populate the database
    ├── get_embedding_function.py # Function to get the embedding function
    ├── test_rag.py             # Unit tests for the project
    ├── requirements.txt        # List of dependencies
    └── README.md               # This README file

## Contributing 

### Contributions are welcome! Feel free to open an issue or submit a pull request. Please ensure that 

1. **Fork the repository**

2. **Create a new branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. **Commit your changes**:
    ```bash
    git commit -m 'Add some feature'
    ```
4. **Push to the branch**:
    ```bash
    git push origin feature/your-feature-name
    ```
5. **Open a pull request**


### Explanation:

1. **Introduction**: A brief description of the project.
2. **Features**: A list of the key features of the project.
3. **Installation**: Step-by-step instructions on how to set up the project locally.
4. **Usage**: How to run the application and any setup steps required before running it.
5. **Populating the Database**": How to clear and populate the database. 
6. **Unit Testing**: Instructions for running the unit tests.
7. **Project Structure**: A tree view of the project files and directories to give an overview of the project layout.
8. **Contributing**: Guidelines for contributing to the project.





