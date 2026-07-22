# AI-Powered Study Assistant

The AI-Powered Study Assistant is an intelligent system that helps students interact with their study materials efficiently. Using Retrieval-Augmented Generation, it allows users to upload documents and ask questions in natural language. The system processes documents by extracting, chunking, and converting text into embeddings stored in a vector database. When a query is asked, it retrieves relevant content and generates accurate, context-based answers using a language model. This reduces manual searching, saves time, and improves learning. The system is scalable, user-friendly, and can be extended with features like summarization, voice interaction, and personalized recommendations.

## Features

- Upload PDF and TXT documents
- Build a FAISS vector database from study material
- Ask natural language questions about uploaded documents
- Retrieve relevant context before generating answers
- Streamlit-based user interface
- Easy to extend with more document types and LLM backends

## Project Structure

```text
project_AI_Study_Assistant/
├── app.py
├── ingest.py
├── utils.py
├── README.md
├── requirements.txt
├── data/
└── store/