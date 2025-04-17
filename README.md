# Corrective RAG

## Overview
Corrective RAG (Retrieval-Augmented Generation) is an enhanced approach to conversational AI that refines and corrects responses by improving the retrieval and generation pipeline. Unlike traditional RAG systems, which can sometimes provide incorrect or misleading answers due to imperfect retrieval, Corrective RAG aims to identify, analyze, and correct its own mistakes dynamically. The goal is to improve the accuracy and reliability of responses by iterating on retrieved documents and generated outputs.

## Project Objective
The main objective of this project is to develop a Corrective RAG system that:
- Retrieves relevant documents from a knowledge base using vector search.
- Evaluates the correctness of responses by detecting inconsistencies and inaccuracies.
- Grades and refines retrieval results to provide more accurate answers.
- Iteratively enhances response quality using corrective mechanisms.

## Technologies Used
This project leverages a combination of powerful tools and frameworks to build an effective Corrective RAG system:

- **LangChain**: Provides the framework for building RAG pipelines, handling document retrieval, and integrating various AI components.
- **Langgraph**: Provides the orchestration framework for managing multi-step workflows
- **Qdrant**: A high-performance vector database used for storing and searching document embeddings efficiently.
- **Tavily**: Enables enhanced retrieval by augmenting searches with relevant online information.
- **FastAPI**: Serves as the backend framework for API endpoints, ensuring fast and efficient interaction between components.
- **Streamlit** : an interactive front-end for users to interact with the Corrective RAG system.

## Current Status
At this stage, the system is under active development, and the primary challenge is ensuring that Corrective RAG produces accurate responses. Ironically, despite its corrective nature, it is currently giving incorrect responses. The current focus is on troubleshooting and improving:
- The retrieval mechanism to ensure the most relevant documents are retrieved.
- The ranking system to prioritize high-quality information.
- The response correction module to detect and fix inconsistencies.





