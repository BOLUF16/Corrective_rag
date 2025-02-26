import os
from pathlib import Path
from werkzeug.utils import secure_filename
from src.exception.operationhandler import system_logger, userops_logger, llmresponse_logger
from src.config.appconfig import *
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, JSONLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant, QdrantVectorStore
from fastapi import HTTPException
from langchain_community.tools import TavilySearchResults


allowed_files = ["txt", "csv", "json", "pdf", "doc", "docx"]

def allowed_file(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_files

def file_checks(files):
    if not files:
        return {
            "detail" : "No file found",
            "status_code" : 400
        }
    for file in files:
        if not file and file.filenames ==  "":
            return {
            "detail" : "No file found",
            "status_code" : 400
        }
        if not allowed_file(file.filename):
            print(file.filename)
            return {
                "detail": f"File format not supported. use any of {allowed_files}",
                "status_code": 415
            }
    
    return {
        "detail" : "success",
        "status_code": 200
    }

async def upload_files(files, temp_dir):
    checks = file_checks(files)
    if checks["status_code"] != 200:
        return checks
    try:
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir,filename)

            file_obj = await file.read()

            with open(file_path, "wb") as buffer:
                buffer.write(file_obj)
            
            print(f"File saved to: {file_path}")  # Debugging statement
            system_logger.info(f"File successfully uploaded to {file_path}")
        return {
                "detail" : "Upload completed",
                "status_code": 200
            }
        
    except FileNotFoundError:
        system_logger.exception("File not found during upload.")
        return {"detail": "File handling error", "status_code": 500}
    except Exception as e:
        system_logger.exception(f"Unexpected error during upload: {e}")
        return {"detail": "Internal server error", "status_code": 500}
    



class DocumentProcessor:

    def __init__(self, chunk_size:int = 1024, chunk_overlap:int = 50):
       
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.system_log = system_logger
        self.qdrant_url = "https://398e383f-3b1b-4900-b002-88776d6c621f.us-east4-0.gcp.cloud.qdrant.io:6333"
        self.qdrant_key = qdrant_key
        self.collection_name = "corrective_rag" 


    def process_and_split_document(self, file_path: str):
        try:
            file_extension = Path(file_path).suffix.lower()
            self.system_log.info(f"Loading file: {file_path}")

            # Choose the appropriate loader based on file type
            if file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path)
            elif file_extension == ".json":
                loader = JSONLoader(file_path)
            elif file_extension in [".doc", ".docx"]:
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Load and split the document
            documents = loader.load()
            self.system_log.info("File loaded successfully.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)

            # Clean and add metadata to documents
            cleaned_texts = [
                Document(
                    page_content=doc.page_content.replace('\t', ' ').strip(),
                    metadata={"source": file_path, "page": doc.metadata.get("page", 0)}
                )
                for doc in texts
            ]

            return cleaned_texts

        except Exception as e:
            self.system_log.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not process documents: {str(e)}")

    @staticmethod
    def get_embedding(api_key: str):
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="BAAI/bge-small-en-v1.5"
        )

    def create_qdrant_client(self):
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

    def create_qdrant_collection(self):
        client = self.create_qdrant_client()
        try:
            # Check if collection exists
            collections = client.get_collections()
            if self.collection_name not in [col.name for col in collections.collections]:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                self.system_log.info(f"Created new collection: {self.collection_name}")
            else:
                self.system_log.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            self.system_log.error(f"Error creating Qdrant collection: {str(e)}")
            raise

    def create_vector_store(self, documents):
        client = self.create_qdrant_client()
        self.create_qdrant_collection()

        return QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.get_embedding(hf_key),
            url=self.qdrant_url,
            api_key=self.qdrant_key,
            collection_name=self.collection_name
        )
    

def web_search(query: str) -> list:
    """
    Perform a web search using the Tavily API and process the results.

    Args:
        query (str): The search query.

    Returns:
        list: A list of strings containing the search results. Each string includes the title, content, and link of a search result.
              If the search fails or no results are found, returns a list with a single dictionary containing the query.
    """
    try:
        system_logger.info(f"Initiating web search for query: {query}")

        # Initialize the Tavily search tool
        tool = TavilySearchResults(
            api_key=tavily_key,  
            max_results=3,
            search_depth="advanced"
        )

        # Execute the search
        try:
            search_results = tool.invoke({"query": query})
        except Exception as e:
            system_logger.error(f"Failed to execute search: {str(e)}")
            return [{"key": {"question": query}}]

        # Check if results are empty
        if not search_results:
            system_logger.warning("No search results found.")
            return [{"key": {"question": query}}]

        # Process and format the search results
        system_logger.info(f"Processing {len(search_results)} search results...")
        web_results = [
            f"Content: {result.get('content', 'No content')}\n"
            f"Link: {result.get('url', 'No link')}\n"
            for result in search_results
        ]

        return web_results

    except Exception as e:
        system_logger.error(f"An unexpected error occurred: {str(e)}")
        return [{"key": {"question": query}}]







    


    
    

    
