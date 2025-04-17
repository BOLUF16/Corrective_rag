import os
from pathlib import Path
from werkzeug.utils import secure_filename
from src.exception.operationhandler import system_logger, userops_logger, llmresponse_logger
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader,  Docx2txtLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant, QdrantVectorStore
from fastapi import HTTPException
from langchain_community.tools import TavilySearchResults
from src.settings import settings


allowed_files = ["txt", "csv", "pdf", "doc", "docx"]

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
        self.qdrant_url = settings.QDRANT_URL
        self.qdrant_key = settings.QDRANT_API_KEY
        


    def process_and_split_document(self, file_path: str):
        try:
            file_extension = Path(file_path).suffix.lower()

            # Choose the appropriate loader based on file type
            if file_extension == ".txt":
                loader = TextLoader(file_path)
            # if file_extension == ".md":
            #     loader = DoclingLoader(
            #         file_path = file_path,
            #         export_type=ExportType.MARKDOWN,
            #         chunker=HybridChunker(tokenizer="BAAI/bge-small-en-v1.5"))
                
            #     document = loader.load()
            #     splitter = MarkdownHeaderTextSplitter(
            #         headers_to_split_on=[("#", "Header 1"),
            #         ("##", "Header 2"),
            #         ("###", "Header 3"),]
            #     )
            #     document = loader.load()
            #     splits = [split for doc in document for split in splitter.split_text(doc.page_content)]

            #     return splits
            
            elif file_extension == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path)
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
    def get_embedding():
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.HF_KEY,
            model_name="BAAI/bge-small-en-v1.5"
        )

    def create_qdrant_client(self):
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

    def create_qdrant_collection(self):
        client = self.create_qdrant_client()
        try:
            # Check if collection exists
            collections = client.get_collections()
            if settings.QDRANT_COLLECTION not in [col.name for col in collections.collections]:
                client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                self.system_log.info(f"Created new collection: {settings.QDRANT_COLLECTION}")
            else:
                # Check if existing collection has expected vector configuration
                collection = client.get_collection(collection_name=settings.QDRANT_COLLECTION)
                collection_config = collection.config.params.vectors
                
                if collection_config.size != 384 or collection_config.distance != Distance.COSINE:
                    self.system_log.warning(
                        f"Collection {collection} has unexpected configs:" f"Vector size {collection_config.size} and distance {collection_config.distance}"
                    )
                self.system_log.info(f"Using existing collection: {settings.QDRANT_COLLECTION}")
        except Exception as e:
            self.system_log.error(f"Error creating Qdrant collection: {str(e)}")
            raise

        return client

    def create_vector_store(self, documents):

        client = self.create_qdrant_collection()

        vectorstore = Qdrant(
                    client=client,
                    embeddings= self.get_embedding(),
                    collection_name= settings.QDRANT_COLLECTION
                    )
        
        vectorstore.add_documents(documents)
        self.system_log.info(f"vector store succesfully created")
        
        return {"status":"success"}

    

