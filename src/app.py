from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse
import tempfile
from typing import List
from src.utils.helper import *
from src.utils.crag_process import *
from contextlib import asynccontextmanager



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.document_processor = DocumentProcessor()
    app.state.vector_store = None
    yield
    

app = FastAPI(lifespan=lifespan)


@app.post('/upload')
async def process(files: List[UploadFile] = None):
    global vector_store

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger.info("awaiting file")
            _uploaded = await upload_files(files, temp_dir)
            system_logger.info("Upload successful")

            if _uploaded["status_code"] == 200:
                file_path = os.path.join(temp_dir, files[0].filename)
                
                system_logger.info(f"Processing file to: {file_path}")
                
                processor = app.state.document_processor
                
                processed_documents = processor.process_and_split_document(file_path)
                
                system_logger.info(f"Processed documents: {processed_documents}")
                
                app.state.vector_store = processor.create_vector_store(processed_documents)
                
                system_logger.info("Vector store created")

                return {"detail": "Documents processed and added to vector store", "status_code": 200}
            else:
                return _uploaded
    except Exception as e:
        system_logger.error(f"Could not process documents: {str(e)}")
        return {
            "detail": f"Could not process documents: {str(e)}",
            "status_code": 500
        }

@app.get('/generate')
async def query(request:Request):
    if app.state.vector_store is None:
        system_logger.info("No documents uploaded yet")
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    

    query = await request.json()
    userops_logger.info(
        f"""
        User Request:
        ------log------
        User data: {query}

        """
    )

    try:
        corrective_rag = CorrectiveRAG(
            model= query["model"],
            temperature = query["temperature"]
        )
        
        system_logger.info("Starting CRAG process....")
        response = corrective_rag.crag_process(
            query = query["question"],
            vector_store= app.state.vector_store
        )
        system_logger.info("CRAG process completed sucessfully")

        llmresponse_logger.info(
            f"""
          LLM Response:
          -----log response-----
          Response: {response}
          """
        )
        
        return PlainTextResponse(content=response)
    except Exception as e:
        # Log and return a generic error response
        system_logger.error(f"Failed to generate response: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"detail": f"Failed to generate response: {str(e)}", "status_code": 500},
            status_code=500
        )
        
    

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)
            
                
