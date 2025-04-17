from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
from typing import List
from pydantic import BaseModel, EmailStr
import pymongo
from pymongo.errors import ConnectionFailure, WriteError
from src.settings import settings
from src.utils.helper import *
from passlib.hash import sha256_crypt
import uuid
from datetime import datetime
import groq
from src.utils.crag_utils import MR_ZION_PLEASE_SELECT_ME_SPEAKING
from docling.document_converter import DocumentConverter





class LoginAuth(BaseModel):
    username: str
    password: str

class SignupAuth(BaseModel):
    email: EmailStr
    username: str
    password: str

class GetChatSessions(BaseModel):
    username: str

class CreateChat(BaseModel):
    user_id: str

class GetMessages(BaseModel):
    chat_id: str

class AddMessage(BaseModel):
    chat_id: str
    role: str
    content: str

class QueryKnowledgeBase(BaseModel):
    chat_id : str
    query: str


mongo_client = pymongo.MongoClient(settings.MONGO_URI)
db_name = "MR_ZION_PLEASE_SELECT_ME"
qdrant_collection = "MR_ZION_PREETY_PLS"
client = groq.Groq(api_key=settings.GROQ_API_KEY)
app = FastAPI()
processor = DocumentProcessor()

@app.post('/signup')
async def sign_up(request: SignupAuth):
    email = request.email
    username = request.username
    password = request.password

    collection_name = "MR_ZION_PLEASE_SELECT_ME_Users"
    if db_name not in mongo_client.list_database_names():
        db = mongo_client[db_name]
        db.create_collection(collection_name)
        try: 
            mongo_client[db_name][collection_name].insert_one({
                    "user_id": str(uuid.uuid4()),
                    "email" : email,
                    "username": username,
                    "password": sha256_crypt.hash(password),
                    "created_at": datetime.now().isoformat()
            })
            return { "detail" : "success", "status_code": 200}
        except ConnectionFailure:
            raise HTTPException(status_code=500, detail="Database connection error")
        except WriteError:
            raise HTTPException(status_code=500, detail="Database write error")
    else:
        try: 
            if mongo_client[db_name][collection_name].find_one({"$or": [{"email": email}, {"usermame": username}]}):
                raise HTTPException(status_code=400, detail= "Email or username already exists")
                
            mongo_client[db_name][collection_name].insert_one({
                    "user_id": str(uuid.uuid4()),
                    "email" : email,
                    "username": username,
                    "password": sha256_crypt.hash(password),
                    "created_at": datetime.now().isoformat()
                    })
            return { "detail" : "success", "status_code": 200}
        except ConnectionFailure:
            raise HTTPException(status_code=500, detail="Database connection error")
        except WriteError:
            raise HTTPException(status_code=500, detail="Database write error")
        

@app.post('/upload')
async def process(use_docling: str = Form(...), files: List[UploadFile] = None):
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger.info("awaiting file")
            _uploaded = await upload_files(files, temp_dir)
            system_logger.info("Upload successful")

            if _uploaded["status_code"] == 200:
                file_path = os.path.join(temp_dir, files[0].filename)
                if use_docling == "Yes":
                    
                    system_logger.info("Converting to markdown file")
                    converter = DocumentConverter().convert(file_path)
                    markdown_content = converter.document.export_to_markdown()
                    system_logger.info("Successfully converted to markdown")

                    markdown_filename = os.path.splitext(files[0].filename)[0] + ".md"
                    markdown_filepath = os.path.join(temp_dir, markdown_filename)

                    with open(markdown_filepath, "w", encoding="utf-8") as w:
                        w.write(markdown_content)

                    processed_documents = processor.process_and_split_document(markdown_filepath)
                    
                    system_logger.info("Processing documents...")
                    
                    vector_store = processor.create_vector_store(processed_documents)
                    
                    system_logger.info("Vector store created")
                else:
                    system_logger.info("Processing and splitting without converting to markdown")
                    
                    processed_documents = processor.process_and_split_document(file_path)
                    
                    system_logger.info("Processing documents...")
                    
                    vector_store = processor.create_vector_store(processed_documents)
                    

                return {"detail": "Documents processed and added to vector store", "status_code": 200}
            else:
                return _uploaded
    except Exception as e:
        system_logger.error(f"Could not process documents: {str(e)}")
        return {
            "detail": f"Could not process documents: {str(e)}",
            "status_code": 500
        }


@app.post('/signin')
async def sign_in(request:LoginAuth):
    username = request.username
    password = request.password

    collection_name = "MR_ZION_PLEASE_SELECT_ME_Users"

    user =  mongo_client[db_name][collection_name].find_one({"username": username})
    if not user:
        raise HTTPException(status_code=500, detail= "User does not exist")
    
    if not sha256_crypt.verify(password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")
    
    return {"detail": "Login successful", "status_code": 200, "user_id": user["user_id"],
        "username": user["username"]}


@app.get("/get_user_chats")
async def get_chats(request:GetChatSessions):
    username = request.username

    user_collection_name = "MR_ZION_PLEASE_SELECT_ME_Users"
    collection_name = "MR_ZION_PLEASE_SELECT_ME_Chat_Sessions"
    db = mongo_client[db_name]
    try:
        user_collection = db[user_collection_name]
        user = user_collection.find_one({"username": username})
        user_id = user["user_id"]
        
        chat_collection = db[collection_name]
        chats = list(chat_collection.find({"user_id": user_id}))
        
        formatted_chats = [
            {
                "_id" : str(chat["_id"]),
                "chat_id": chat["chat_id"],
                "user_id": chat["user_id"],
                "title": chat["title"],
                "created_at" : chat["created_at"],
                "updated_at" : chat["updated_at"]
            } for chat in chats
        ]
        return {"user_id": user_id, "chats": formatted_chats}
    
    except ConnectionFailure:
            raise HTTPException(status_code=500, detail="Database connection error")

@app.post("/create_chat")
async def create_chats(request:CreateChat):
    user_id = request.user_id
    collection_name = "MR_ZION_PLEASE_SELECT_ME_Chat_Sessions"
    chat_id = str(uuid.uuid4())
    chat_title = "New Convo with MR ZION PLEASE SELECT ME"
    try:
        mongo_client[db_name][collection_name].insert_one({
            "chat_id": chat_id,
            "user_id": user_id,
            "title": chat_title,
            "created_at" : datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        
        return{"chat_id": chat_id, "title": chat_title}
    except ConnectionFailure:
            raise HTTPException(status_code=500, detail="Database connection error")
    except WriteError:
        raise HTTPException(status_code=500, detail="Database write error")

@app.get("/get_chat_messages/{chat_id}")
async def get_messages(chat_id: str):
    
    collection_name = "MR_ZION_PLEASE_SELECT_ME_Chat_Messages"
    try:
        message_collection = mongo_client[db_name][collection_name]
        messages = list(message_collection.find({"chat_id":chat_id}).sort("timestamp", 1))

        formatted_messages = [
            {
                
                "role": message["role"],
                "content": message["content"],
                
            } for message in messages
        ]

        return{ "messages": formatted_messages}
    except ConnectionFailure:
        raise HTTPException(status_code=500, detail = "Database connection error")
    except Exception as e:
        raise HTTPException(status_code=500, detail= str({e}) )


@app.post("/add_message")
async def add_message(request: AddMessage):
    collection_name = "MR_ZION_PLEASE_SELECT_ME_Chat_Messages"
    system_prompt = """
                       Act as an intelligent summarizer. Condense the user's query into a concise, clear title that captures the essence of the content, making it easy to reference in a chat.
                         """
    try:
        mongo_client[db_name][collection_name].insert_one({
            "chat_id": request.chat_id,
            "role": request.role,
            "content": request.content,
            "timestamp": datetime.now().isoformat()
        })

        mongo_client[db_name]["MR_ZION_PLEASE_SELECT_ME_Chat_Sessions"].update_one(
            {"chat_id": request.chat_id},
            {"$set": {"updated_at": datetime.now().isoformat()}}
        )

        if request.role == "user":
            message_count = mongo_client[db_name][collection_name].count_documents({"chat_id": request.chat_id})
            if message_count <= 1:
                response = client.chat.completions.create(
                    model = "gemma2-9b-it",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.content}
                    ],
                    temperature = 0,
                )
                
                title_text = response.choices[0].message.content
                mongo_client[db_name]["MR_ZION_PLEASE_SELECT_ME_Chat_Sessions"].update_one(
                    {"chat_id": request.chat_id},
                    {"$set": {"title": title_text}}
                )
                


        return {"status": "success"}
    except ConnectionFailure:
        raise HTTPException(status_code=500, detail="Database connection error")
    except WriteError:
        raise HTTPException(status_code=500, detail="Database write error")


@app.post('/process_query')
async def process_query(request: QueryKnowledgeBase):
    chat_id = request.chat_id
    query = request.query
    userops_logger.info(
        f"""
        User Request:
        ------log------
        User data: {query}

        """
    )

    try:
        response = MR_ZION_PLEASE_SELECT_ME_SPEAKING(chat_id=chat_id, query=query)
        response = response.split("</think>", 1)[1].strip()
        llmresponse_logger.info(
            f"""
          LLM Response:
          -----log response-----
          Response: {response}
          """
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail="issue processing query")