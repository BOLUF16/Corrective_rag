import streamlit as st
import requests
from datetime import datetime



BACKEND_URL = "http://127.0.0.1:5000"

def send_signup_request(email:str, username:str, password:str):
    url = f"{BACKEND_URL}/signup"
    payload = {
        "email": email,
        "username": username,
        "password": password
    }
    response = requests.post(url=url,json=payload)
    return response

def send_signin_request(username:str, password:str):
    url = f"{BACKEND_URL}/signin"
    payload = {
        "username": username,
        "password": password
    }
    response = requests.post(url=url,json=payload)
    return response

def get_chat_request(username:str):
    url = f"{BACKEND_URL}/get_user_chats"
    payload = {
        "username": username
    }
    response = requests.get(url=url,json=payload)
    return response

def create_new_chat(user_id: str):
    url = f"{BACKEND_URL}/create_chat"
    payload = {
        "user_id": user_id
    }
    response = requests.post(url=url, json=payload)
    return response

def get_chat_messages(chat_id: str):
    url = f"{BACKEND_URL}/get_chat_messages/{chat_id}"
    response = requests.get(url)
    return response

def embed_link(document_url: str):
    api_endpoint = f"{BACKEND_URL}/embed_doc_url"
    payload = {"url": document_url}
    response = requests.post(url=api_endpoint, json=payload)
    return response

def add_message(chat_id: str, role: str, content: str):
    url = f"{BACKEND_URL}/add_message"
    payload = {
        "chat_id": chat_id,
        "role": role,
        "content": content
    }
    response = requests.post(url=url, json=payload)
    return response

def process_query(chat_id: str, query: str):
    url = f"{BACKEND_URL}/process_query"
    payload = {
        "chat_id": chat_id,
        "query": query
    }
    response = requests.post(url=url, json=payload)

    return response

def upload_files(files, use_docling: str = "No"):
    url = f"{BACKEND_URL}/upload"
    files_data = [("files", file) for file in files]
    data = {"use_docling": use_docling}
    response = requests.post(url, files=files_data, data=data)
    return response

def main():
    
    if "signed_in" not in st.session_state:
        st.session_state.signed_in = False
        st.session_state.username = None

    if not st.session_state.signed_in:
        st.title("Welcome to MR ZION PLEASE SELECT ME CORRECTIVE RAG")
        menu = ["Sign In", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Sign Up":
            st.subheader("Sign Up")
            email = st.text_input("email")
            username = st.text_input("username")
            password = st.text_input("password")
            if st.button("Create account"):
                with st.spinner("Creating...."):
                    response = send_signup_request(email, username, password)
                    if response.status_code == 200:
                        st.success("Account Successfully Created! Proceed to Sign In")
                    else:
                        st.error(response.json().get("detail", "Sign Up error"))
       
        elif choice == "Sign In":
            st.subheader("Sign In")
            username = st.text_input("username")
            password = st.text_input("password")
            if st.button("Sign In"):
                with st.spinner("loading..."):
                    response = send_signin_request(username, password)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.signed_in = True
                        st.session_state.username = username
                        st.session_state.user_id = data["user_id"]
                        st.success(f"Logged in as {username}")
                        st.rerun()
                    else:
                        st.error(response.json().get("detail", "Sign In unsuccessful"))
                                   
    else:
        st.title(f"Welcome to MR ZION PLEASE SELECT ME , {st.session_state.username}!")

        if "chats_sessions" not in st.session_state:
            with st.spinner("Loading chats..."):
                response = get_chat_request(st.session_state.username)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chats_sessions = data.get("chats", [])
                else:
                    st.error("Failed to load chats")
                    st.session_state.chats_sessions = []

        with st.sidebar:
            with st.expander("***Document Upload***"):
                st.markdown("### Upload Document")
                upload_document = st.file_uploader("Choose document / documents", accept_multiple_files=True)
                # use_docling = st.radio("Use docling for better parsing of document", 
                #                     ["Yes", "No"], 
                #                     captions=["Better document parsing but increases the time taken to embed", 
                #                             "Default document parsing with reduced time taken to embed"])
                
                if st.button("Process document"):
                    # if upload_document:
                        with st.spinner("Embedding..."):
                            response = upload_files(upload_document)
                            if response.status_code == 200:
                                st.success("Document embedding successful")
                            else:
                                st.error(f"Embedding Unsuccessful: {response.text}")
            st.divider()
            
            st.title("Your Chats")  

            if st.button("Start New Chat", use_container_width = True):
                with st.spinner("Creating new chat..."):
                    response = create_new_chat(st.session_state.user_id)
                    if response.status_code == 200:
                        data = response.json()
                        new_chat = {
                            "chat_id": data["chat_id"],
                            "title": data["title"],
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat()
                        }
                        st.session_state.chats_sessions.append(new_chat)
                        st.session_state.current_chat_id = data["chat_id"]
                        st.session_state.current_chat_title = data["title"]
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Failed to create new chat")
            
            st.divider()

            if st.session_state.chats_sessions:
                for chat in sorted(st.session_state.chats_sessions, key=lambda x:datetime.fromisoformat(x["updated_at"]),
                                   reverse=True):
                    if st.button(chat["title"], use_container_width = True, key=f"chat_{chat['chat_id']}"):
                        st.session_state.current_chat_id = chat["chat_id"]
                        st.session_state.current_chat_title = chat["title"]

                        with st.spinner("Loading conversation..."):
                            response = get_chat_messages(chat["chat_id"])
                            if response.status_code == 200:
                                st.session_state.messages = response.json().get("messages", [])
                            else:
                                st.error("Failed to load messages")
                                st.session_state.messages = []
                        
                        st.rerun()
                
            else:
                st.info("No chats found. Start a new chat!")

        if "current_chat_id" in st.session_state:
            st.header(f"Chat: {st.session_state.current_chat_title}")

            if "messages" not in st.session_state:
                with st.spinner("Loading conversation..."):
                    response = get_chat_messages(st.session_state.current_chat_id)
                    if response.status_code == 200:
                        st.session_state.messages = response.json().get("messages", [])
                    else:
                        st.error("Failed to load messages")
                        st.session_state.messages = []
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if prompt := st.chat_input("Type question.."):

                with st.chat_message("user"):
                    st.write(prompt)
                
                add_message(st.session_state.current_chat_id, "user", prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking...."):
                        response = process_query(st.session_state.current_chat_id, prompt)
                        if response.status_code == 200:
                            ai_response = response.json().get("response", "Sorry, I couldn't process your request.")
                            st.write(ai_response)


                            add_message(st.session_state.current_chat_id, "assistant", ai_response)
                        else:
                            st.error("Failed to process your query")
                
                response = get_chat_messages(st.session_state.current_chat_id)
                if response.status_code == 200:
                    st.session_state.messages = response.json().get("messages", [])
            
                response = get_chat_request(st.session_state.username)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chats_sessions = data.get("chats", [])

                st.rerun()

        else:
            st.info("select a chat from the sidebar or create a new one to get started!")


        
                    

                    



            

                    




       
if __name__ == "__main__":
    main()
