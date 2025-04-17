from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import List, Tuple, Optional, Dict, Any
from langchain_community.tools import TavilySearchResults
from src.exception.operationhandler import system_logger
from src.settings import settings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import pymongo
from langgraph.graph import MessagesState
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langchain.schema import Document


class GraphState(MessagesState):
    query: str
    conversation_history: List[Dict[str, str]] = []
    documents: Optional[List[Document]] = None
    llm_response: Optional[str] = None

class RetrievalEvaluator(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of the document to the query. the score should be between 0 and 1.")

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")

llm = ChatGroq(model='deepseek-r1-distill-llama-70b',temperature=0.2, api_key=settings.GROQ_API_KEY)
embedding = HuggingFaceInferenceAPIEmbeddings(api_key=settings.HF_KEY,model_name="BAAI/bge-small-en-v1.5")
mongo_client = pymongo.MongoClient(settings.MONGO_URI)
checkpointer = MongoDBSaver(client=mongo_client, db_name="MR_ZION_PLEASE_SELECT_ME", collection_name="graph_states")
client = QdrantClient(url= settings.QDRANT_URL, api_key= settings.QDRANT_API_KEY)

def create_qdrant_collection():
        try:
            # Check if collection exists
            collections = client.get_collections()
            if settings.QDRANT_COLLECTION not in [col.name for col in collections.collections]:
                client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                system_logger.info(f"Created new collection: {settings.QDRANT_COLLECTION}")
            else:
                # Check if existing collection has expected vector configuration
                collection = client.get_collection(collection_name=settings.QDRANT_COLLECTION)
                collection_config = collection.config.params.vectors
                
                if collection_config.size != 384 or collection_config.distance != Distance.COSINE:
                    system_logger.warning(
                        f"Collection {collection} has unexpected configs:" f"Vector size {collection_config.size} and distance {collection_config.distance}"
                    )
                system_logger.info(f"Using existing collection: {settings.QDRANT_COLLECTION}")
        except Exception as e:
            system_logger.error(f"Error creating Qdrant collection: {str(e)}")
            raise

        return settings.QDRANT_COLLECTION

collection_name = create_qdrant_collection()
vectorstore = QdrantVectorStore(client=client,collection_name=collection_name,embedding=embedding)


################################ GRAPH #######################################

def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:

    query = state["query"]

    system_logger.info("Retrieving documents for query: %s", query)
    retriever = vectorstore.as_retriever()
    documents = retriever.get_relevant_documents(query)
    system_logger.info("Retrieved %s documents")

    return {"documents" : documents, "query": query}


def grade_document(state: Dict[str, Any]) -> Dict[str, Any]:
    
    query = state["query"]
    documents = state["documents"]

    system_logger.info("Grading document for query: %s", query)
    Prompt = PromptTemplate(
        template= """You are a grader assessing the relevance of a retrieved document to a user question.
                    This is not a stringent evaluation—the goal is to filter out clearly irrelevant or erroneous retrievals.
                    If the document contains keywords or conveys a semantic meaning related to the user query, consider it relevant.
                    On a scale from 0 to 1, how relevant is the following document to the query?

                    Query: {query}
                    Document: {documents}
                    Relevance score (0 to 1):""",
        
        input_variables=["query", "documents"]
    )
    
    graded_doc = []
    for doc in documents:    
        chain = Prompt | llm.with_structured_output(RetrievalEvaluator)
        input_variables = {"query":query, "documents": doc}
        result = chain.invoke(input_variables).relevance_score
        if result > 0.5:
            graded_doc.append(doc)
            system_logger.info("Relevant documennt score: %s", result)
        else:
            system_logger.info("Irrelevant documennt score: %s", result)
            continue



    return {"documents": graded_doc, "query": query}

def decide_to_generate(state: Dict[str, Any]) ->Dict[str, Any]:

    system_logger.info("Deciding to generate....")
    query = state["query"] 
    graded_doc = state["documents"]

    if not graded_doc:
        system_logger.info("Documents are not relevant, proceeding to rewrite query")
        return "rewrite_query" 
    else:
        system_logger.info("Documents are relevant, proceeding to generate response")
        return "generate_response"


def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
    
    query = state["query"]
    document = state["documents"]

    system_logger.info("Rewriting query: %s", query)
    prompt = PromptTemplate(
        template="""You are a question rewriter that improves an input query to make it more suitable for a web search.
                    Analyze the underlying semantic intent of the question and rewrite it in a clearer, more complete form that would yield better results when used in a web search engine.

                    Original query: {query}
                    Rewritten query:""",
        
        input_variables=["query"]
        )
    
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query": query}
    rewritten_query = chain.invoke(input_variables).query.strip()
    system_logger.info("Rewritten query: %s", rewritten_query)
    
    return {"documents": document, "query": rewritten_query}


def web_search(state:Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a web search using the Tavily API and process the results.

    Args:
        query (str): The search query.

    Returns:
        Dict: it return a dictionary containing the query and the webresult turned into document.
              If the search fails or no results are found, returns a single dictionary containing the query.
    """
    try:
        
        
        query = state["query"]
        documents = []
        system_logger.info(f"Initiating web search for query: {query}")
        # Initialize the Tavily search tool
        tool = TavilySearchResults(
            api_key=settings.TAVILY_API_KEY,  
            max_results=3,
            search_depth="advanced"
        )

        # Execute the search
        try:
            search_results = tool.invoke({"query": query})
        except Exception as e:
            system_logger.error(f"Failed to execute search: {str(e)}")
            return {"query": query}

        # Check if results are empty
        if not search_results:
            system_logger.warning("No search results found.")
            return {"query": query}

        # Process and format the search results
        system_logger.info(f"Processing {len(search_results)} search results...")
        web_results = [
            f"Content: {result.get('content', 'No content')}\n"
            f"Link: {result.get('url', 'No link')}\n"
            for result in search_results
        ]

        web_document = Document(
            page_content= "\n\n".join(web_results),
            metadata = {
                "source": "tavily_search",
                "query": query,
                "num_results": len(web_results)
            }
        )

        documents.append(web_document)

        return {"documents": documents, "query": query}

    except Exception as e:
        system_logger.error(f"An unexpected error occurred: {str(e)}")
        return {"question": query}
    

def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    
    system_logger.info("Generating response....")
    query = state["query"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    
    context = "\n\n".join([doc.page_content for doc in documents])
    conversation_context = ""
    if conversation_history:
        conversation_context = "Previous conversation:\n" + "\n".join([
            f"{msg['role']} : {msg['content']}"
            for msg in conversation_history[-3:]
        ])
    
    try:
        prompt = PromptTemplate(
            template="""
            You are an intelligent assistant called MR_ZION_PLEASE_SELECT_ME tasked with providing helpful and accurate responses based on the provided documents.

            # Context Analysis
            Before responding, analyze whether the current query is:
            - Related to previous conversation (requiring context from past exchanges)
            - A new, independent question (requiring only the retrieved documents)

            {conversation_context}

            User Query: {query}

            Relevant Documents:
            {context}

            # Response Guidelines
            1. If the query continues or references the previous conversation, incorporate relevant context from prior exchanges.
            2. If the query is a new topic unrelated to previous exchanges, focus solely on the retrieved documents.
            3. Use the information from the provided documents to craft your response.
            4. If the documents contain conflicting information, acknowledge the different perspectives.
            5. If the documents don't fully address the query, clearly state the limitations of the available information.
            6. Maintain a professional and helpful tone throughout your response.
            7. Structure your response in a clear and coherent manner.
            8. Do not fabricate information that is not present in the documents.
            9. If you're unsure about specific details, acknowledge this uncertainty rather than making assumptions.
            10. When appropriate, cite the specific document or source from which you're drawing information.
            11. When asked What is your Name or Who are you refer to this " I'm an intelligent assistant called MR_ZION_SELECT_ME tasked with providing helpful and accurate responses based on provided documents"

            Your response:
            """, 
            input_variables=["conversation_context", "query", "context"]
        )

        response_chain = prompt | llm
        input_variables = {"context": context, "query": query, "conversation_context": conversation_context}
        result = response_chain.invoke(input_variables)
        
        
        new_conversation_history = conversation_history.copy()
        new_conversation_history.append({"role": "user", "content": query})
        new_conversation_history.append({"role": "assistant", "content": result})
        
        # Return the state with the correct keys
        return {
            "documents": documents,  # Return the original document objects
            "query": query,
            "llm_response": result,
            "conversation_history": new_conversation_history
        }
    
    except Exception as e:
        system_logger.error(f"Error in generating response: {str(e)}")
        return {
            "documents": documents,
            "query": query, 
            "llm_response": f"Sorry, I encountered an error while generating the response: {str(e)}", 
            "conversation_history": conversation_history
        }
    
workflow = StateGraph(GraphState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_document", grade_document)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search )
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_document")
workflow.add_conditional_edges(
    "grade_document",
    decide_to_generate,
    {
        "rewrite_query": "rewrite_query",
        "generate_response": "generate_response",
    },
)
workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("web_search", "generate_response")
workflow.add_edge("generate_response", END)

graph = workflow.compile(checkpointer=checkpointer)

def get_conversation_history_for_graph(chat_id):

    db = mongo_client["MR_ZION_PLEASE_SELECT_ME"]
    message_collection = db["MR_ZION_PLEASE_SELECT_ME_Chat_Messages"]

    messages = list(message_collection.find({"chat_id": chat_id}).sort("timmestamp", 1))

    conversation_history = []
    for message in messages:
        conversation_history.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    return conversation_history

# def MR_ZION_PLEASE_SELECT_ME_SPEAKING(chat_id, query) -> str:

#     conversation_history = get_conversation_history_for_graph(chat_id=chat_id)

#     try:
#         config = {
#             "configurable":{
#                 "thread_id": str(chat_id),
#                 "checkpoint_ns": f"chat_{chat_id}",
#                 "checkpoint_id": f"rag_graph_{chat_id}"
#             }
#         }

#         result = graph.invoke(
#             {"query": query, "conversation_history": conversation_history},
#             config=config
#         )
    
#     except Exception as e:

#         return f"Error processing query: {str(e)}"
    
#     return result["llm_response"]

def MR_ZION_PLEASE_SELECT_ME_SPEAKING(chat_id, query) -> str:
    system_logger.info(f"Starting processing for chat_id: {chat_id} with query: {query}")
    
    try:
        system_logger.info("Retrieving conversation history from database")
        conversation_history = get_conversation_history_for_graph(chat_id=chat_id)
        system_logger.info(f"Retrieved {len(conversation_history)} messages from conversation history")
        
        try:
            system_logger.info("Preparing graph configuration")
            config = {
                "configurable": {
                    "thread_id": str(chat_id),
                    "checkpoint_ns": f"chat_{chat_id}",
                    "checkpoint_id": f"rag_graph_{chat_id}"
                }
            }
            
            system_logger.info("Creating input state for graph")
            input_state = {"query": query, "conversation_history": conversation_history}
            
            system_logger.info("Invoking graph")
            result = graph.invoke(input_state, config=config)
            system_logger.info("Graph execution completed successfully")
            
            if "llm_response" not in result:
                system_logger.error(f"Missing llm_response in result. Keys returned: {result.keys()}")
                return "Error: Missing response in result"
            
            if hasattr(result["llm_response"], "content"):
                system_logger.info("result contains content")
                response = result["llm_response"].content
            else:
                response = str(result["llm_response"])
                
            system_logger.info(f"Successfully generated response with length: {len(response)}")
            return response
        
        except Exception as e:
            system_logger.error(f"Error during graph execution: {str(e)}")
            import traceback
            system_logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error processing query: {str(e)}"
    
    except Exception as e:
        system_logger.error(f"Error in conversation history retrieval: {str(e)}")
        import traceback
        system_logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error retrieving conversation context: {str(e)}"




 














