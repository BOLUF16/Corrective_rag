from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.config.appconfig import *
from typing import List, Tuple
from src.utils.helper import web_search
from src.exception.operationhandler import system_logger


class RetrievalEvaluator(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of the document to the query. the score should be between 0 and 1.")
class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The document to extract key information from.")
class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")



class CorrectiveRAG:
    """
    A class for Corrective Retrieval-Augmented Generation (CRAG) that integrates document retrieval,
    evaluation, knowledge refinement, and web search to generate accurate and context-aware responses.
    """

    def __init__(self, model: str, temperature: float):
        """
        Initialize the CorrectiveRAG class.

        Args:
            model (str): The name of the Groq model to use.
            temperature (float): The temperature for the Groq model.
            groq_key (str): The API key for the Groq service.
        """
        self.model = model
        self.temperature = temperature
        self.groq_key = groq_key
        self.llm = ChatGroq(model=self.model, temperature=self.temperature, api_key=self.groq_key)
        system_logger.info("CorrectiveRAG initialized with model: %s and temperature: %s", self.model, self.temperature)

    def grade_document(self, query: str, document: str) -> float:
        """
        Grade the relevance of a document to a query.

        Args:
            query (str): The query to evaluate against.
            document (str): The document to evaluate.

        Returns:
            float: The relevance score between 0 and 1.
        """
        system_logger.info("Grading document for query: %s", query)
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is the following document to the query? Query: {query}\nDocument: {document}\nRelevance score:"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluator)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        system_logger.info("Document relevance score: %s", result)
        return result

    def knowledge_refinement(self, document: str) -> List[str]:
        """
        Extract key information from a document.

        Args:
            document (str): The document to extract key points from.

        Returns:
            List[str]: A list of key points extracted from the document.
        """
        system_logger.info("Refining knowledge from document")
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract the key information from the following document in bullet points:\n{document}\nKey points:"
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        result = chain.invoke(input_variables).key_points
        key_points = [point.strip() for point in result.split('\n') if point.strip()]
        system_logger.info("Extracted %s key points", len(key_points))
        return key_points

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to make it more suitable for a web search.

        Args:
            query (str): The query to rewrite.

        Returns:
            str: The rewritten query.
        """
        system_logger.info("Rewriting query: %s", query)
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        rewritten_query = chain.invoke(input_variables).query.strip()
        system_logger.info("Rewritten query: %s", rewritten_query)
        return rewritten_query

    def retrieve_documents(self, query: str, vector_store, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant documents from a vector store.

        Args:
            query (str): The query to retrieve documents for.
            vector_store: The vector store to retrieve documents from.

        Returns:
            List[str]: A list of retrieved document contents.
        """
        system_logger.info("Retrieving documents for query: %s", query)
        documents = vector_store.similarity_search(query, top_k)
        document_contents = [doc.page_content for doc in documents]
        system_logger.info("Retrieved %s documents", len(document_contents))
        return document_contents

    def evaluate_documents(self, query: str, documents: List[str]) -> List[float]:
        """
        Evaluate the relevance of a list of documents to a query.

        Args:
            query (str): The query to evaluate against.
            documents (List[str]): The list of documents to evaluate.

        Returns:
            List[float]: A list of relevance scores for the documents.
        """
        system_logger.info("Evaluating %s documents for query: %s", len(documents), query)
        return [self.grade_document(query, doc) for doc in documents]

    def perform_web_search(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Perform a web search and refine the results into key points.

        Args:
            query (str): The query to search for.

        Returns:
            Tuple[List[str], List[Tuple[str, str]]]: A tuple containing:
                - A list of key points extracted from the web search results.
                - A list of tuples containing the content and link of each web result.
        """
        system_logger.info("Performing web search for query: %s", query)
        rewritten_query = self.rewrite_query(query)
        web_results = web_search(rewritten_query)

        # Process web_results to extract content and link
        sources = []
        for result in web_results:
            lines = result.split("\n")
            content = lines[0].replace("Content: ", "").strip() if len(lines) > 0 else "No content"
            link = lines[1].replace("Link: ", "").strip() if len(lines) > 1 else "No link"
            sources.append((content, link))

        system_logger.info("Web search completed with %s results", len(sources))

        # Refine the web results into key points
        web_knowledge = self.knowledge_refinement("\n".join([content for content, _ in sources]))
        return web_knowledge, sources

    def generate_response(self, query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
        """
        Generate a response based on the query, knowledge, and sources.

        Args:
            query (str): The query to answer.
            knowledge (str): The knowledge to use for generating the response.
            sources (List[Tuple[str, str]]): A list of tuples containing the content and link of each source.

        Returns:
            str: The generated response.
        """
        system_logger.info("Generating response for query: %s", query)
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="Based on the following knowledge, answer the query. Include the web results with their links (if available) at the end of your answer:\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": "\n".join([f"{content}: {link}" if link else content for content, link in sources])
        }
        response_chain = response_prompt | self.llm
        response = response_chain.invoke(input_variables).content
        system_logger.info("Response generated")
        return response

    def crag_process(self, query: str, vector_store) -> str:
        """
        Perform the full CRAG process: retrieve, evaluate, refine, and generate a response.

        Args:
            query (str): The query to process.
            vector_store: The vector store to retrieve documents from.

        Returns:
            str: The final response to the query.
        """
        system_logger.info("Starting CRAG process for query: %s", query)
        retrieved_docs = self.retrieve_documents(query, vector_store)
        eval_scores = self.evaluate_documents(query, retrieved_docs)
        max_score = max(eval_scores) if eval_scores else 0
        sources = []

        if max_score > 0.7:
            system_logger.info("Action: Correct - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append(("Retrieved document", ""))
        elif max_score < 0.3:
            system_logger.info("Action: Incorrect - Performing web search")
            final_knowledge, sources = self.perform_web_search(query)
        else:
            system_logger.info("Action: Ambiguous - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved document", "")] + web_sources

        system_logger.info("Final knowledge: %s", final_knowledge)
        system_logger.info("Sources: %s", sources)

        response = self.generate_response(query, final_knowledge, sources)
        system_logger.info("CRAG process completed")
        return response

        

    


    

    


    

        
 

    

