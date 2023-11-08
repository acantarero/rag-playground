import cassio
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from langchain.vectorstores import Cassandra

from loguru import logger
from time import time

from src.astra import Astra
from src.state import State

class Chatbot:

    def __init__(self, state: State, db: Astra) -> None:
        self.state = state
        self.db = db
    
    def respond(
        self, 
        history, 
        prompt_input, 
        model, 
        api_key, 
        embedding_model, 
        embedding_api_key
    ):

        input_variables = []
        if "{question}" in prompt_input:
            input_variables.append("question")

        if "{context}" in prompt_input:
            input_variables.append("context")   
        
        prompt = PromptTemplate(
            template=prompt_input,
            input_variables=input_variables,
        )

        if model == "openai":
            try:
                llm = OpenAI(openai_api_key=api_key)
            except Exception as e:
                raise ValueError("Invalid OpenAI API key.")
        else:
            raise ValueError(f"Invalid LLM. Set on models tab.")

        embed_model = self.db._create_embedding_model(embedding_model, embedding_api_key)
        vectorstore = self.db._create_vectorstore(embed_model)

        question = history[-1][0]

        rag_chain = {
            "context": vectorstore.as_retriever(),
            "question": RunnablePassthrough()
        } | prompt | llm

        start = time()
        history[-1][1] = rag_chain.invoke(question)
        end = time()
 
        logger.info(f"RAG chain took {end-start} seconds.")

        return history