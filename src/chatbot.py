import cassio
from langchain.llms import Anyscale, OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from langchain.vectorstores import Cassandra

from loguru import logger
import os
from time import time

from src.astra import Astra
from src.state import State

class Chatbot:

    def __init__(self, state: State, db: Astra) -> None:
        self.state = state
        self.db = db

    def _create_llm(self, model: str, api_key: str):
        if model == "openai":
            logger.info("Using OpenAI LLM.")
            try:
                return OpenAI(openai_api_key=api_key)
            except Exception as e:
                raise ValueError("Invalid OpenAI API key.")
        elif model == "anyscale_llama2_70b_chat":
            logger.info("Using anyscale Llama-2-70b-chat-hf model.")
            os.environ["ANYSCALE_API_KEY"] = api_key           
            return Anyscale(model_name="meta-llama/Llama-2-70b-chat-hf")
        else:
            raise ValueError(f"Invalid LLM. Set on models tab.")

    
    def respond(
        self, 
        history, 
        prompt_input, 
        generate_model, 
        generate_api_key, 
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

        embed_model = self.db._create_embedding_model(embedding_model, embedding_api_key)
        vectorstore = self.db._create_vectorstore(embed_model)
        llm = self._create_llm(generate_model, generate_api_key)

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