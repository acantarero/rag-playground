from langchain.chat_models import ChatOpenAI
from langchain.llms import Anyscale
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from langchain.vectorstores import Cassandra

import gradio as gr
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
        if model == "openai_gpt_35_turbo":
            logger.info("Using OpenAI gpt-3.5-turbo LLM.")
            try:
                return ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")
            
        elif model == "openai_gpt_4":
            logger.info("Using OpenAI gpt-4 LLM.")
            try:
                return ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")
            
        elif model == "anyscale_llama2_70b_chat":
            logger.info("Using anyscale Llama-2-70b-chat-hf model.")
            os.environ["ANYSCALE_API_KEY"] = api_key           
            return Anyscale(model_name="meta-llama/Llama-2-70b-chat-hf")
        
        elif model == "anyscale_mistral_7b_instruct":
            logger.info("Using anyscale mistralai/Mistral-7B-Instruct-v0.1 model.")
            os.environ["ANYSCALE_API_KEY"] = api_key           
            return Anyscale(model_name="mistralai/Mistral-7B-Instruct-v0.1")
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
        result = rag_chain.invoke(question)
        if "openai" in generate_model:
            result = result.content

        history[-1][1] = result
        end = time()
 
        logger.info(f"RAG chain took {end-start} seconds.")
        gr.Info(f"RAG chain took {(end-start):.2f} seconds.")

        return history