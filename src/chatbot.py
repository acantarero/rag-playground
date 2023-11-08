from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import cassio
from langchain.vectorstores import Cassandra

from loguru import logger
from time import time

from src.state import State

class Chatbot:

    def __init__(self, state: State) -> None:
        self.state = state
        self.session = self._connect_to_astra()
        self.keyspace = "rag_playground" # TODO: DRY

    # TODO: DRY
    def _connect_to_astra(self):
        cassio.init(token=self.state.token, database_id=self.state.database_id)
        logger.info(f"Connected to Astra DB: {self.state.database_id}")
        return cassio.config.resolve_session()    
    
    # TODO: DRY
    def _create_vectorstore(self, embed_model, table_name):
        return Cassandra(
            embedding=embed_model,
            session=self.session,
            keyspace=self.keyspace,
            table_name=table_name,
        )
    
    def respond(
        self, 
        history, 
        prompt_input, 
        model, 
        api_key, 
        embedding_model, 
        embedding_api_key,
        table_name,
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
        
        # TODO: this code is duplicated in astra.py
        if embedding_model == "openai":
            try:     
                embed_model = OpenAIEmbeddings(openai_api_key=embedding_api_key)
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")
        else:
            raise ValueError(f"Invalid embedding model. Set on models tab.")
        

        # TODO: DRY
        vectorstore = self._create_vectorstore(embed_model, table_name)

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