import cassio
from langchain.vectorstores import Cassandra
from langchain.schema.embeddings import Embeddings
from langchain.embeddings import OpenAIEmbeddings

import gradio as gr

from loguru import logger
import os
import time

# TODO: this class isn't really a proper astra wrapper, instead its an embedding generator
# with hard coded astra
class Astra:
    def __init__(self, state) -> None:
        self.keyspace = os.environ["ASTRA_KEYSPACE"]
        self.database_id = os.environ["ASTRA_DATABASE_ID"]
        self.table = os.environ["ASTRA_TABLE"]
        self.token = os.environ["ASTRA_TOKEN"]
        self.session = self._connect_to_astra()
        self.vectorstore = None
        self.state = state
    
    def _create_vectorstore(self, embed_model):
        # user can change embedding model in UI, so init this at runtime of task
        return Cassandra(
            embedding=embed_model,
            session=self.session,
            keyspace=self.keyspace,
            table_name=self.table,
        )

    def _connect_to_astra(self):
        cassio.init(token=self.token, database_id=self.database_id)
        logger.info(f"Connected to Astra DB: {self.database_id}")
        return cassio.config.resolve_session()    

    def _create_embedding_model(self, embedding_model: str, embedding_api_key: str) -> Embeddings:
        if embedding_model == "openai":
            try:     
                return OpenAIEmbeddings(openai_api_key=embedding_api_key)
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")
        else:
            raise ValueError(f"Invalid embedding model. Set on models tab.")

    def get_vectorstore(self):
        return self.vectorstore


    def store_chunks(
        self, 
        embedding_model: str,
        embedding_api_key:str,
    ) -> None:
               
        logger.info("In store chunks.")
        gr.Info("Adding chunks to Astra DB.")

        embed_model = self._create_embedding_model(embedding_model, embedding_api_key)
        self.vectorstore = self._create_vectorstore(embed_model)

        logger.info("Adding document to Astra DB.")

        chunks = self.state.get_chunks()
        start = time.time()
        self.vectorstore.add_texts(texts=chunks)
        end = time.time() 

        gr.Info(f"Stored {len(chunks)} embeddings. Avg time per embedding {(end-start)/len(chunks)} seconds.")
        logger.info(f"Added {len(chunks)} embeddings to table: {self.table}.")
        logger.info(f"Average time per embedding {(end-start)/len(chunks)} seconds.")

    def delete_table(self) -> None:
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.table};") 

        gr.Info("Deleted embeddings table.")
        logger.info(f"Deleted table: {self.table}.")  