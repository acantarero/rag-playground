import cassio
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from loguru import logger
import os
import time

# TODO: this class isn't really a proper astra wrapper, instead its an embedding generator
# with hard coded astra
class Astra:
    def __init__(self, state) -> None:
        # passing through chunks is an ugly hack to get a reference
        # to the state object here

        self.keyspace = "rag_playground" # hard-code
        self.token = os.environ["ASTRA_TOKEN"]
        self.database_id = os.environ["ASTRA_DATABASE_ID"]
        self.state = state
        self.session = self._connect_to_astra()
        self.vectorstore = None
    
    def _create_vectorstore(self, embed_model, table_name):
        return Cassandra(
            embedding=embed_model,
            session=self.session,
            keyspace=self.keyspace,
            table_name=table_name,
        )

    def _connect_to_astra(self):
        cassio.init(token=self.state.token, database_id=self.state.database_id)
        logger.info(f"Connected to Astra DB: {self.state.database_id}")
        return cassio.config.resolve_session()    

    def get_vectorstore(self):
        return self.vectorstore

    def store_chunks(
        self, 
        table_name: str, 
        embedding_model: str,
        embedding_api_key:str,
    ) -> None:
        
        if embedding_model == "openai":
            try:     
                embed_model = OpenAIEmbeddings(openai_api_key=embedding_api_key)
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")
        else:
            raise ValueError(f"Invalid embedding model. Set on models tab.")

        logger.info("Adding document to Astra DB.")

        if self.vectorstore is None:
            self.vectorstore = self._create_vectorstore(embed_model, table_name)

        chunks = self.state.get_chunks()
        start = time.time()
        self.vectorstore.add_texts(texts=chunks)
        end = time.time() 
        logger.info(f"Added {len(chunks)} embeddings to table: {table_name}.")
        logger.info(f"Average time per embedding {(end-start)/len(chunks)} seconds.")

    def delete_table(self, table_name: str) -> None:
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{table_name};")  
        logger.info(f"Deleted table: {table_name}.")  