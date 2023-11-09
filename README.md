# rag-playground

Copy `env.example` to `.env`.

Set the variables for `ASTRA_TOKEN` and `ASTRA_DATABASE_ID`.

Run with docker:

```
docker compose up
```

Run locally

1. Set the env vars in the `.env` file.
2. Run the commands below.

```
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in Chrome.

## API Keys for services

- [https://astra.datastax.com](https://astra.datastax.com) for vector database credentials
- [https://platform.openai.com](https://platform.openai.com) for OpenAI LLMs and embeddings
- [https://dashboard.cohere.com/](https://dashboard.cohere.com/) for Cohere embedding models and reranker
- [https://app.endpoints.anyscale.com/](https://app.endpoints.anyscale.com/) for OSS LLMs Llama2 70B and Mistral 7B