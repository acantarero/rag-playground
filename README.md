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