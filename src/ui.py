import gradio as gr
from pypdf import PdfReader
import trafilatura

from src.astra import Astra
from src.chatbot import Chatbot
from src.chunker import Chunker
from src.state import State

app_state = State()
db = Astra(app_state)
cb = Chatbot(app_state, db)
chunker = Chunker(app_state)


def user_chatbot(user_message, history):
    return "", history + [[user_message, None]]

def upload_pdf(pdf):
    reader = PdfReader(pdf.name)
    text = ""
    for page in reader.pages:
       text += page.extract_text()
    app_state.set_document_text(text)
    return text

def scrape_webpage(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    app_state.set_document_text(text)
    return text

def on_llm_choice_change(llm_choice):
    if llm_choice in [
        "openai_gpt_35_turbo", 
        "openai_gpt_4",        
        "anyscale_llama2_70b_chat",
        "anyscale_mistral_7b_instruct"]:
        return {
            llm_params: gr.Row(visible=True),
        }
    else:
        return {
            llm_params: gr.Row(visible=False),
        }
    
def on_embedding_choice_change(embedding_choice):
    if embedding_choice == "openai":
        return {
            embedding_openai_params: gr.Row(visible=True),
        }
    else:
        return {
            embedding_openai_params: gr.Row(visible=False),
        }
    
def on_chunking_choice_change(chunking_choice):
    app_state.set_chunking_method(chunking_choice)

with gr.Blocks() as playground:
    gr.Markdown("""# RAG Playground""")

    with gr.Tab("Overview"):
        gr.Markdown("""This is a playground for experimeting with RAG. 
            On different tabs you'll be able to configure and explore different parts of the RAG process.

            * **Chat** - Talk to the documents that you have uploaded.
            * **Documents** - Upload a PDF or scrape the contents of a webpage to use as a document.
            * **Store** - Store the document from the **Documents** tab for retrieval. Experiment with chunking and embedding models.
            * **Retrieve** - Use an Approximate Nearest Neighbor (ANN) search to retrieve documents for a query.
                    
            You can upload a PDF or scrape a webpage to load a document. Then, you can ask questions about the document and see the answers.
        """)

    with gr.Tab("Chat"):
        gr.Markdown("""Customize the prompt to chat with your documents. 
            
            Include:
                    
            * `{context}` to add documents
            * `{question}` to add the user query submitted to the chatbot.
                    
            Example: 
                
                    Consider the following context when you answer the user's question.
                    {context}
                    {question}
        """)
        prompt = gr.Textbox(label="Prompt", lines=5, value="{question}")
        chatbox = gr.Chatbot()
        textbox = gr.Textbox(show_label=False, placeholder="Ask questions to your documents.")
        clear = gr.ClearButton([textbox, chatbox])

        clear.click(lambda: None, None, chatbox, queue=False)

    with gr.Tab("Documents"):
        gr.Markdown("### Select content for RAG.")
        
        upload_btn = gr.UploadButton(
            label="Upload a PDF", 
            file_types=[".pdf"],
            file_count="single",
        )
        gr.Markdown("### or")
        url_tb = gr.Textbox(label="Load content from the web.", placeholder="Enter URL to load a website.")
        scrape_btn = gr.Button(value="Scrape")

        doc_textbox = gr.Textbox(label="Document text", lines=20)

        upload_btn.upload(upload_pdf, upload_btn, doc_textbox)
        scrape_btn.click(scrape_webpage, url_tb, doc_textbox)

    with gr.Tab("Store"):
        chunking_choice = gr.Dropdown(
            label="Chunking method",
            choices=[("Recursive Character Splitter", "recursive_character")],
            type="value",
        )

        chunk_size = gr.Number(label="Chunk Size", value=100)
        overlap = gr.Number(label="Overlap", value=20)
        chunk_btn = gr.Button(value="Chunk")
        show_chunks = gr.Textbox(label="Chunks", lines=20)

        gr.Markdown("""## Add chunks to Astra
        Generate embeddings for each chunk and store to a table in Astra.

        *Note that if you change the embedding model you need to create a new table
        or delete the existing table.*            
        """)
        add_chunks_btn = gr.Button(value="Add Chunks")

        with gr.Accordion("Danger Zone", open=False):
           delete_btn = gr.Button(value="Delete Table")

        # store tab actions
        chunking_choice.change(on_chunking_choice_change, chunking_choice)
        chunk_btn.click(chunker.chunk, [chunk_size, overlap], show_chunks)
        delete_btn.click(db.delete_table)

       
    with gr.Tab("Retrieve"):
        gr.Markdown("Coming soon.")

    with gr.Tab("Models"):
        gr.Markdown("Select the language models to use.")
        llm_choice = gr.Dropdown(
            label="Generation model (LLM)", 
            choices=[
                ("OpenAI - GPT 3.5 Turbo", "openai_gpt_35_turbo"),
                ("OpenAI - GPT 4", "openai_gpt_4"),
                ("Anyscale - Llama2 70B Chat", "anyscale_llama2_70b_chat"),
                ("Anyscale - Mistral 7B Instruct", "anyscale_mistral_7b_instruct"),
            ], 
            type="value",
        )

        with gr.Row(visible=False) as llm_params:
            llm_api_key = gr.Textbox(label=f"API Key", lines=1)

        embedding_choice = gr.Dropdown(
            label="Embedding model", 
            choices=[("OpenAI - text-embedding-ada-002", "openai")], 
            type="value",
        )

        with gr.Row(visible=False) as embedding_openai_params:
            embedding_api_key = gr.Textbox(label=f"OpenAI API Key", lines=1)


    # cross tab actions
    textbox.submit(user_chatbot, [textbox, chatbox], [textbox, chatbox], queue=False).then(
        cb.respond, 
        [chatbox, prompt, llm_choice, llm_api_key, embedding_choice, embedding_api_key], 
        chatbox
    )
    
    llm_choice.change(on_llm_choice_change, llm_choice, [llm_params])
    embedding_choice.change(on_embedding_choice_change, embedding_choice, [embedding_openai_params])

    add_chunks_btn.click(db.store_chunks, [embedding_choice, embedding_api_key])
