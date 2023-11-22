class State:
    # This class preserves state across tabs in the gradio app

    def __init__(self):
        self.chunking_method = None
        self.chunks = None
        self.document_text = None
        self.llm = None

    def set_chunking_method(self, method):
        self.chunking_method = method

    def get_chunking_method(self):
        return self.chunking_method

    def set_chunks(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def get_chunks(self) -> list[str]:
        return self.chunks

    def set_document_text(self, text):
        self.document_text = text

    def get_document_text(self):
        return self.document_text
    
    def set_llm(self, llm):
        self.llm = llm

    def get_llm(self):
        return self.llm