from dotenv import load_dotenv

from src.ui import playground

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    playground.launch(
        server_name="0.0.0.0", 
        share=False, 
        show_error=True, 
    )