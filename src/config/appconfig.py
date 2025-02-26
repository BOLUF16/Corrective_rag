from dotenv import load_dotenv
import os
load_dotenv()


groq_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HF_KEY")
qdrant_key = os.getenv("QDRANT_API")
tavily_key = os.getenv("TAVILY_API_KEY")


