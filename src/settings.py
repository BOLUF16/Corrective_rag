from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file= ".env", extra= "ignore", env_file_encoding="utf-8"
    )

    GROQ_API_KEY: str
    HF_KEY: str
   
    MONGO_URI: str

    TAVILY_API_KEY: str

    QDRANT_API_KEY: str 
    QDRANT_URL: str
    QDRANT_COLLECTION: str = "MR_ZION_PREETY_PLS"

settings = Settings()