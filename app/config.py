"""Settings loaded from environment / .env (never hardcode secrets)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
NOT_FOUND = "Information not found in the current guidelines."


class ServiceError(Exception):
    """A provider/config failure whose message is safe to show to users (no secrets)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    pdf_path: str = "data/Compendium_Contaminants_Regulations.pdf"
    chroma_path: str = "storage/chroma"
    sqlite_path: str = "storage/fssai.db"

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path


settings = Settings()
