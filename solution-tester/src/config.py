import os


class Settings:
    """Application settings."""

    def __init__(self):
        # Server configuration
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8008"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # Timeout settings
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))


settings = Settings()
