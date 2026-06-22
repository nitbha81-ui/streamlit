import os
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2