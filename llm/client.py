"""LLM Client for Google Gemini API with retry logic."""

import time
# from google import genai
import google.generativeai as genai

from config import GEMINI_API_KEY, DEFAULT_MODEL, MAX_RETRIES, RETRY_DELAY
from utils import get_logger

logger = get_logger(__name__)


# class LLMClient:
#     """Gemini API client"""
    
#     def __init__(self, model_name=DEFAULT_MODEL, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
#         self.model_name = model_name
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay
#         self.client = genai.Client(api_key=GEMINI_API_KEY)

#     def call(self, prompt: str, **kwargs) -> str:
#         """Send prompt to Gemini API"""
#         for attempt in range(self.max_retries + 1):
#             try:
#                 response = self.client.models.generate_content(
#                     contents=prompt, 
#                     model=self.model_name, 
#                     **kwargs
#                 )
#                 if response and response.text:
#                     return response.text.strip()
#                 else:
#                     raise Exception("Empty response received")
                
#             except Exception as e:
#                 if attempt == self.max_retries:
#                     raise Exception(f"API call failed after {self.max_retries + 1} attempts: {e}")
                
#                 time.sleep(self.retry_delay * (2 ** attempt))

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Load your API key from .env
# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = "AIzaSyBAnIP-_pUCSWuXVT228uFOUEixfFWSp8s"
if not GEMINI_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)


# class LLMClient:
#     """Wrapper for Google Generative AI (Gemini) model."""

#     def __init__(self, model_name="gemini-1.5-flash"):
#         self.model = genai.GenerativeModel(model_name)

#     def generate(self, prompt: str) -> str:
#         """Generate content using Gemini model."""
#         try:
#             response = self.model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             print(f"❌ Error generating response: {e}")
#             return "Error during LLM generation"


class LLMClient:
    """Wrapper for Google Generative AI (Gemini) model."""

    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        """Generate content using Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "Error during LLM generation"

    def call(self, prompt: str) -> str:
        """Alias for backward compatibility with old code."""
        return self.generate(prompt)