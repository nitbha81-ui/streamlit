from google import genai
from config import *

class LLMClient:

    def __init__(self):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def call(self, prompt):

        response = self.client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt
        )

        return response.text
    