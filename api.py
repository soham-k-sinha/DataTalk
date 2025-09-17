from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain how AI works in a few words",
)

print(response.text)