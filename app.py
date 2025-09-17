from fastapi import FastAPI
from dotenv import load_dotenv
import os
from google import genai

app = FastAPI()


@app.get("/")
def root():
    return {}


@app.post("/call_api/")
async def api_call(prompt: str):
    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"{prompt}",
    )

    return {"output": f"{response.text}"}
