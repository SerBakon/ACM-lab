import os
import requests
from dotenv import load_dotenv
load_dotenv()

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
    },
    json={
        "model": "google/gemma-4-31b-it:free",
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    }
)

print(response.json())