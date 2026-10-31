import os
import time
from time import sleep
import requests
from dotenv import load_dotenv

load_dotenv()

payload = {
    "models": [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
    ],
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ]
}

headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

for attempt in range(5):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    data = response.json()

    if response.status_code == 200:
        print("Success:")
        print(data["choices"][0]["message"]["content"])
        break
    elif response.status_code == 429:
        wait_time = 2 ** attempt
        print(f"Shared pool busy (429). Retrying in {wait_time}s...")
        time.sleep(wait_time)
    else:
        print("Error:", data)
        break
else:
    print("Failed after maximum retries. Last response:", data)
