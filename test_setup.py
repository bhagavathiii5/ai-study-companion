import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No API key found — check your .env file")
else:
    print("API key loaded")

# Create the client
client = genai.Client(api_key=api_key)

# Test a basic call
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one short sentence."
)

print("Gemini responded:", response.text)