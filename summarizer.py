import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_transcript(transcript: str) -> str:
    """
    Takes a raw transcript and returns a structured summary
    with a short overview and a bulleted action items list.
    """
    prompt = f"""You are an assistant that summarizes lecture or meeting transcripts.

Given the transcript below, produce:
1. A concise summary (3-5 sentences) of the key points discussed
2. A bulleted list of clear action items, if any were mentioned (if none, say "No action items identified")

Transcript:
{transcript}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# Quick test — only runs when you execute this file directly
if __name__ == "__main__":
    sample_transcript = """
    Professor: Today we covered binary search trees. Remember, insertion is O(log n)
    on average but can degrade to O(n) if the tree becomes unbalanced. For next week,
    please read chapter 7 on AVL trees and submit the practice problems by Friday.
    Also, the mid-term will cover everything up to today's lecture.
    """

    result = summarize_transcript(sample_transcript)
    print(result)