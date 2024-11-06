from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('.env')

client = OpenAI(
      api_key= os.getenv('OPENAI_API_KEY')
)

def detect_pun_word(text):
    print(text)
    prompt = f"""
    Analyze the following sentence and detect only one word which is the most probable word or phrase that might be a pun word. If pun word is detected pun word/words else say no pun detected in the below format:
    
    sentence: "{text}"

    Output format:
    <True/False>,<pun word/words>
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a pun detector"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    return response.choices[0].message.content

def detect_pun_word_match(text):
    prompt = f"""
    Analyze the following pun word/words then try to match the Homophones word for that pun word/words else say no pun detected in the format
     
    Pun Word/ words: "{text}"

    Output format:
    <True/False>,<word/words>
    """

    response = client.chat.completions.create(
        model="gpt-4o",  # Ensure you're using the GPT-4 model
        messages=[
            {"role": "system", "content": "You are a pun word Homophones detector"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    return response.choices[0].message.content

