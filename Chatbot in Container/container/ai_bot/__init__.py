import openai
import os
import sys

from openai import OpenAI

openai_apikey = ""
with open("/run/secrets/openai.txt", "r") as key_file:
    openai_apikey = key_file.read().strip()

llm1 = OpenAI(api_key=openai_apikey)

def Normal_LLM(query, llm1, chat_history):

    full_query = "\n".join([f"User: {q}\nAssistance: {a}" for q, a in chat_history])
    full_query += f"\nUser: {query}"

    messages = [
        {"role": "system", "content": "คุณเป็น AI assistant ที่เป็นมิตรและช่วยเหลือ"},
        {"role": "user", "content": full_query}
    ]

    chain1 = llm1.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )

    result = chain1.choices[0].message.content
    # result = chain2({"question": query, "chat_history": chat_history})
    return result



def make_chat(incoming_data:str, chat_history:list[tuple[str, str]])->str:

    result = Normal_LLM(incoming_data, llm1, chat_history)
    print("Chatbot:", result)

    chat_history.append((incoming_data, result))
    return result
    
    