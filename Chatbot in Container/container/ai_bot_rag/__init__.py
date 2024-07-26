import openai
import os
import sys

# Import classes from modules

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from pydantic import SecretStr



import json
from typing import cast, overload
from langchain_core.documents import Document #type: ignore

openai_apikey = ""
with open("/run/secrets/openai.txt", "r") as key_file:
    openai_apikey = key_file.read().strip()

@overload
def extract_data(path: str, main_key: str)->list[Document]:...

@overload
def extract_data(path: str)->list[Document]:...

def extract_data(path: str, main_key:str="items") -> list[Document]:
    documents: list[Document] = []
    with open(path, "r", encoding='utf8') as json_file:
        data:list[dict[str, str]] = json.load(json_file)[main_key]
        
        for each_data in data:
            page_content: str = ""
            for k in each_data:
                page_content = page_content + f"{k} {each_data[k]}"
            current_document = Document(
                page_content=page_content,
            )
            documents.append(current_document)
    return documents

json_file_path="/container/test_data.json"
data = extract_data(json_file_path)
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

connection = "postgresql+psycopg://langchain:langchain@pgvector:5432/langchain"  # Uses psycopg3!
collection_name = "my_docs"
embeddings = OpenAIEmbeddings(api_key=SecretStr(value=openai_apikey))
vectorstore = PGVector(embeddings=embeddings, collection_name=collection_name, connection=connection, use_jsonb=True)

vectorstore.add_documents(data)

custom_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Answer the question based ONLY on the following context, answer : adding watermark
    If the question asked by the user does not match the context, answer : I don't known.


    {context}
    Original question: {question}""",
)


llm1 = ChatOpenAI(model="gpt-3.5-turbo-16k",api_key=SecretStr(value=openai_apikey))

# Create the chain
chain1 = ConversationalRetrievalChain.from_llm(
    llm=llm1,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": custom_template}
)

def make_chat_with_rag(incoming_data: str, chat_history: list[BaseMessage])->str:
    
    result = chain1.invoke({"question": incoming_data, "chat_history": chat_history})  
    print("Chatbot:", result['answer'])

    chat_history.append(HumanMessage(content=incoming_data))
    chat_history.append(AIMessage(content=result['answer']))

    return cast(str, chat_history[-1].content)
