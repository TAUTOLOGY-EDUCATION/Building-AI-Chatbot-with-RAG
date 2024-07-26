import sys
from typing import cast

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from pydantic import SecretStr


local_path = "/container/drugs.pdf"

if local_path:
    loader = PyMuPDFLoader(file_path=local_path)
    data = loader.load_and_split()
else:
    print("Upload a PDF file")
    sys.exit()

text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

openai_apikey = ""
with open("/run/secrets/openai.txt", "r") as key_file:
    openai_apikey = key_file.read().strip()
embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_apikey))
connection = "postgresql+psycopg://langchain:langchain@pgvector:5432/langchain"  
collection_name = "drug"

vectorstore = PGVector(embeddings=embeddings, collection_name=collection_name, connection=connection, use_jsonb=True)

vectorstore.add_documents(texts)

custom_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI thai language model assistant.
    You are an expert at answering questions about medicine.
    Answer the question based ONLY on the following context.


    {context}


    Original question: {question}""",
)
anthropic_apikey = ""
with open("/run/secrets/claude.txt", "r") as key_file:
    anthropic_apikey = key_file.read().strip()

llm1 = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=SecretStr(anthropic_apikey),timeout=None)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm1,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": custom_template}
)


def make_facebook_chat(incoming_data: str, chat_history: list[BaseMessage])->str:
    
    result = chain.invoke({"question": incoming_data, "chat_history": chat_history})  
    print("Chatbot:", result['answer'])

    chat_history.append(HumanMessage(content=incoming_data))
    chat_history.append(AIMessage(content=result['answer']))

    return cast(str, chat_history[-1].content)