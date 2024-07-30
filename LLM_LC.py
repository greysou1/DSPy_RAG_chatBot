import os
import boto3
import streamlit as st

import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_aws import BedrockLLM as Bedrock
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def get_chat_history(chat_history_list):
    history = "Chat History : \n"
    for chat_history_item in chat_history_list:
        history += f"{chat_history_item[0]} : {chat_history_item[1]} \n"
    
    return history

def load_vector_db(persist_directory="chromadb"):
    # Create embeddings instance
    embeddings = HuggingFaceEmbeddings()

    # Load the persisted Chroma vector database
    client = chromadb.HttpClient(host="127.0.0.1", settings=Settings(allow_reset=True))
    db = Chroma(client=client, embedding_function=embeddings)
    retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retv

# Step1: setup LLM Model
# =========== LLM MODEL ===========
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

#bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

modelID = "meta.llama2-13b-chat-v1"

llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_gen_len": 2000,"temperature":0.1}
)
# =====================================


# Memory
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key="answer")

# RAG
retriever = load_vector_db()

prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
            You are a chatbot on JetBlue website. Help customer by answering their questions.
            Example
            human: what is the baggage weight limit policy?
            ai: the baggage limit is 50 pounds in weight 
            """
    )

# Memory + RAG chain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, 
                                                memory=memory, 
                                                return_source_documents=True, 
                                                condense_question_prompt=prompt)


# StreamLit 
st.title('Welcome to the ChatBot')

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


for msg in st.session_state["chat_history"]:
    st.chat_message(msg[0]).write(msg[1])

if x := st.chat_input():
    st.session_state["chat_history"].append(['human', x])
    st.chat_message("human").write(x)

    # print(get_chat_history(st.session_state["chat_history"]))
    chat_history = get_chat_history(st.session_state["chat_history"])
    print(chat_history)
    response = qa.invoke({"question": x, "chat_history": chat_history})["answer"]

    # response = "text"
    st.session_state["chat_history"].append(['AI', response])
    st.chat_message("ai").write(response)
