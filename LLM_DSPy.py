import os
import dspy
import uuid
import chromadb
import streamlit as st


from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from dotenv import load_dotenv
load_dotenv()

class Chatbot(dspy.Signature):
    # """Answer questions with short factoid answers."""
    """Act as a ChatBot on JetBlue airlines website, have a friendly and helpful conversation with customers, answer queries based on facts and context."""

    context = dspy.InputField(desc="may contain relevant facts")
    chat_history = dspy.InputField(desc="contains the history of the chat so far")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="between 10-50 words. Detailed and answer.")

class RAG_chatbot(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(Chatbot)
    
    def get_chat_history(self, st_chat_history):
        chat_history = "Chat History : \n"
        for chat_history_item in st_chat_history:
            chat_history += f"{chat_history_item[0]} : {chat_history_item[1]} \n"
        
        return chat_history
    
    def forward(self, question):
        chat_history = self.get_chat_history(st.session_state["chat_history"] )
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question, chat_history=chat_history)

        return dspy.Prediction(context=context, answer=prediction.answer)

def get_chat_history(chat_history_list):
    history = "Chat History : \n"
    for chat_history_item in chat_history_list:
        history += f"{chat_history_item[0]} : {chat_history_item[1]} \n"
    
    return history

def load_vector_db(persist_directory="chromadb2"):
    # Create embeddings instance
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002"
    )
    # embedding_function = SentenceTransformerEmbeddingFunction()

    retriever_model = ChromadbRM(
        'JetBlueHelp',
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        k=8)
    
    return retriever_model

def load_llm_model(use_model='openai'):
    if use_model == 'cohere':
        llm_model = dspy.Cohere(model='command-xlarge-nightly', api_key=os.getenv("COHERE_API_KEY"))
    elif use_model == 'phi':
        llm_model = dspy.OllamaLocal(model='phi')
    elif use_model == 'openai':
        llm_model =  dspy.OpenAI(model='gpt-3.5-turbo-1106', api_key=os.getenv("OPENAI_API_KEY"))
    else: # use phi local model
        llm_model = dspy.OllamaLocal(model='llama3')
    
    return llm_model

dspy.settings.configure(lm=load_llm_model(), rm=load_vector_db()) # configure dspy


if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

chatbot = RAG_chatbot()
chatbot.load("compiled_models/chatbot_RAG.json")

st.title('jetBlue Assistant')

for msg in st.session_state["chat_history"]:
    st.chat_message(msg[0]).write(msg[1])

if x := st.chat_input():
    st.session_state["chat_history"].append(['human', x])
    st.chat_message("human").write(x)
    
    response = chatbot(question=x).answer
    
    print(f"Human: {x}")
    print(f"AI: {response}\n")

    st.session_state["chat_history"].append(['AI', response])
    st.chat_message("ai").write(response)
