import os
import dspy
import streamlit as st

from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from dotenv import load_dotenv
load_dotenv()


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

    retriever_model = ChromadbRM(
        'JetBlueHelp',
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        k=5
    )

    return retriever_model

# Step1: setup LLM Model
# =========== LLM MODEL ===========
use_model = 'llama3'
if use_model == 'cohere':
    llm_model = dspy.Cohere(model='command-xlarge-nightly', api_key=os.getenv("COHERE_API_KEY"))
if use_model == 'phi':
    llm_model = dspy.OllamaLocal(model='phi')
else: # use phi local model
    llm_model = dspy.OllamaLocal(model='llama3')

print(f"Using model: {use_model}")
# =====================================

dspy.settings.configure(lm=llm_model, rm=load_vector_db()) # configure dspy

class Chatbot(dspy.Signature):
    # """Answer questions with short factoid answers."""
    """Act as a ChatBot on JetBlue airlines website, answer queries by customers based on facts and context."""

    context = dspy.InputField(desc="may contain relevant facts")
    chat_history = dspy.InputField(desc="contains the history of the chat so far")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="No more than 20 words. may have metrics")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(Chatbot)
    
    def get_chat_history(self, st_chat_history):
        chat_history = "Chat History : \n"
        for chat_history_item in st_chat_history:
            chat_history += f"{chat_history_item[0]} : {chat_history_item[1]} \n"
        
        return chat_history
    
    def forward(self, question, st_chat_history):
        context = self.retrieve(question).passages
        chat_history = self.get_chat_history(st_chat_history)
        prediction = self.generate_answer(context=context, question=question, chat_history=chat_history)

        return dspy.Prediction(context=context, answer=prediction.answer)


generate_answer_with_chain_of_thought = dspy.ChainOfThought(Chatbot)

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
    retv = dspy.Retrieve(k=3)
    context = retv(x + " ".join(chat_history)).passages
    # print(context)
    response = generate_answer_with_chain_of_thought(question=x, 
                                                     chat_history=chat_history, 
                                                     context=context).answer

    print(response)
    # response = "text"
    st.session_state["chat_history"].append(['AI', response])
    st.chat_message("ai").write(response)
