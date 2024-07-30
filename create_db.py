import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

def create_vector_db(pdf_folder, persist_directory="chromadb2"):
    # Load all PDF files from the specified folder
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    all_documents = []
    
    # Load documents from each PDF file
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        all_documents.extend(documents)
    
    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    
    
    # Create and persist the vector database
    vector_db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    vector_db.add_documents(texts)
    
    # Persist the database
    vector_db.persist()
    
    return vector_db

# Example usage
pdf_folder_path = "/Users/prudvikamtam/Documents/Projects/LLM/JetBlue_LLM/documents/"
vector_db = create_vector_db(pdf_folder_path)


# embeddings = OpenAIEmbeddingFunction(
#         api_key=os.environ.get('OPENAI_API_KEY'),
#         model_name="text-embedding-ada-002"
#         )