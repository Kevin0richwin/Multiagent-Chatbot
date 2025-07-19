from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key

loader = PyPDFLoader("./data/cbt_knowledge.pdf")
documents = loader.load()


splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


db = Chroma.from_documents(chunks, embedding, persist_directory="./cbt_chroma_db")
db.persist()

print("✅ CBT ChromaDB has been built and saved using Google Gemini Embeddings.")
