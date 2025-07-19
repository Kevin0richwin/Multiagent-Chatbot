from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 🔐 Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC5HztCYhvsyf-y65cV1sTTO3lxcZ5ZZMQ"  # <<< Replace with your actual key

# 1. Load the CBT knowledge PDF
loader = PyPDFLoader("./data/cbt_knowledge.pdf")
documents = loader.load()

# 2. Split content into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Generate embeddings using Google Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. Store in ChromaDB
db = Chroma.from_documents(chunks, embedding, persist_directory="./cbt_chroma_db")
db.persist()

print("✅ CBT ChromaDB has been built and saved using Google Gemini Embeddings.")
