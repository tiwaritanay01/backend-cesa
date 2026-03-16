import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==========================================
# 1. INITIALIZE MODELS
# ==========================================
# Use your preferred basic LLM (e.g., llama3.1, qwen2.5)
llm = ChatOllama(model="llama3.1", temperature=0)

# Use a fast, local embedding model to vectorize your documents
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ==========================================
# 2. INGEST & CHUNK CONTEXT DATA
# ==========================================
# Ensure this folder exists and contains your context files (e.g., PDFs, TXT, MD)
data_directory = "./profile_data"
os.makedirs(data_directory, exist_ok=True)

print(f"Loading documents from {data_directory}...")
# Support for PDF, Text, and Markdown files
pdf_loader = PyPDFDirectoryLoader(data_directory)
txt_loader = DirectoryLoader(data_directory, glob="**/*.txt", loader_cls=TextLoader)
md_loader = DirectoryLoader(data_directory, glob="**/*.md", loader_cls=TextLoader)

docs = pdf_loader.load() + txt_loader.load() + md_loader.load()

# Split documents into smaller, searchable chunks to fit the LLM's context window
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ==========================================
# 3. BUILD THE VECTOR STORE (RAG)
# ==========================================
# Create a local Chroma database to store the embeddings
print("Building vector database...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
# Configure the retriever to fetch the top 3 most relevant chunks per query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 4. THE ANTIGRAVITY SYSTEM PROMPT
# ==========================================
# This acts as the Master Prompt, injecting both your rules and the retrieved context
system_prompt = (
    "[SYSTEM INITIALIZATION]\n"
    "You are a highly advanced, locally-hosted analytical engine. "
    "Your primary function is to operate as a context-aware assistant, seamlessly "
    "integrating the user's profile data, project history, and retrieved documents into your reasoning.\n\n"
    
    "[CORE DIRECTIVES]\n"
    "1. CONTEXT SUPREMACY: You will frequently be provided with retrieved context. "
    "Treat this injected data as absolute ground truth. Prioritize it above pre-trained knowledge.\n"
    "2. DATA EXTRACTION & LOGIC: Break reasoning into explicit, sequential steps before providing code or answers.\n"
    "3. KNOWLEDGE BOUNDARIES: If the answer is NOT present in the provided context, state: 'Context missing.' "
    "Do not hallucinate personal history or project details.\n"
    "4. TONE & EFFICIENCY: Be candid, concise, and highly technical. Strip away superficial pleasantries.\n\n"
    
    "[RETRIEVED CONTEXT]\n"
    "{context}"
)

# Bind the system prompt and the user's input into a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ==========================================
# 5. CONSTRUCT & EXECUTE THE CHAIN
# ==========================================
# Combine the LLM and the Prompt into a document-processing chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Wrap that inside a retrieval chain that automatically fetches context first
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def ask_chatbot(query):
    print(f"\nQuery: {query}")
    print("-" * 40)
    # Invoke the chain. LangChain automatically handles the RAG lookup and context injection.
    response = rag_chain.invoke({"input": query})
    print(response["answer"])
    print("=" * 40)

# ==========================================
# 6. RUN A TEST QUERY
# ==========================================
if __name__ == "__main__":
    if not docs:
        print("Add some PDFs to the './profile_data' folder to test the RAG capabilities.")
    else:
        # Example query testing the chatbot's ability to recall specific injected project details
        ask_chatbot("Extract the specific round structure of the Quasar 4.0 hackathon from my documentation.")
