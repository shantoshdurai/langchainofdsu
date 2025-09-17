from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Choose embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Load your university data
texts = [
    "Class starts at 9 AM every day.",
    "The library closes at 7 PM.",
    "Final exams are in December.",
    "The canteen offers vegetarian food."
]

# Step 3: Store in ChromaDB
db = Chroma.from_texts(texts, embeddings)

# Step 4: Connect to Llama with Retrieval
llm = OllamaLLM(model="llama3.1:8b")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Step 5: Ask questions
query = "When does the library close?"
print(qa.run(query))

query = "What time do classes start?"
print(qa.run(query))
