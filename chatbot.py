
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage, HumanMessage


class Chatbot:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        temperature: float = 0.7,
        system_message: str = None,
        memory_window: int = 5
    ):
        """Initialize the chatbot with LLM and memory.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness in the model's responses
            system_message: Initial system message to set the assistant's behavior
            memory_window: Number of past messages to keep in memory
        """
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096  # Increase context window
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=memory_window
        )
        self.vectorstore = None
        self.retriever = None
        self.conversation_chain = None
        self.system_message = system_message or """
        You are a helpful AI assistant for DSU university students and staff.
        Answer questions based on the provided context and be as helpful as possible.
        If you don't know the answer, say you don't know instead of making up an answer.
        """
        self.chat_history = []

    def load_documents(self, document_paths: List[str]) -> None:
        """Load documents from various file types and create vector store.
        
        Supported formats: .pdf, .txt, .md, .docx
        """
        if not document_paths:
            raise ValueError("No document paths provided")
            
        documents = []
        
        for path in document_paths:
            path = str(Path(path).absolute())
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
                
            try:
                if path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                elif path.lower().endswith(('.txt', '.md')):
                    loader = TextLoader(path, encoding='utf-8')
                    documents.extend(loader.load())
                elif path.lower().endswith(('.doc', '.docx')):
                    loader = UnstructuredWordDocumentLoader(path)
                    documents.extend(loader.load())
                else:
                    print(f"Warning: Unsupported file format: {path}")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
        
        if not documents:
            raise ValueError("No valid documents were loaded")
            
        print(f"Loaded {len(documents)} document chunks.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            splits,
            OllamaEmbeddings(model="llama3.1:8b")
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum marginal relevance
            search_kwargs={"k": 5}  # Number of documents to retrieve
        )

    def setup_conversation_chain(self):
        """Set up the conversation chain with the LLM and retriever."""
        if not self.retriever:
            # No retriever available (no docs). We'll use chat-only mode.
            self.conversation_chain = None
            return
        # Define the prompt template
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        # Create the conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )

    def chat(self, question: str) -> Dict[str, Any]:
        """Process a user question and return the assistant's response.
        Fallback to chat-only mode if no retriever/chain is set.
        """
        if self.retriever and not self.conversation_chain:
            self.setup_conversation_chain()

        try:
            # Add timestamp and user question to chat history
            self.chat_history.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })

            if self.conversation_chain:
                # RAG path
                response = self.conversation_chain.invoke({"question": question})
                answer_text = response["answer"]
                sources = [doc.metadata.get("source", "Unknown") for doc in response.get("source_documents", [])]
            else:
                # Chat-only path (no RAG)
                # Build a simple prompt with system message + brief conversation context
                window = self.memory.k if hasattr(self.memory, 'k') else 5
                recent = self.chat_history[-(2 * window):]  # approximate pairs
                convo_lines = []
                for msg in recent:
                    prefix = "User" if msg["role"] == "user" else "Assistant"
                    convo_lines.append(f"{prefix}: {msg['content']}")
                prompt = (
                    f"{self.system_message.strip()}\n\n"  # system behavior
                    f"Conversation so far:\n" + "\n".join(convo_lines) + "\n\n" +
                    f"User: {question}\nAssistant:"
                )
                answer_text = self.llm.invoke(prompt)
                sources = []

            # Add assistant's response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "answer": answer_text,
                "sources": list(set(sources))
            }

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(error_msg)
            return {"answer": "I'm sorry, I encountered an error processing your request.", "sources": []}


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the chatbot banner."""
    banner = """
    ╔══════════════════════════════════════╗
    ║         DSU University Chatbot           ║
    ║  Powered by LangChain & Ollama LLM   ║
    ╚══════════════════════════════════════╝
    Type 'exit' to quit. Type 'clear' to clear the screen.
    """
    print(banner)

def main():
    # Clear screen and show banner
    clear_screen()
    print_banner()
    
    # System message that defines the chatbot's behavior
    system_message = """
    You are "DSU Chatbot", an AI assistant for university students and staff.
    
    Your purpose is to provide helpful and accurate information about the university, 
    including but not limited to:
    - Academic programs and courses
    - University rules and regulations
    - Important dates and deadlines
    - Campus facilities and services
    - Events and activities
    - Administrative procedures
    
    Guidelines for responses:
    1. Be polite, professional, and helpful at all times
    2. Base your answers strictly on the provided context
    3. If you don't know the answer, say so instead of making up information
    4. For complex queries, break down the information into clear, organized points
    5. If a question is unclear, ask for clarification
    6. Always refer to the university as "DSU TRICHY"
    7. Dsu university is located in Trichy, Tamil Nadu, India
    """
    
    # Initialize the chatbot
    print("Initializing UniBot...")
    chatbot = Chatbot(
        model_name="llama3.1:8b",
        temperature=0.7,
        system_message=system_message,
        memory_window=10  
    )
    
    # Load documents from the data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print(f"Created 'data' directory. Please add your documents (PDF, TXT, DOCX) there and restart.")
        # Continue to chat-only mode even if no documents
    else:
        # Find all supported documents in the data directory
        supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc']
        document_paths = []
        for ext in supported_extensions:
            document_paths.extend(list(data_dir.glob(f'*{ext}')))
        
        has_docs = len(document_paths) > 0
        if not has_docs:
            print(f"No documents found in the 'data' directory. Running in chat-only mode (no RAG).")
            print("Supported formats: " + ", ".join(supported_extensions))

    try:
        if has_docs:
            print("Loading documents...")
            chatbot.load_documents([str(path) for path in document_paths])
        chatbot.setup_conversation_chain()
        
        print("\nDSU Chatbot is ready! How can I help you today?")
        print("Type 'exit' to quit. Type 'clear' to clear the screen.\n")
        
        # Main chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("Goodbye! Have a great day!")
                    break
                    
                if user_input.lower() == 'clear':
                    clear_screen()
                    print_banner()
                    continue
                
                # Get response from the chatbot
                response = chatbot.chat(user_input)
                
                # Print the response
                print("\n" + "="*80)
                print(f"DSU Chatbot: {response['answer']}")
                
                # Print sources if available
                if response.get('sources'):
                    print("\nSources:")
                    for i, source in enumerate(set(response['sources']), 1):
                        print(f"  {i}. {os.path.basename(source)}")
                print("="*80)
                
            except KeyboardInterrupt:
                print("\nType 'exit' to quit or continue chatting...")
                continue
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running (run 'ollama serve' in a terminal)")
        print("2. Ensure you have the model: 'ollama pull llama3.1:8b'")
        print("3. Check that your documents are in the 'data' directory")
        print("4. Verify your documents are in a supported format (PDF, TXT, DOCX, MD)")
        print("\nRestart the chatbot after fixing these issues.")

if __name__ == "__main__":
    main()
