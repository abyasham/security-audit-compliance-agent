import os
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, MessageGraph

# Configuration
DOC_DIR = "docs"
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-r1:14b"

def validate_environment():
    """Check required services are running"""
    try:
        import ollama
        ollama.list()
    except Exception:
        print("Error: Ollama service not running. Please start it first.")
        print("Installation guide: https://ollama.ai/download")
        exit(1)

def load_documents() -> List[Dict]:
    """Load all supported documents from docs directory"""
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".json": lambda path: JSONLoader(path, jq_schema=".", text_content=False),
        ".csv": CSVLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
    }
    
    documents = []
    for filename in os.listdir(DOC_DIR):
        filepath = os.path.join(DOC_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in loaders:
            try:
                loader = loaders[ext](filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return documents

def initialize_components():
    """Create vector store and processing chain"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    documents = load_documents()
    if not documents:
        print("No documents found in 'docs' directory")
        exit(1)
        
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LangGraph setup for conversation flow
    def router(state):
        if "question" in state:
            return "generate_answer"
        return "end"
    
    workflow = MessageGraph()
    workflow.add_node("generate_answer", lambda state: {"answer": "..."})
    workflow.add_conditional_edges("start", router)
    workflow.set_entry_point("start")
    workflow.add_edge("generate_answer", END)
    
    # RAG prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context. 
        If you don't know the answer, say you don't know.
        Context: {context}
        Question: {question}"""
    )
    
    from langchain_community.chat_models import ChatOllama
    model = ChatOllama(model=OLLAMA_MODEL)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain, workflow

def chat_loop(chain):
    """Interactive chat interface"""
    print("RAG Chat System - Type 'exit' to quit\n")
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        response = chain.invoke(query)
        print(f"\nAnswer: {response}\n")

if __name__ == "__main__":
    validate_environment()
    
    if not os.path.exists(DOC_DIR):
        os.makedirs(DOC_DIR)
        print(f"Created '{DOC_DIR}' directory - add your documents there")
        exit(0)
        
    rag_chain, _ = initialize_components()
    chat_loop(rag_chain)