import streamlit as st
import os
from io import BytesIO
from typing import List, Any
from dotenv import load_dotenv
import time # Import time for the simulated progress

# --- LangChain Core Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate 
from langchain_core.documents import Document 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pypdf import PdfReader 


# --- Configuration and Initialization ---

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Cache the embedding model download - Note: This function MUST NOT depend on the API key
@st.cache_resource
def get_embeddings():
    """Loads the multilingual embedding model once."""
    with st.spinner("Downloading and caching multilingual embedding model..."):
        return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# --- Core RAG Logic Functions ---

def format_docs_for_context(docs: List[Document]) -> str:
    """Formats the retrieved documents to clearly show source and page number."""
    formatted_text = ""
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.replace('{', '(').replace('}', ')')
        
        formatted_text += (
            f"--- SOURCE: {source}, PAGE: {page} ---\n"
            f"{content}\n"
            "------------------------------------\n\n"
        )
    return formatted_text

def create_rag_chain(vector_db: FAISS, api_key: str):
    """Initializes the LLM and creates the LCEL chain."""
    
    # CRITICAL: Pass the API key directly to the LLM constructor
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        google_api_key=api_key 
    )
    
    # Custom Prompt Template
    RAG_PROMPT_TEMPLATE = """
You are an expert financial research analyst. Your sole purpose is to answer the user's question accurately and concisely,
based **ONLY** on the provided context.
... (Prompt continues as before)
"""
    custom_rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # The LCEL Chain
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs_for_context,
            question=RunnablePassthrough()
        )
        | custom_rag_prompt
        | llm
        | StrOutputParser() 
    )
    return rag_chain


def index_documents(uploaded_file, embeddings_model):
    """Handles file upload, parsing, chunking, and FAISS indexing with a progress bar."""
    
    temp_pdf_stream = BytesIO(uploaded_file.read())
    pdf_reader = PdfReader(temp_pdf_stream)
    num_pages = len(pdf_reader.pages)
    
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    # Initialize Streamlit progress bar elements
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    page_chunks = {}

    # PHASE 1: Parsing and Chunking
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        page_content = page.extract_text()
        
        if page_content:
            chunks = text_splitter.split_text(page_content)
            page_chunks[page_num] = chunks

        # Update progress bar for parsing (first half of the work)
        progress = int((page_num + 1) / num_pages * 50)
        progress_text.text(f"Parsing and Chunking Page {page_num + 1} of {num_pages}...")
        progress_bar.progress(progress)
        
    # PHASE 2: Embedding and Indexing
    total_chunks = sum(len(c) for c in page_chunks.values())
    processed_chunks = 0
    
    for page_num, chunks in page_chunks.items():
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": uploaded_file.name,
                    "page": page_num + 1, 
                    "chunk_id": f"{uploaded_file.name}_{page_num + 1}_{i+1}"
                }
            )
            all_documents.append(doc)
            processed_chunks += 1
            
            # Update progress bar for embedding (second half of the work)
            current_progress = 50 + int((processed_chunks / total_chunks) * 50)
            progress_text.text(f"Embedding chunk {processed_chunks} of {total_chunks}...")
            progress_bar.progress(current_progress)


    if not all_documents:
        progress_bar.empty()
        progress_text.empty()
        return None, 0

    # Final step: Create FAISS index
    db = FAISS.from_documents(all_documents, embeddings_model)
    
    progress_bar.progress(100)
    progress_text.text("Indexing complete!")
    
    # Clear the bar and text after a brief pause
    time.sleep(1)
    progress_bar.empty()
    progress_text.empty()
    
    return db, len(all_documents)


# --- Streamlit Application UI ---

st.set_page_config(page_title="PDF Analyst RAG Tool", layout="wide")

st.title("🏦 Analyst RAG Tool: Document-Grounded Q&A")
st.markdown("Upload bank documents and ask questions based ONLY on the content. **Citations provided.**")

# Initialize chat history and RAG status in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR: Configuration and Indexing ---
with st.sidebar:
    st.header("1. API Key & Authentication")
    api_key_input = st.text_input(
        "Enter your Gemini API Key:", 
        type="password",
        help="The key is stored securely in the app's memory (session state) and is required to run the model."
    )
    
    if api_key_input:
        st.session_state.api_key_valid = True
        st.session_state.api_key = api_key_input
        st.success("API Key loaded successfully.")
    else:
        st.session_state.api_key_valid = False
        st.warning("Please enter your API Key to enable the Q&A system.")


    st.header("2. Document Indexing")
    uploaded_file = st.file_uploader(
        "Upload a PDF Document (e.g., Annual Report)", 
        type="pdf", 
        accept_multiple_files=False
    )

    index_button = st.button("Create Search Index")

    if index_button and uploaded_file and st.session_state.api_key_valid:
        
        # Load cached model first
        embeddings_model = get_embeddings()

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                # 1. Perform indexing (Progress bar runs inside this function)
                db, num_chunks = index_documents(uploaded_file, embeddings_model)
                
                if db:
                    # 2. Create and store the RAG chain using the runtime key
                    st.session_state.rag_chain = create_rag_chain(db, st.session_state.api_key)
                    st.success(f"✅ RAG System Ready! {num_chunks} chunks indexed.")
                else:
                    st.error("Indexing failed: Could not extract usable text.")
            except Exception as e:
                st.error(f"An error occurred during indexing: {e}")
                st.error("Hint: Ensure the API Key is correct, as initialization may fail.")

    elif index_button and not st.session_state.api_key_valid:
        st.error("Cannot index: Please provide a valid API Key first.")


# --- Main Chat Interface ---

is_ready = st.session_state.rag_chain is not None

if is_ready:
    st.markdown("---")
    st.subheader(f"Ask Questions about: {uploaded_file.name}")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Searching and generating response..."):
            try:
                # Invoke the stored RAG chain
                response = st.session_state.rag_chain.invoke(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred during the query. Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Query Error: {e}"})

else:
    st.info("Please follow the steps in the sidebar: 1. Enter your API Key, and 2. Upload and Index your PDF.")
