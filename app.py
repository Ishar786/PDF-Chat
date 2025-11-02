import streamlit as st
import os
from io import BytesIO
from typing import List, Any
from dotenv import load_dotenv

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
# as it runs before the key is provided.
@st.cache_resource
def get_embeddings():
    """Loads the multilingual embedding model once."""
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
        google_api_key=api_key # The runtime key is passed here
    )
    
    # Custom Prompt Template
    RAG_PROMPT_TEMPLATE = """
You are an expert financial research analyst. Your sole purpose is to answer the user's question accurately and concisely,
based **ONLY** on the provided context.

**STRICT RULES:**
1. **NO OUTSIDE KNOWLEDGE:** You must only use the text provided in the 'CONTEXT' section below.
2. **NO DATA FOUND:** If the context does not contain the answer, you **MUST** respond with the exact phrase: "No relevant data found in the provided documents."
3. **LANGUAGE:** The answer **MUST** be in English, even if the source text is in another language (you must translate).
4. **CITATION:** After your answer, you MUST provide a list of citations in a markdown block, referencing the source file name and page number.

CONTEXT:
{context}

QUESTION: {question}

**FINAL ENGLISH ANSWER:**
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
    """Handles file upload, parsing, chunking, and FAISS indexing."""
    
    temp_pdf_stream = BytesIO(uploaded_file.read())
    pdf_reader = PdfReader(temp_pdf_stream)
    num_pages = len(pdf_reader.pages)
    
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        page_content = page.extract_text()
        
        if page_content:
            chunks = text_splitter.split_text(page_content)
            
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

    if not all_documents:
        return None, 0

    db = FAISS.from_documents(all_documents, embeddings_model)
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
    # Secure runtime input for the API key
    api_key_input = st.text_input(
        "Enter your Gemini API Key:", 
        type="password",
        help="The key is stored securely in the app's memory (session state) and is required to run the model."
    )
    
    if api_key_input:
        st.session_state.api_key_valid = True
        # Store the key in session state for later use in create_rag_chain
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
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                # 1. Perform indexing
                embeddings_model = get_embeddings()
                db, num_chunks = index_documents(uploaded_file, embeddings_model)
                
                if db:
                    # 2. Create and store the RAG chain using the runtime key
                    st.session_state.rag_chain = create_rag_chain(db, st.session_state.api_key)
                    st.success(f"✅ RAG System Ready! {num_chunks} chunks indexed.")
                else:
                    st.error("Indexing failed: Could not extract usable text.")
            except Exception as e:
                st.error(f"An error occurred during indexing: {e}")

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
                st.error(f"An error occurred during the query. Check your key and permissions. Error: {e}")

else:
    st.info("Please follow the steps in the sidebar: 1. Enter your API Key, and 2. Upload and Index your PDF.")
