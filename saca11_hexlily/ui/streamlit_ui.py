"""
Streamlit UI components for SACA
Contains the main interface and user interaction logic
"""

import os
import tempfile
import streamlit as st
import logging
from typing import List, Tuple, Optional

from processors.file_processors import (
    process_pcap_file, process_text_file, 
    process_pdf_file, process_json_file
)
from models.llm_models import initialize_llm_model
from embeddings.embeddings import CustomInstructEmbeddings
from core.audit_engine import Saca11


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('saca11.log'),
            logging.StreamHandler()
        ]
    )


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'saca11' not in st.session_state:
        st.session_state.saca11 = None
    if 'log_chat_history' not in st.session_state:
        st.session_state.log_chat_history = []
    if 'policy_chat_history' not in st.session_state:
        st.session_state.policy_chat_history = []


def display_sidebar():
    """Display the sidebar with configuration options"""
    st.sidebar.title("SACA Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Lily Cybersecurity LLM (Recommended)", "Ollama (DeepSeek R1)", "Hugging Face Transformers"],
        help="Lily Cybersecurity LLM is specialized for security analysis with 4-bit quantization for efficiency."
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=500,
            help="Size of text chunks for processing"
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=200, 
            value=50,
            help="Overlap between consecutive chunks"
        )
        top_k_violations = st.number_input(
            "Top K Violations", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Number of top violations to report"
        )
    
    return model_type, chunk_size, chunk_overlap, top_k_violations


def handle_file_uploads() -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Handle file uploads and process them
    Returns: (log_data, policy_texts)
    """
    log_data = []
    policy_texts = []
    
    # Log/PCAP file uploads
    st.subheader("📊 Upload Log/PCAP Files")
    log_files = st.file_uploader(
        "Choose log or PCAP files",
        type=['pcap', 'pcapng', 'txt', 'log', 'json'],
        accept_multiple_files=True,
        help="Upload network capture files (PCAP/PCAPNG) or log files for analysis"
    )
    
    if log_files:
        for uploaded_file in log_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process based on file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['pcap', 'pcapng']:
                        content = process_pcap_file(tmp_file_path)
                    elif file_extension == 'json':
                        content = process_json_file(tmp_file_path)
                    else:  # txt, log, or other text files
                        content = process_text_file(tmp_file_path)
                    
                    if content and content.strip():
                        log_data.append((content, uploaded_file.name))
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name} appears to be empty or unreadable")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    logging.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Policy file uploads
    st.subheader("📋 Upload Policy Documents")
    policy_files = st.file_uploader(
        "Choose policy documents",
        type=['txt', 'pdf', 'json'],
        accept_multiple_files=True,
        help="Upload cybersecurity policy documents for compliance checking"
    )
    
    if policy_files:
        for uploaded_file in policy_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process based on file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'pdf':
                        content = process_pdf_file(tmp_file_path)
                    elif file_extension == 'json':
                        content = process_json_file(tmp_file_path)
                    else:  # txt or other text files
                        content = process_text_file(tmp_file_path)
                    
                    if content and content.strip():
                        policy_texts.append(content)
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name} appears to be empty or unreadable")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    logging.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return log_data, policy_texts


def initialize_saca_system(log_data: List[Tuple[str, str]], policy_texts: List[str], 
                          model_type: str, chunk_size: int, chunk_overlap: int) -> Optional[Saca11]:
    """
    Initialize the SACA system with uploaded data
    """
    if not log_data and not policy_texts:
        st.warning("⚠️ Please upload at least one log/PCAP file and one policy document to proceed.")
        return None
    
    if not log_data:
        st.warning("⚠️ Please upload at least one log/PCAP file for analysis.")
        return None
    
    if not policy_texts:
        st.warning("⚠️ Please upload at least one policy document for compliance checking.")
        return None
    
    try:
        with st.spinner("🔧 Initializing SACA system..."):
            # Initialize LLM
            llm = initialize_llm_model(model_type)
            if not llm:
                st.error("❌ Failed to initialize language model")
                return None
            
            # Initialize embedding model
            embedding_model = CustomInstructEmbeddings()
            
            # Create SACA instance
            saca11 = Saca11(
                log_data=log_data,
                policy_texts=policy_texts,
                embedding_model=embedding_model,
                llm=llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            st.success("✅ SACA system initialized successfully!")
            return saca11
            
    except Exception as e:
        st.error(f"❌ Error initializing SACA system: {str(e)}")
        logging.error(f"Error initializing SACA system: {str(e)}")
        return None


def display_compliance_analysis(saca11: Saca11, top_k_violations: int):
    """Display the compliance analysis section"""
    st.header("🔍 Compliance Analysis")
    
    if st.button("🚀 Run Compliance Analysis", type="primary"):
        with st.spinner("🔍 Analyzing compliance... This may take a few minutes."):
            try:
                analysis_result = saca11.analyze_compliance(top_k=top_k_violations)
                
                st.subheader("📊 Analysis Results")
                st.markdown(analysis_result)
                
                # Option to download results
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=analysis_result,
                    file_name="compliance_analysis_report.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Error during compliance analysis: {str(e)}")
                logging.error(f"Error during compliance analysis: {str(e)}")


def display_document_chat(saca11: Saca11):
    """Display the document chat interface"""
    st.header("💬 Document Chat")
    
    # Document type selection
    doc_type = st.selectbox(
        "Select document type to chat with:",
        ["log", "policy"],
        format_func=lambda x: "Log/PCAP Data" if x == "log" else "Policy Documents"
    )
    
    # Chat interface
    chat_history = (st.session_state.log_chat_history if doc_type == "log" 
                   else st.session_state.policy_chat_history)
    
    # Display chat history
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(f"Ask a question about the {'logs' if doc_type == 'log' else 'policies'}..."):
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = saca11.chat_with_document(prompt, doc_type=doc_type)
                    st.markdown(response)
                    chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    logging.error(f"Error in document chat: {str(e)}")


def display_system_info():
    """Display system information and statistics"""
    with st.expander("ℹ️ System Information"):
        if st.session_state.saca11:
            saca11 = st.session_state.saca11
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Log Chunks", len(saca11.log_chunks))
                st.metric("Policy Chunks", len(saca11.policy_chunks))
            
            with col2:
                st.metric("Chunk Size", saca11.chunk_size)
                st.metric("Chunk Overlap", saca11.chunk_overlap)
            
            # Display policy clauses if available
            if saca11.policy_clauses:
                st.subheader("📋 Detected Policy Clauses")
                for clause_num, clause_text in list(saca11.policy_clauses.items())[:10]:
                    st.text(f"{clause_num}: {clause_text[:100]}...")


def main_ui():
    """Main UI function"""
    st.set_page_config(
        page_title="SACA - Security Audit Compliance Agent",
        page_icon="🔒",
        layout="wide"
    )
    
    # Setup
    setup_logging()
    initialize_session_state()
    
    # Header
    st.title("🔒 SACA - Security Audit Compliance Agent")
    st.markdown("*Automated cybersecurity compliance analysis using RAG and LLMs*")
    
    # Sidebar
    model_type, chunk_size, chunk_overlap, top_k_violations = display_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🔍 Analysis", "💬 Chat", "ℹ️ Info"])
    
    with tab1:
        st.header("📁 File Upload & Processing")
        log_data, policy_texts = handle_file_uploads()
        
        # Initialize system if files are uploaded
        if log_data or policy_texts:
            if st.button("🔧 Initialize SACA System", type="primary"):
                saca11 = initialize_saca_system(
                    log_data, policy_texts, model_type, 
                    chunk_size, chunk_overlap
                )
                if saca11:
                    st.session_state.saca11 = saca11
    
    with tab2:
        if st.session_state.saca11:
            display_compliance_analysis(st.session_state.saca11, top_k_violations)
        else:
            st.info("👆 Please upload files and initialize the system first.")
    
    with tab3:
        if st.session_state.saca11:
            display_document_chat(st.session_state.saca11)
        else:
            st.info("👆 Please upload files and initialize the system first.")
    
    with tab4:
        display_system_info()
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, LangChain, and ChromaDB*")


if __name__ == "__main__":
    main_ui()