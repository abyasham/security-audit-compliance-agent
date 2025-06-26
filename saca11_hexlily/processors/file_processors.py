"""
File Processing module for SACA
Handles PCAP, text, PDF, and JSON file processing
"""

import os
import subprocess
import tempfile
import json
import streamlit as st


def extract_text_from_policy(uploaded_file):
    """
    Extract text from a policy file. Supports plain text and PDF.
    For PDF extraction, PyPDF2 is used.
    """
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
        except ImportError:
            st.error("Please install PyPDF2 to process PDF files.")
            return ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "application/json":
        try:
            json_data = json.load(uploaded_file)
            # Convert JSON to a readable string format for embedding
            text = json.dumps(json_data, indent=2)
            st.success(f"Successfully loaded JSON policy: {uploaded_file.name}")
        except json.JSONDecodeError:
            st.error(f"Invalid JSON file: {uploaded_file.name}. Please upload a valid JSON.")
            text = ""
        except Exception as e:
            st.error(f"Error processing JSON policy file: {str(e)}")
            text = ""
    else:
        st.error(f"Unsupported file format for policy file: {uploaded_file.type}")
        text = ""
    return text


def process_log_file(uploaded_file):
    """
    Process a log file. If a PCAP/PCAPNG file is uploaded, use tshark to convert it to JSON.
    Otherwise, assume a plain text log.
    """
    file_ext = os.path.splitext(uploaded_file.name.lower())[1]
    
    if file_ext in ['.pcap', '.pcapng', '.cap']:
        return _process_pcap_file(uploaded_file)
    else:
        return _process_text_file(uploaded_file)


def _process_pcap_file(uploaded_file):
    """Process PCAP/PCAPNG files using tshark"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name.lower())[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    st.info(f"Processing packet capture file: {uploaded_file.name}")
    
    try:
        # Use tshark to extract payload hex dump (data.data field) and stream output
        command = ["tshark", "-nlr", tmp_path, "-x"] # -x for full packet hex dump
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        log_text = ""
        for line in process.stdout:
            log_text += line.strip() + "\n"
        
        # Check for errors
        stderr = process.stderr.read().strip()
        if stderr:
            st.error(f"Error processing PCAP file: {stderr}")
            return f"Error processing file: {stderr}"
        
        log_text = log_text.strip() # Strip whitespace
        
        if log_text:
            st.success(f"Successfully extracted payload hex dump from {uploaded_file.name}")
        else:
            st.warning("PCAP file produced an empty payload hex dump after stripping. The file might be empty or contain no data fields.")
            log_text = "Empty PCAP file or no payload data extracted."
            
    except Exception as e:
        st.error(f"Error during PCAP processing: {str(e)}")
        log_text = f"Error processing file: {str(e)}"
    
    # Clean up temporary files
    try:
        os.remove(tmp_path)
    except Exception as e:
        st.warning(f"Could not remove temporary files: {str(e)}")
    
    return log_text


def _process_text_file(uploaded_file):
    """Process text-based log files"""
    try:
        log_text = uploaded_file.read().decode("utf-8")
        st.success(f"Successfully loaded text log: {uploaded_file.name}")
        return log_text
    except UnicodeDecodeError:
        st.error(f"Cannot decode {uploaded_file.name} as text. Make sure it's a valid text file.")
        return f"Error: Could not decode {uploaded_file.name} as text."
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")
        return f"Error reading file: {str(e)}"


def process_uploaded_files(log_files, policy_files):
    """
    Process all uploaded files and return processed data
    
    Returns:
        tuple: (processed_log_data, policy_texts) or (None, None) if processing fails
    """
    # Process the log files (pcap conversion or plain text)
    st.info("Processing log files...")
    processed_log_data = []
    for i, uploaded_log_file in enumerate(log_files):
        log_text = process_log_file(uploaded_log_file)
        if log_text:
            processed_log_data.append((log_text, uploaded_log_file.name))
            st.success(f"Processed log file: {uploaded_log_file.name}")
        else:
            st.error(f"Failed to process log file: {uploaded_log_file.name}")
    
    if not processed_log_data:
        st.error("Could not process any of the uploaded log files.")
        return None, None
    
    # Process policy files and extract their text
    st.info("Processing policy files...")
    policy_texts = []
    for file in policy_files:
        text = extract_text_from_policy(file)
        if text:
            policy_texts.append(text)
            st.success(f"Processed policy file: {file.name}")
        else:
            st.error(f"Failed to extract text from: {file.name}")
    
    if not policy_texts:
        st.error("Could not extract text from any of the uploaded policy files.")
        return None, None
    
    return processed_log_data, policy_texts
def process_pcap_file(file_path):
    """
    Process PCAP file using tshark to extract readable content
    
    Args:
        file_path: Path to the PCAP file
        
    Returns:
        String containing extracted packet data
    """
    try:
        # Use tshark to convert PCAP to JSON format
        result = subprocess.run([
            'tshark', '-r', file_path, '-T', 'json', '-c', '1000'  # Limit to 1000 packets
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"tshark error: {result.stderr}")
            return ""
    except subprocess.TimeoutExpired:
        st.error("PCAP processing timed out")
        return ""
    except FileNotFoundError:
        st.error("tshark not found. Please install Wireshark.")
        return ""
    except Exception as e:
        st.error(f"Error processing PCAP file: {str(e)}")
        return ""


def process_text_file(file_path):
    """
    Process text file and return content
    
    Args:
        file_path: Path to the text file
        
    Returns:
        String containing file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    except Exception as e:
        st.error(f"Error processing text file: {str(e)}")
        return ""


def process_pdf_file(file_path):
    """
    Process PDF file and extract text content
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        String containing extracted text
    """
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        st.error("Please install PyPDF2 to process PDF files: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        return ""


def process_json_file(file_path):
    """
    Process JSON file and return formatted content
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        String containing formatted JSON content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            return json.dumps(json_data, indent=2)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error processing JSON file: {str(e)}")
        return ""