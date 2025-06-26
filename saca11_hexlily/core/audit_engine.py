"""
Core Audit Engine for SACA
Contains the main Saca11 class for compliance analysis and document chat
"""

import os
import re
import logging
import streamlit as st
import torch
import chromadb
from langchain_community.vectorstores import Chroma

from embeddings.embeddings import ChromaEmbeddingFunctionWrapper
from utils.utils import split_text


def extract_policy_clauses(policy_text):
    """
    Extract numbered policy clauses using a regex.
    Returns a dictionary mapping clause numbers (e.g. "1.1.1") to clause text.
    """
    pattern = r'(\d+(?:\.\d+)+)\s+(.*)'
    clauses = re.findall(pattern, policy_text)
    clause_dict = {}
    for clause, content in clauses:
        clause_dict[clause] = content.strip()
    return clause_dict


class Saca11:
    """Main Security Audit Compliance Agent class"""
    
    def __init__(self, log_data, policy_texts, embedding_model, llm, chunk_size=500, chunk_overlap=50):
        """
        Initialize the audit with the log (or pcap) text and a list of policy texts.
        Both documents are split into chunks and stored in local vector DBs.
        
        :param log_data: List of tuples (log_text, log_name) from log or pcap files
        :param policy_texts: List of policy document texts
        :param embedding_model: Model for creating embeddings
        :param llm: Language model for analysis
        :param chunk_size: Size of text chunks
        :param chunk_overlap: Overlap between chunks
        """
        self.log_data = log_data # Store the raw log data
        self.policy_texts = policy_texts
        # Store the combined policy text for prompt generation
        self.policy_text = "\n\n".join(policy_texts)
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # For chat history
        self.log_chat_history = []
        self.policy_chat_history = []

        # Extract policy clauses for better reference
        self.policy_clauses = extract_policy_clauses(self.policy_text)

        # Split documents into chunks with metadata
        self._prepare_chunks()

        # Build vector databases for the log and policies
        self.build_vector_dbs()

    def _prepare_chunks(self):
        """Prepare text chunks from log and policy data"""
        self.log_chunks = []
        for log_text, log_name in self.log_data:
            chunks = split_text(
                log_text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                source_type="log",
                source_name=log_name
            )
            if chunks:
                self.log_chunks.extend(chunks)
            else:
                logging.warning(f"No valid chunks generated for log file: {log_name}. It might be empty or contain only unprocessable content.")
        
        # Process each policy document separately to maintain source information
        self.policy_chunks = []
        for i, policy_text in enumerate(self.policy_texts):
            policy_name = f"policy_{i+1}"
            chunks = split_text(
                policy_text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                source_type="policy",
                source_name=policy_name
            )
            self.policy_chunks.extend(chunks)

    def build_vector_dbs(self):
        """Build vector databases for logs and policies with metadata"""
        st.write("Building vector DB for pcap or log data...")

        # Ensure the persist directory exists
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "persist")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize Chroma persistent client
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Create a wrapper for the embedding model to be compatible with ChromaDB's EmbeddingFunction
        chroma_embedding_function = ChromaEmbeddingFunctionWrapper(self.embedding_model)

        # Build log database
        self._build_log_database(client, chroma_embedding_function)
        
        # Build policy database
        self._build_policy_database(client, chroma_embedding_function)

    def _build_log_database(self, client, chroma_embedding_function):
        """Build vector database for log data"""
        # Extract texts and metadata from chunks
        if self.log_chunks and not isinstance(self.log_chunks[0], dict):
            st.error(f"Error: log_chunks contains {type(self.log_chunks[0])} instead of dict. First item: {self.log_chunks[0]}")
            return
        
        log_texts = []
        log_metadatas = []
        for i, chunk in enumerate(self.log_chunks):
            if isinstance(chunk, dict):
                log_texts.append(chunk["text"])
                metadata = {
                    "source_type": str(chunk.get("source_type", "log")),
                    "source_name": str(chunk.get("source_name", "unknown")),
                    "chunk_id": str(i)
                }
            else:
                # Fallback for string chunks
                st.warning(f"Warning: Found string chunk instead of dict at index {i}: {chunk}")
                log_texts.append(str(chunk))
                metadata = {
                    "source_type": "log",
                    "source_name": "unknown",
                    "chunk_id": str(i)
                }
            log_metadatas.append(metadata)
        
        # Get or create the 'logs' collection
        log_collection = client.get_or_create_collection(
            name="logs",
            embedding_function=chroma_embedding_function
        )
        
        # Add texts to the collection if it's empty or needs updating
        if log_texts:
            if log_collection.count() == 0 or len(log_texts) != log_collection.count():
                st.write("Adding log data to ChromaDB...")
                if log_collection.count() > 0:
                    log_collection.delete(ids=[f"log_chunk_{i}" for i in range(log_collection.count())])
                
                log_ids = [f"log_chunk_{i}" for i in range(len(log_texts))]
                log_collection.add(
                    documents=log_texts,
                    metadatas=log_metadatas,
                    ids=log_ids
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            st.warning("No log data to add to ChromaDB. Skipping log collection update.")
        
        self.log_db = Chroma(
            client=client,
            collection_name="logs",
            embedding_function=chroma_embedding_function
        )

    def _build_policy_database(self, client, chroma_embedding_function):
        """Build vector database for policy data"""
        st.write("Building vector DB for policy data...")
        
        # Debug: Check the structure of policy_chunks
        if self.policy_chunks and not isinstance(self.policy_chunks[0], dict):
            st.error(f"Error: policy_chunks contains {type(self.policy_chunks[0])} instead of dict. First item: {self.policy_chunks[0]}")
            return
        
        policy_texts = []
        policy_metadatas = []
        for i, chunk in enumerate(self.policy_chunks):
            if isinstance(chunk, dict):
                policy_texts.append(chunk["text"])
                metadata = {
                    "source_type": str(chunk.get("source_type", "policy")),
                    "source_name": str(chunk.get("source_name", "unknown")),
                    "chunk_id": str(i)
                }
            else:
                # Fallback for string chunks
                st.warning(f"Warning: Found string chunk instead of dict at index {i}: {chunk}")
                policy_texts.append(str(chunk))
                metadata = {
                    "source_type": "policy",
                    "source_name": "unknown",
                    "chunk_id": str(i)
                }
            policy_metadatas.append(metadata)
        
        policy_collection = client.get_or_create_collection(
            name="policies",
            embedding_function=chroma_embedding_function
        )
        
        if policy_texts:
            if policy_collection.count() == 0 or len(policy_texts) != policy_collection.count():
                st.write("Adding policy data to ChromaDB...")
                if policy_collection.count() > 0:
                    policy_collection.delete(ids=[f"policy_chunk_{i}" for i in range(policy_collection.count())])
                
                policy_ids = [f"policy_chunk_{i}" for i in range(len(policy_texts))]
                policy_collection.add(
                    documents=policy_texts,
                    metadatas=policy_metadatas,
                    ids=policy_ids
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            st.warning("No policy data to add to ChromaDB. Skipping policy collection update.")
            
        self.policy_db = Chroma(
            client=client,
            collection_name="policies",
            embedding_function=chroma_embedding_function
        )

    def chat_with_document(self, query, doc_type="log", k=5):
        """
        Chat with either logs or policy documents using RAG approach.
        
        :param query: User's question
        :param doc_type: Type of document to query ("log" or "policy")
        :param k: Number of chunks to retrieve
        :return: Response from the LLM
        """
        # Select the appropriate vector DB based on document type
        db = self.log_db if doc_type == "log" else self.policy_db
        
        if not db.get(limit=1)['ids']:
            return f"No {doc_type} data available to chat with. Please ensure your files contain valid and extractable data."
        
        # Update chat history
        chat_history = self.log_chat_history if doc_type == "log" else self.policy_chat_history
        chat_history.append({"role": "user", "content": query})
        
        # Retrieve relevant chunks
        try:
            # Try to use MMR search for better diversity
            docs = db.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k*2,
                lambda_mult=0.5
            )
        except (AttributeError, TypeError, ValueError) as e:
            # Fall back to regular similarity search
            docs = db.similarity_search(query, k=k)
        
        # Format chunks with metadata for better context
        chunks_formatted = []
        for i, doc in enumerate(docs):
            try:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = str(metadata.get('source_name', 'unknown')) if metadata else 'unknown'
            except (AttributeError, TypeError):
                source = 'unknown'
            
            chunks_formatted.append(f"CHUNK {i+1} [Source: {source}]:\n{doc.page_content}")
        
        context = "\n\n".join(chunks_formatted)
        
        # Format chat history for context
        history_formatted = ""
        if len(chat_history) > 1:  # If there's previous conversation
            for i, msg in enumerate(chat_history[:-1]):  # Exclude current query
                role = "User" if msg["role"] == "user" else "Assistant"
                history_formatted += f"{role}: {msg['content']}\n\n"
        
        # Build prompt based on document type
        doc_type_name = "security logs" if doc_type == "log" else "security policy documents"
        
        # Build the prompt with proper string concatenation to avoid f-string backslash issues
        prompt_parts = [
            f"You are a cybersecurity expert assistant specializing in analyzing {doc_type_name}.",
            "",
            "CONTEXT INFORMATION:",
            "--------------------",
            context,
            "--------------------",
            "",
        ]
        
        # Add conversation history if it exists
        if history_formatted:
            prompt_parts.extend([
                "PREVIOUS CONVERSATION:",
                history_formatted,
                ""
            ])
        
        prompt_parts.extend([
            f"USER QUERY: {query}",
            "",
            "INSTRUCTIONS:",
            "1. Answer the user's query based on the provided context information.",
            "2. If the answer is not in the context, say so clearly rather than making up information.",
            "3. Be concise but thorough in your response.",
            "4. When referencing specific parts of logs or policies, cite the source if available."
        ])
        
        # Join all parts with newlines
        prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.llm.invoke(
            input=prompt,
            temperature=0.0  # Set temperature to 0 for more deterministic output
        )
        
        # Update chat history with assistant's response
        chat_history.append({"role": "assistant", "content": response})
        
        return response

    def analyze_compliance(self, top_k=3, log_k=5, log_fetch_k=10, policy_k=10):
        """
        Uses an enhanced retrieval-augmented approach to compare log events against
        cybersecurity policies, with improved context retrieval and prompt engineering
        to reduce hallucination.
        
        :param top_k: Number of top violations to report
        :param log_k: Number of log chunks to retrieve per query
        :param log_fetch_k: Number of log chunks to consider before selecting diverse subset
        :param policy_k: Number of policy chunks to retrieve
        :return: Analysis response from the LLM
        """
        try:
            # Check if log database has data
            log_data_check = self.log_db.get(limit=1)
            if not log_data_check['ids']:
                return "No log data available for compliance analysis. Please ensure your log/PCAP files contain valid and extractable data."
            logging.info(f"Log database contains {len(log_data_check['ids'])} items")
        except Exception as e:
            error_msg = f"Error accessing log database: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)
            return f"Failed to access log database: {str(e)}"
        
        # Step 1: Generate multiple queries to improve retrieval coverage
        base_queries = [
            "Identify log events that potentially violate cybersecurity policies",
            "Find suspicious network activities in logs",
            "Detect unauthorized access attempts in logs",
            "Identify data exfiltration evidence in logs",
            "Find policy violations related to authentication in logs"
        ]
        logging.info(f"Generated base queries: {base_queries}")
        
        # Step 2: Retrieve relevant log chunks using multiple queries
        all_retrieved_docs = []
        for query in base_queries:
            logging.info(f"Retrieving log chunks for query: {query}")
            try:
                # Try to use MMR search for better diversity
                docs = self.log_db.max_marginal_relevance_search(
                    query,
                    k=log_k,  # Retrieve log_k docs per query
                    fetch_k=log_fetch_k,  # Consider log_fetch_k before selecting diverse subset
                    lambda_mult=0.5  # Balance between relevance and diversity
                )
            except (AttributeError, TypeError, ValueError) as e:
                # Fall back to regular similarity search if MMR is not available
                st.warning(f"Using regular similarity search instead of MMR: {str(e)}")
                logging.warning(f"Using regular similarity search instead of MMR for query {query}: {str(e)}")
                try:
                    docs = self.log_db.similarity_search(query, k=log_k)
                except Exception as search_error:
                    error_msg = f"Error in similarity search for query '{query}': {str(search_error)}"
                    logging.error(error_msg)
                    st.error(error_msg)
                    return f"Failed during log retrieval: {str(search_error)}"
            
            all_retrieved_docs.extend(docs)
            logging.info(f"Retrieved {len(docs)} log chunks for query: {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        # Limit to top 15 most relevant chunks
        retrieved_docs = unique_docs[:15]
        
        # Step 3: Retrieve relevant policy chunks based on the log content
        # Create a combined query from the retrieved log chunks
        log_content = " ".join([doc.page_content for doc in retrieved_docs[:3]])
        policy_query = f"Find policies relevant to: {log_content}"
        logging.info(f"Generated policy query: {policy_query}")
        
        # Retrieve relevant policy chunks
        try:
            policy_docs = self.policy_db.similarity_search(
                policy_query,
                k=policy_k
            )
            logging.info(f"Retrieved {len(policy_docs)} policy chunks")
        except Exception as e:
            error_msg = f"Error retrieving policy chunks: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)
            return f"Failed during policy retrieval: {str(e)}"
        
        # Format log chunks with metadata for better context
        log_chunks_formatted = []
        for i, doc in enumerate(retrieved_docs):
            # Safely handle metadata
            try:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = str(metadata.get('source_name', 'unknown')) if metadata else 'unknown'
            except (AttributeError, TypeError):
                source = 'unknown'
            
            log_chunks_formatted.append(f"LOG CHUNK {i+1} [Source: {source}]:\n{doc.page_content}")
        
        log_context = "\n\n".join(log_chunks_formatted)
        
        # Format policy chunks with metadata
        policy_chunks_formatted = []
        for i, doc in enumerate(policy_docs):
            # Safely handle metadata
            try:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = str(metadata.get('source_name', 'unknown')) if metadata else 'unknown'
            except (AttributeError, TypeError):
                source = 'unknown'
                
            policy_chunks_formatted.append(f"POLICY CHUNK {i+1} [Source: {source}]:\n{doc.page_content}")
        
        policy_context = "\n\n".join(policy_chunks_formatted)
        
        # Add clause information if available
        clause_info = ""
        if self.policy_clauses:
            clause_entries = [f"{num}: {text}" for num, text in self.policy_clauses.items()]
            clause_info = "POLICY CLAUSES:\n" + "\n".join(clause_entries[:20])  # Limit to top 20 clauses
        
        # Step 4: Build an enhanced prompt with better structure and constraints
        logging.info("Building enhanced prompt for compliance analysis")
        prompt = f"""You are a cybersecurity audit assistant with expertise in compliance analysis.
        
        TASK:
        Analyze the provided log events against the cybersecurity policies and identify instances of policy violations.
        
        LOG EVENTS (from network capture or log file):
        --------------------
        {log_context}
        --------------------
        
        RELEVANT POLICY SECTIONS:
        --------------------
        {policy_context}
        --------------------
        
        {clause_info if clause_info else ""}
        
        INSTRUCTIONS:
        1. Carefully analyze each log event against the policy requirements.
        2. Identify the top {top_k} most significant policy violations.
        3. For each violation, provide:
           a) The specific policy clause number that is violated (ONLY use clause numbers that exist in the provided policy text)
           b) A clear explanation of how the log event violates this policy
           c) The exact log snippet that constitutes evidence of the violation
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        ```
        VIOLATION 1:
        Policy Clause: [clause number]
        Explanation: [concise explanation of the violation]
        Evidence: [exact log snippet showing the violation]
        
        VIOLATION 2:
        ...
        ```
        
        IMPORTANT CONSTRAINTS:
        - Only reference policy clause numbers that explicitly appear in the provided policy text
        - If you cannot find exactly {top_k} violations, report only those you can confidently identify
        - Do not fabricate or assume information not present in the logs
        - If no clear violations are found, state this explicitly rather than forcing matches
        - Provide direct quotes from the logs as evidence, not paraphrased content
        """
        
        # Step 5: Use temperature setting to reduce randomness and hallucination
        try:
            # Clear GPU cache before LLM invocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logging.info("Invoking LLM for compliance analysis...")
            st.info("Generating compliance analysis... This may take a moment.")
            
            # Add progress indicator and timeout handling
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Preparing analysis...")
                progress_bar.progress(0.2)
                
                progress_bar.progress(0.5)
                
                # Invoke LLM with optimized prompt to avoid hanging
                # Keep the format instructions but limit the data context
                optimized_prompt = f"""You are a cybersecurity compliance analyst. Analyze the provided log data against the given policies and identify exactly 3 policy violations.

LOG DATA:
{log_context[:800]}

POLICIES:
{policy_context[:600]}

For each violation, provide:
a) The specific policy clause number being violated
b) A clear explanation of how the log event violates this policy
c) The exact log snippet that constitutes evidence of the violation

FORMAT YOUR RESPONSE AS FOLLOWS:
```
VIOLATION 1:
Policy Clause: [clause number]
Explanation: [concise explanation of the violation]
Evidence: [exact log snippet showing the violation]

VIOLATION 2:
Policy Clause: [clause number]
Explanation: [concise explanation of the violation]
Evidence: [exact log snippet showing the violation]

VIOLATION 3:
Policy Clause: [clause number]
Explanation: [concise explanation of the violation]
Evidence: [exact log snippet showing the violation]
```

IMPORTANT CONSTRAINTS:
- Only reference policy clause numbers that explicitly appear in the provided policy text
- Provide direct quotes from the logs as evidence, not paraphrased content
- Focus on the most critical security violations
"""
                
                response = self.llm.invoke(
                    input=optimized_prompt,
                    temperature=0.1
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis completed!")
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                logging.info("LLM invocation completed successfully")
                
            except Exception as llm_error:
                progress_bar.empty()
                status_text.empty()
                
                st.warning("⚠️ Lily Cybersecurity LLM encountered an issue, trying fallback...")
                
                # Fallback to a simpler analysis with proper format
                simple_prompt = f"""Analyze these log events for cybersecurity policy violations:

LOG DATA:
{log_context[:800]}

POLICIES:
{policy_context[:400]}

Provide exactly 3 security violations in this format:

VIOLATION 1:
Policy Clause: [clause number or policy name]
Explanation: [brief explanation]
Evidence: [log snippet]

VIOLATION 2:
Policy Clause: [clause number or policy name]
Explanation: [brief explanation]
Evidence: [log snippet]

VIOLATION 3:
Policy Clause: [clause number or policy name]
Explanation: [brief explanation]
Evidence: [log snippet]"""
                
                try:
                    response = self.llm.invoke(input=simple_prompt, temperature=0.1)
                    st.info("✅ Fallback analysis completed")
                except Exception as fallback_error:
                    raise Exception(f"Both main and fallback analysis failed: {str(llm_error)}, {str(fallback_error)}")
        except Exception as e:
            error_msg = f"Failed to invoke LLM: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)  # Show error in Streamlit UI
            response = f"Failed to analyze compliance due to an error: {str(e)}"
        
        return response