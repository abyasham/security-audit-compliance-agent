# Security Audit Compliance Agent

In today's digital landscape, it's no secret that companies face many compliance requirements. Certain expectations from a business are no longer up for negotiation, such as maintaining integrity, availability, data confidentiality and information security and ensuring ethical practices, especially considering the common application of AI. Compliance audits ensure these expectations are met and serve as crucial checkpoints to assess a company's adherence to specific regulatory frameworks.

AI-powered systems can capture and analyse audit trails on autopilot and provide a chronological record of activities, mitigating the risk of errors or omissions. In addition, auditors can also leverage AI to generate comprehensive prediction compliance reports, which analyse data from multiple sources and evaluate them against the critical compliance metrics. By doing so, auditors can save time and simultaneously benefit from abilities they didn’t have until now.

Ultimately, AI cannot and should not replace the human element that is critically important in the auditing process.

AI provides auditors with a chance to elevate their proficiency and knowledge. To stay in the game, auditors must enhance their skills and fuse their technical and interpersonal abilities. Think of it as a culmination of data analysis, critical thinking, and adept communication.

SACA (Security Audit Compliance Agent) is an AI-based agent to help auditors conduct cybersecurity audits while maintaining security & privacy

# SACA 1
First attempt on SACA, a simple RAG (Retrieval-Augmented Generation) system using Deepseek, LangChain, and Streamlit to chat with PDFs and answer complex questions about your local documents.

* Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

ollama pull deepseek-r1:14b

Install the dependencies using pip:

pip install -r requirements.txt

Security audit compliance analysis
PCAP file processing and analysis
Policy document comparison
Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
Interactive chat interface
Vector database for document storage
* Run
Run the Streamlit app:

streamlit run pdf_rag.py

# SACA 2
Second attempt on SACA, focusing on better embedding models to chat with documents and manage them in your local database/repository.

* Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the
Security audit compliance analysis
PCAP file processing and analysis
Policy document comparison
Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
Interactive chat interface
Vector database for document storage, Deepseek model:

ollama pull deepseek-r1:14b

Install the dependencies using pip:

pip install -r requirements.txt

* Run
Make sure to choose your preferred embedding model in get_embedding_function.py (Amazon Bedrock (require API) or local Ollama (e.g. mxbai-embed-large, require pull procedure from ollama.com)
Run the populate_database.py to convert all files in the folder 'data' into a vector and store in Chromadb:

python run populate_database.py
Run rag_deepseek.py to chat with your documents :

python run rag_deepseek.py

# SACA 3
Next attempt on SACA, adding several different file types for RAG knowledge base, including JSON file type as the security criterion/standard needed to conduct a compliance audit.

* Run
Run the upload.py to convert your selected type of files (pdf, csv, json, txt) into vector and store in the database :

python run upload.py
Run localrag.py to chat with your documents :

Security audit compliance analysis
PCAP file processing and analysis
Policy document comparison
Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
Interactive chat interface
Vector database for document storage
python run localrag.py
Security audit compliance analysis
PCAP file processing and analysis
Policy document comparison
Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
Interactive chat interface
Vector database for document storage

# SACA 11 (HexLily)
The newest development for SACA is [SACA11_HexLily](https://github.com/abyasham/security-audit-compliance-agent/tree/main/saca11_hexlily), which has utilised PCAP Hexdump/Binary parsing for embeddings, a customised retrieval technique and the option to use [Lily-Cybersecurity] LLM (https://huggingface.co/segolilylabs/Lily-Cybersecurity-7B-v0.2) besides the Deepseek LLM. It features the following:
- Security audit compliance analysis
- PCAP file processing and analysis
- Policy document comparison
- Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
- Interactive chat interface
- Vector database for document storage

# n8n workflows
Contain JSON schema files that you can paste into your n8n self-hosted machine (https://github.com/n8n-io/self-hosted-ai-starter-kit) and test the rag_seccompliance workflow

# ollama webui
Contain JSON schema files that you can paste into your Ollama web running on your local web UI (check out this cool video on that https://www.youtube.com/watch?v=DYhC7nFRL5I) and test the saca_deepseek or saca_llama workflows.

# Gotham use case
A bunch of documents that showcase Gotham Ltd as the auditee company, complete with dummy policy and network traffic logs.

# knowledge base
Basically contains needed documents that will be added to the RAG system to give context for the LLM model as a security compliance audit assistant. In this case, a security compliance audit for SOC 2 Type 2 attestation focuses on its mandatory controls, specifically the Trust Service Criteria (TSC).
