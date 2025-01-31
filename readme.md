# Security Audit Compliance Agent

In today's digital landscape, it's no secret that companies face many compliance requirements. Certain expectations from a business are no longer up for negotiation, such as maintaining integrity, availability, data confidentiality and information security and ensuring ethical practices, especially considering the common application of AI. Compliance audits ensure these expectations are met and serve as crucial checkpoints to assess a company's adherence to specific regulatory frameworks.

AI-powered systems can capture and analyze audit trails on auto-pilot and provide a chronological record of activities, mitigating the risk of errors or omissions. In addition, auditors can also leverage AI to generate comprehensive prediction compliance reports, which analyze data from multiple sources and evaluate them against the critical compliance metrics. By doing so, auditors can save time and simultaneously benefit from abilities they didn’t have until now.

Ultimately, AI cannot and should not replace the human element that is critically important in the auditing process.

AI provides auditors with a chance to elevate their proficiency and knowledge. To stay in the game, auditors must enhance their skills and fuse their technical and interpersonal abilities. Think of it as a culmination of data analysis, critical thinking, and adept communication.

SACA (Security Audit Compliance Agent) is an AI-based agent to help auditors conduct cybersecurity audit while maintaining security & privacy

# SACA 1
First attempt on SACA, a simple RAG (Retrieval-Augmented Generation) system using Deepseek, LangChain, and Streamlit to chat with PDFs and answer complex questions about your local documents.

* Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

ollama pull deepseek-r1:14b

Install the dependencies using pip:

pip install -r requirements.txt

* Run
Run the Streamlit app:

streamlit run pdf_rag.py

# SACA 2
Second attempt on SACA, focusing on better enbedding models to chat with documents and manage it in your local database/repository.

* Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

ollama pull deepseek-r1:14b

Install the dependencies using pip:

pip install -r requirements.txt

* Run
Make sure to choose your preferred embedding model in get_embedding_function.py (amazon bedrock (require API) or local ollama (e.g. mxbai-embed-large, require pull procedure from ollama.com)
Run the populate_database.py to convert all files in folder 'data' into vector and store in chromadb :

python run populate_database.py
Run rag_deepseek.py to chat with your documents :

python run rag_deepseek.py

# SACA 3
Next attempt on SACA, adding several different file type for RAG knowledge base, including json file type as the security critera/standard needed to conduct a compliance audit.

* Run
Run the upload.py to convert your selected type files (pdf, csv, json, txt) into vector and store in database :

python run upload.py
Run localrag.py to chat with your documents :

python run localrag.py

* SACA 4
--under development

# n8n workflows
Contain json schema files that you can paste into your n8n self hosted machine (https://github.com/n8n-io/self-hosted-ai-starter-kit) and test the rag_seccompliance workflow

# ollama webui
Contain json schema files that you can paste into your ollama web running on your local webui (check out this cool video on that https://www.youtube.com/watch?v=DYhC7nFRL5I) and test the saca_deepseek or saca_llama workflows.

# Gotham use case
Bunch of documents that showcase Gotham Ltd as the auditee company complete with dummy policy and network traffic logs.

# knowledge base
Basically contain needed documents that will be addedd to the RAG system to give context for the LLM model as a security compliance audit assistant. In this case a security compliance audit for SOC 2 type 2 attestation which focus on its mandatory controls, the Trust Service Criteria (TSC).
