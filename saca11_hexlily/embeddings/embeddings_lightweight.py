"""
Lightweight embeddings for SACA
Uses sentence-transformers for faster initialization
"""

import logging
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List


class LightweightEmbeddings:
    """Lightweight embedding model using sentence-transformers"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize with a lightweight, fast-loading model
        all-MiniLM-L6-v2 is small (~90MB) and fast
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            st.info(f"🔄 Loading lightweight embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            st.success("✅ Lightweight embedding model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {str(e)}")
            logging.error(f"Failed to load embedding model: {str(e)}")
            raise e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error embedding documents: {str(e)}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logging.error(f"Error embedding query: {str(e)}")
            raise e


class ChromaEmbeddingFunctionWrapperLightweight:
    """Wrapper to make LightweightEmbeddings compatible with ChromaDB"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def __call__(self, input_texts):
        """ChromaDB calls this method to get embeddings"""
        try:
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            
            embeddings = self.embedding_model.embed_documents(input_texts)
            return embeddings
        except Exception as e:
            logging.error(f"Error in ChromaDB embedding wrapper: {str(e)}")
            raise e
    
    def name(self):
        """Return the name of the embedding function"""
        return f"lightweight_embeddings_{self.embedding_model.model_name}"