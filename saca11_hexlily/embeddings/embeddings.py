from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from langchain_community.vectorstores import Chroma

class CustomInstructEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2", use_gpu=False):
        """
        Initialize the custom embeddings with a SentenceTransformer model.

        :param model_name: Name of the Hugging Face model to use
        :param use_gpu: Whether to use GPU if available
        """
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            # Try to load the model on the specified device
            self.model = SentenceTransformer(model_name, device=device)
            # Set smaller batch size for GPU to reduce memory usage
            self.batch_size = 4 if device == "cuda" else 16
            print(f"Successfully loaded {model_name} on {device}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                print(f"GPU out of memory with {model_name}, trying smaller model...")
                # Clear GPU cache and try a smaller model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Fallback to a smaller model
                fallback_model = "all-MiniLM-L12-v2"
                try:
                    self.model = SentenceTransformer(fallback_model, device=device)
                    self.batch_size = 8 if device == "cuda" else 16
                    print(f"Successfully loaded fallback model {fallback_model} on {device}")
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    print("Still out of memory, falling back to CPU...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    device = "cpu"
                    self.model = SentenceTransformer(fallback_model, device=device)
                    self.batch_size = 16
                    print(f"Successfully loaded {fallback_model} on CPU")
            else:
                raise e
        
        self.device = device

    def embed_documents(self, texts):
        """
        Embed a list of documents with task-specific instructions.

        :param texts: List of texts to embed
        :return: List of embeddings
        """
        # Create instruction pairs for cybersecurity context
        instruction_pairs = [
            ["Represent the cybersecurity log or policy for retrieval:", text]
            for text in texts
        ]

        # Process in batches for efficiency
        all_embeddings = []
        for i in range(0, len(instruction_pairs), self.batch_size):
            batch = instruction_pairs[i:i+self.batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            # Convert from tensor to list
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text):
        """
        Embed a single query text with task-specific instruction.

        :param text: Text to embed
        :return: Embedding vector
        """
        # For instructor models, we use task-specific instructions
        instruction_pair = ["Represent the cybersecurity query for retrieving relevant logs and policies:", text]

        embedding = self.model.encode(
            [instruction_pair],
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )

        # Convert from tensor to list
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()

        return embedding[0].tolist()

# New class to wrap CustomInstructEmbeddings for ChromaDB compatibility
class ChromaEmbeddingFunctionWrapper:
    def __init__(self, custom_embeddings_instance):
        self.custom_embeddings = custom_embeddings_instance
        # Required attribute for ChromaDB compatibility
        self._type = "custom_instruct_embeddings"

    def __call__(self, input):
        # This is for ChromaDB's internal EmbeddingFunction interface (for documents)
        return self.custom_embeddings.embed_documents(input)

    def embed_documents(self, texts):
        # This is for LangChain's Chroma wrapper
        return self.custom_embeddings.embed_documents(texts)

    def embed_query(self, text):
        # This is for LangChain's Chroma wrapper
        return self.custom_embeddings.embed_query(text)
    
    def name(self):
        # Required method for ChromaDB compatibility
        return "custom_instruct_embeddings"