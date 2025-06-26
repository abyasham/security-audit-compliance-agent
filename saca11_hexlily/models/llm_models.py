"""
LLM Models module for SACA
Contains Ollama and Hugging Face model implementations
"""

import requests
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st


class OllamaLLM:
    """Ollama LLM wrapper for local model inference"""
    
    def __init__(self, model_name="deepseek-r1:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def invoke(self, input, temperature=0.1):
        """
        Invoke Ollama model for text generation
        """
        prompt = input if isinstance(input, str) else str(input)
        
        # Show status message
        st.info(f"🤖 Ollama ({self.model_name}) is analyzing...")
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated.")
            else:
                return f"Ollama API error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            return f"Error in Ollama generation: {str(e)}"


class HuggingFacePipelineLLM:
    """Hugging Face pipeline wrapper with memory optimization"""
    
    def __init__(self, pipeline, tokenizer):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
    
    def invoke(self, input, temperature=0.0):
        """Generate text using Hugging Face pipeline with memory management"""
        prompt = input if isinstance(input, str) else str(input)
        
        # Show status message
        st.info("🤖 Hugging Face Transformers is analyzing...")
        
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adjust generation parameters as needed
            outputs = self.pipeline(
                prompt,
                max_new_tokens=256, # Reduced from 512 to save memory
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else 0.1,  # Avoid 0 temperature
                top_p=0.9, # Nucleus sampling
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id, # Use eos_token_id for pad_token_id
                truncation=True,  # Truncate if input is too long
                return_full_text=False  # Only return generated text, not input
            )
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Remove the input prompt from the output if it's included
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                return "No response generated."
                
        except Exception as e:
            error_str = str(e)
            # Handle various CUDA errors including device-side assert
            if any(cuda_error in error_str.lower() for cuda_error in [
                "cuda out of memory", "out of memory", "cuda error", "device-side assert",
                "cuda kernel", "cuda runtime error", "device assert"
            ]):
                print(f"GPU out of memory during generation, trying CPU fallback...")
                try:
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Move model to CPU for this generation
                    device_before = next(self.pipeline.model.parameters()).device
                    self.pipeline.model = self.pipeline.model.cpu()
                    
                    # Try generation on CPU with smaller parameters
                    outputs = self.pipeline(
                        prompt,
                        max_new_tokens=128, # Even smaller for CPU
                        do_sample=False,  # Greedy decoding for CPU
                        truncation=True,
                        return_full_text=False
                    )
                    
                    # Move model back to original device
                    self.pipeline.model = self.pipeline.model.to(device_before)
                    
                    if outputs and len(outputs) > 0:
                        generated_text = outputs[0]['generated_text']
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        return f"[CPU Generated] {generated_text}"
                    else:
                        return "No response generated on CPU fallback."
                        
                except Exception as cpu_error:
                    return f"Error in both GPU and CPU generation: GPU: {error_str}, CPU: {str(cpu_error)}"
            else:
                print(f"Error in LLM generation: {error_str}")
                return f"Error generating response: {error_str}"


def load_ollama_model(model_name):
    """Load and validate Ollama model"""
    try:
        # Test Ollama connection
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            available_models = [model["name"] for model in test_response.json().get("models", [])]
            if model_name in available_models:
                llm = OllamaLLM(model_name=model_name)
                st.success(f"Successfully connected to Ollama model: {model_name}")
                return llm
            else:
                st.error(f"Model {model_name} not found in Ollama. Available models: {available_models}")
                st.info(f"To install the model, run: ollama pull {model_name}")
                return None
        else:
            st.error("Cannot connect to Ollama. Make sure Ollama is running.")
            st.info("Start Ollama with: ollama serve")
            return None
    except requests.exceptions.RequestException:
        st.error("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        st.info("Start Ollama with: ollama serve")
        return None


def load_huggingface_model(model_name):
    """Load Hugging Face model with memory optimization and fallbacks"""
    try:
        # Clear GPU cache before loading the large model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to reduce memory usage
            device_map="auto",  # Automatically distribute across available devices
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            trust_remote_code=True,  # Allow custom model code
            load_in_8bit=False,  # Set to True if you have bitsandbytes installed
        )
        
        # Create a text generation pipeline with memory optimizations
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_length=512,  # Limit max length to save memory
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return HuggingFacePipelineLLM(llm_pipeline, tokenizer)

    except Exception as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            st.warning(f"GPU out of memory with {model_name}. Trying fallback options...")
            return _try_fallback_models()
        else:
            st.error(f"Failed to load Hugging Face model: {str(e)}")
            st.info("Please ensure the model name is correct and you have sufficient resources.")
            return None


def _try_fallback_models():
    """Try smaller fallback models if main model fails"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    fallback_models = [
        "microsoft/DialoGPT-medium",  # Smaller conversational model
        "microsoft/DialoGPT-small",   # Even smaller model
        "gpt2"  # Smallest fallback
    ]
    
    for fallback_model in fallback_models:
        try:
            st.info(f"Trying fallback model: {fallback_model}")
            
            # Clear cache again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_length=256,  # Even smaller max length
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipelineLLM(llm_pipeline, tokenizer)
            st.success(f"Successfully loaded fallback model: {fallback_model}")
            return llm
            
        except Exception as fallback_e:
            st.warning(f"Fallback model {fallback_model} also failed: {str(fallback_e)}")
            continue
    
    st.error("All models failed to load. Please try running on CPU or with more GPU memory.")
    return None
def initialize_llm_model(model_type):
    """
    Initialize LLM model based on type selection
    
    Args:
        model_type: String indicating model type
    
    Returns:
        Initialized LLM model or None if failed
    """
    if "Ollama" in model_type:
        return load_ollama_model("deepseek-r1:8b")
    elif "Hugging Face" in model_type:
        return load_huggingface_model("microsoft/DialoGPT-medium")
    elif "Lily Cybersecurity" in model_type:
        from .lily_transformers import load_lily_transformers_model
        return load_lily_transformers_model()
    else:
        st.error(f"Unknown model type: {model_type}")
        return None