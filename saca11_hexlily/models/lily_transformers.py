"""
Lily Cybersecurity LLM using Transformers with 4-bit quantization
Based on working notebook implementation
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import subprocess
import sys


def install_bitsandbytes():
    """Install or upgrade bitsandbytes to latest version"""
    try:
        st.info("🔄 Installing latest bitsandbytes...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "bitsandbytes"])
        st.success("✅ bitsandbytes upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Failed to install bitsandbytes: {e}")
        return False


def check_bitsandbytes():
    """Check if bitsandbytes is available"""
    try:
        import bitsandbytes as bnb
        st.success("✅ bitsandbytes is available")
        return True
    except ImportError:
        st.warning("⚠️ bitsandbytes not found, installing...")
        if install_bitsandbytes():
            try:
                import bitsandbytes as bnb
                st.success("✅ bitsandbytes installed successfully")
                return True
            except ImportError:
                st.error("❌ Failed to import bitsandbytes after installation")
                return False
        else:
            st.error("❌ Failed to install bitsandbytes")
            return False


class LilyTransformersLLM:
    """Lily Cybersecurity LLM using Transformers with 4-bit quantization"""

    def __init__(self, model_name="segolilylabs/Lily-Cybersecurity-7B-v0.2", display_name="Lily Cybersecurity LLM"):
        self.model_name = model_name
        self.display_name = display_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the Lily Cybersecurity model with 4-bit quantization and fallback options"""
        try:
            # Clear GPU cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.info(f"🧹 Cleared GPU cache. Available memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            
            # Check bitsandbytes availability
            if not check_bitsandbytes():
                return False
            
            st.info(f"🔄 Loading Lily Cybersecurity model: {self.model_name}")
            st.info("📥 This will download ~7GB model on first run...")
            
            # Try GPU first with 4-bit quantization
            try:
                return self._load_with_gpu_quantization()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as gpu_error:
                if "CUDA out of memory" in str(gpu_error) or "out of memory" in str(gpu_error):
                    st.warning("⚠️ GPU out of memory, trying CPU fallback...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return self._load_with_cpu_fallback()
                else:
                    raise gpu_error
                    
        except Exception as e:
            error_str = str(e)
            st.error(f"❌ Error loading Lily Cybersecurity model: {error_str}")
            
            # Check if it's a bitsandbytes version issue
            if "latest version of bitsandbytes" in error_str:
                st.warning("🔄 Upgrading bitsandbytes to latest version...")
                if install_bitsandbytes():
                    return self.load_model()  # Retry after upgrade
            
            st.info("💡 Make sure you have sufficient GPU memory or the model will use CPU")
            return False
    
    def _load_with_gpu_quantization(self):
        """Try to load model with GPU and 4-bit quantization"""
        # Define the configuration for 4-bit loading
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load the tokenizer
        st.info("🔄 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the model with quantization
        st.info("🔄 Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatically use GPU if available
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        return self._create_pipeline("GPU with 4-bit quantization")
    
    def _load_with_cpu_fallback(self):
        """Fallback to CPU loading without quantization"""
        st.info("🔄 Loading model on CPU (no quantization)...")
        
        # Load the tokenizer if not already loaded
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            st.info("🔄 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the model on CPU without quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        return self._create_pipeline("CPU")
    
    def _create_pipeline(self, device_info):
        """Create the text generation pipeline"""
        # Create pipeline
        st.info("🔄 Creating text generation pipeline...")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,  # Increased for complete compliance analysis
            temperature=0.01,  # Low temperature for factual outputs
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        st.success(f"✅ Lily Cybersecurity model loaded successfully on {device_info}!")
        st.info("🎯 Model specialized for cybersecurity analysis")
        return True
            
    def invoke(self, input, temperature=0.01):
        """Generate response using Lily Cybersecurity model with timeout handling"""
        if not self.pipeline:
            if not self.load_model():
                return "Error: Failed to load Lily Cybersecurity model"

        prompt = input if isinstance(input, str) else str(input)
        
        # Optimize prompt length for better performance
        if len(prompt) > 2500:
            # Try to find format instructions to preserve them
            format_start = prompt.find("FORMAT YOUR RESPONSE")
            if format_start != -1:
                # Keep beginning + format instructions
                data_section = prompt[:format_start-100]  # Keep some context before format
                format_section = prompt[format_start:]
                if len(data_section) > 1200:
                    data_section = data_section[:1200] + "...\n\n"
                prompt = data_section + format_section
            else:
                prompt = prompt[:2000] + "..."
            st.info("📝 Prompt optimized for better performance")
        
        try:
            st.info(f"🤖 {self.display_name} is analyzing...")
            
            # Generate response with parameters for complete analysis
            outputs = self.pipeline(
                prompt,
                max_new_tokens=400,  # Increased for complete compliance analysis
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                truncation=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True  # Stop early if possible
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Clean up the response
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
                st.warning("⚠️ GPU memory issue during generation, trying CPU fallback...")
                try:
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Try CPU generation
                    if hasattr(self, 'model'):
                        self.model = self.model.to('cpu')
                        outputs = self.pipeline(
                            prompt,
                            max_new_tokens=200,  # Reduced for CPU
                            temperature=temperature,
                            do_sample=True if temperature > 0 else False,
                            truncation=True,
                            return_full_text=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        if outputs and len(outputs) > 0:
                            generated_text = outputs[0]['generated_text']
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt):].strip()
                            return generated_text
                        
                except Exception as cpu_error:
                    st.error(f"CPU fallback also failed: {str(cpu_error)}")
                    return f"Error: Both GPU and CPU generation failed. GPU: {error_str}, CPU: {str(cpu_error)}"
            
            return f"Error in Lily LLM generation: {error_str}"


def load_lily_transformers_model():
    """Load and validate Lily Transformers model"""
    try:
        llm = LilyTransformersLLM()

        if llm.load_model():
            st.success("🎯 Lily Cybersecurity LLM (Transformers) loaded successfully!")
            st.info("🔒 This model is specifically trained for cybersecurity tasks")
            return llm
        else:
            st.error("❌ Failed to load Lily Cybersecurity LLM")
            return None

    except Exception as e:
        st.error(f"❌ Error loading Lily Transformers model: {str(e)}")
        return None


def load_lily_transformers_model():
    """Load and validate Lily Transformers model"""
    try:
        llm = LilyTransformersLLM()
        
        if llm.load_model():
            st.success("🎯 Lily Cybersecurity LLM (Transformers) loaded successfully!")
            st.info("🔒 This model is specifically trained for cybersecurity tasks")
            return llm
        else:
            st.error("❌ Failed to load Lily Cybersecurity LLM")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading Lily Transformers model: {str(e)}")
        return None
                    
        st.error(f"❌ Error loading Lily Transformers model: {str(e)}")
        return None