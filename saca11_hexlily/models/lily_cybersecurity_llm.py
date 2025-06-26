"""
Lily Cybersecurity LLM integration for SACA
Uses llama.cpp server for efficient and stable inference
"""

import requests
import streamlit as st
import subprocess
import time
import os


class LilyCybersecurityLLM:
    """Lily Cybersecurity LLM wrapper using llama.cpp server"""
    
    def __init__(self, base_url="http://localhost:8080", model_name="Lily-Cybersecurity-7B-v0.2"):
        self.base_url = base_url
        self.model_name = model_name
        self.server_process = None
        
    def start_server(self):
        """Start the llama.cpp server with Lily Cybersecurity model"""
        try:
            # Check if server is already running
            if self.is_server_running(show_status=False):
                st.info("🟢 Lily Cybersecurity LLM server is already running")
                return True
            
            st.info("🚀 Starting Lily Cybersecurity LLM server...")
            st.info("📥 This may download the model on first run (~4GB)")
            
            # Try to start the server
            cmd = [
                "llama-server", 
                "-hf", "segolilylabs/Lily-Cybersecurity-7B-v0.2-GGUF:Q4_K_M",
                "--port", "8080",
                "--host", "localhost"
            ]
            
            # Start server in background
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 60  # 60 seconds timeout
            wait_time = 0
            
            while wait_time < max_wait:
                if self.is_server_running(show_status=False):
                    st.success("✅ Lily Cybersecurity LLM server started successfully!")
                    return True
                time.sleep(2)
                wait_time += 2
                st.info(f"⏳ Waiting for server to start... ({wait_time}s)")
            
            st.error("❌ Server failed to start within timeout")
            return False
            
        except FileNotFoundError:
            st.error("❌ llama-server not found. Please install llama.cpp:")
            st.code("brew install llama.cpp  # macOS")
            st.code("winget install llama.cpp  # Windows")
            return False
        except Exception as e:
            st.error(f"❌ Error starting server: {str(e)}")
            return False
    
    def is_server_running(self, show_status=True):
        """Check if the llama.cpp server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def invoke(self, input, temperature=0.1):
        """
        Generate response using Lily Cybersecurity LLM
        
        Args:
            input: The prompt/input text
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated response text
        """
        prompt = input if isinstance(input, str) else str(input)
        
        # Ensure server is running
        if not self.is_server_running(show_status=False):
            if not self.start_server():
                return "Error: Could not start Lily Cybersecurity LLM server"
        
        try:
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": 512,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "\n\n"],
                "stream": False
            }
            
            # Make request to llama.cpp server
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("content", "No response generated.")
                return generated_text.strip()
            else:
                return f"Lily LLM API error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Lily LLM: {str(e)}"
        except Exception as e:
            return f"Error in Lily LLM generation: {str(e)}"
    
    def stop_server(self):
        """Stop the llama.cpp server"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                st.info("🛑 Lily Cybersecurity LLM server stopped")
            except:
                self.server_process.kill()
                st.warning("⚠️ Lily Cybersecurity LLM server force killed")


def load_lily_cybersecurity_model():
    """Load and validate Lily Cybersecurity LLM"""
    try:
        llm = LilyCybersecurityLLM()
        
        # Try to start the server
        if llm.start_server():
            st.success("🎯 Successfully connected to Lily Cybersecurity LLM!")
            st.info("🔒 This model is specifically trained for cybersecurity tasks")
            return llm
        else:
            st.error("❌ Failed to start Lily Cybersecurity LLM server")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading Lily Cybersecurity LLM: {str(e)}")
        return None


def check_llama_cpp_installation():
    """Check if llama.cpp is installed"""
    try:
        result = subprocess.run(['llama-server', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def install_instructions():
    """Display installation instructions for llama.cpp"""
    st.error("❌ llama.cpp not found")
    st.markdown("### 📥 Installation Instructions:")
    
    st.markdown("#### macOS:")
    st.code("brew install llama.cpp")
    
    st.markdown("#### Windows:")
    st.code("winget install llama.cpp")
    
    st.markdown("#### Linux (Ubuntu/Debian):")
    st.code("""
# Download pre-built binary
wget https://github.com/ggerganov/llama.cpp/releases/latest/download/llama.cpp-linux-x64.tar.gz
tar -xzf llama.cpp-linux-x64.tar.gz
sudo cp llama.cpp-linux-x64/llama-server /usr/local/bin/
    """)
    
    st.markdown("#### Build from Source:")
    st.code("""
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build -j --target llama-server
sudo cp build/bin/llama-server /usr/local/bin/
    """)