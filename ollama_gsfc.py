import requests
import json
import sys

class OllamaGSFC:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.api_base}/tags")
            print(f"Ollama connection test: {response.status_code}")
            print(f"Available models: {response.json()}")
        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
        
        # Load GSFC University data
        try:
            with open('data.json', 'r') as f:
                self.university_data = json.load(f)
            print("Successfully loaded university data")
        except Exception as e:
            print(f"Error loading university data: {str(e)}")
    
    def generate_response(self, prompt, max_tokens=512):
        try:
            # Prepare the API request
            url = f"{self.api_base}/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_tokens
                }
            }
            
            print(f"\nSending request to Ollama:")
            print(f"URL: {url}")
            print(f"Model: {self.model_name}")
            
            # Make the API call
            response = requests.post(url, json=data)
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                response.raise_for_status()
            
            # Extract and return the response
            return response.json()["response"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {str(e)}")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    
    def query_university_info(self, question):
        # Create a context-aware prompt with examples
        context = json.dumps(self.university_data, indent=2)
        prompt = f"""You are a helpful assistant for GSFC University, Vadodara. Use the following context to answer questions accurately and concisely.

Context about GSFC University:
{context}

Here are some examples of how to answer questions:

Question: What programs are offered at GSFC University?
Answer: GSFC University offers various programs:
1. Undergraduate Programs:
   - B.Tech in Chemical, Computer Science, Mechanical, Electrical, and Civil Engineering
   - B.Sc in Chemistry, Physics, Mathematics, and Computer Science
2. Postgraduate Programs:
   - M.Tech in Chemical, Computer Science, and Mechanical Engineering
   - M.Sc in Chemistry, Physics, and Mathematics
3. PhD programs in various departments

Question: Who is the current Vice Chancellor?
Answer: Dr. J. N. Patel is the current Vice Chancellor of GSFC University. He serves as the academic and administrative head of the university.

Question: What are the admission requirements for B.Tech?
Answer: Admission to B.Tech programs at GSFC University is based on GUJCET/JEE Main scores. The university offers B.Tech programs in Chemical, Computer Science, Mechanical, Electrical, and Civil Engineering.

Now, please answer the following question using the same style and format:

Question: {question}

Answer:"""
        
        return self.generate_response(prompt)

def main():
    try:
        # Initialize the Ollama interface
        print("Initializing GSFC University Assistant...")
        gsfc_bot = OllamaGSFC()
        
        print("\nGSFC University Information Assistant")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            question = input("\nHey it's focus-1, how can I help you today?: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
                
            print("\nThinking...")
            response = gsfc_bot.query_university_info(question)
            
            if response:
                print("\nResponse:", response)
            else:
                print("\nSorry, I couldn't generate a response. Please try again.")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
