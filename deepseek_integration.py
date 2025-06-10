import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class DeepSeekGSFC:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Load GSFC University data
        with open('data.json', 'r') as f:
            self.university_data = json.load(f)
    
    def generate_response(self, prompt, max_length=512):
        # Prepare the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def query_university_info(self, question):
        # Create a context-aware prompt
        context = json.dumps(self.university_data, indent=2)
        prompt = f"""Context about GSFC University:
{context}

Question: {question}

Answer:"""
        
        return self.generate_response(prompt)

def main():
    model_path = "path/to/your/deepseek/model"
    
    try:
        gsfc_bot = DeepSeekGSFC(model_path)
        
        # Example usage
        while True:
            question = input("\nAsk a question about GSFC University (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            response = gsfc_bot.query_university_info(question)
            print("\nResponse:", response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()