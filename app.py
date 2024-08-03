from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class DistilChatGPT:
    def __init__(self):
        # Load pre-trained model and tokenizer for distilgpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.model.eval()  # Set the model to evaluation mode

    def generate_response(self, prompt):
        # Encode the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=2,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat(self):
        print("DistilGPT2 (local). Type 'exit' to end the chat.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                response = self.generate_response(user_input)
                print("ChatGPT: " + response)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    chatgpt = DistilChatGPT()
    chatgpt.chat()
