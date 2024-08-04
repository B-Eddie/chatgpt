from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Initialize the DialoGPT-large model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

history = []

def generate_reply(prompt, history):
    """Generates a conversational reply based on the user's input and conversation history."""
    try:
        # Add the new user prompt to the conversation history
        conversation = "\n".join(history) + f"\nUser: {prompt}\nAI:"
        
        # Tokenize the conversation
        inputs = tokenizer.encode(conversation + tokenizer.eos_token, return_tensors='pt')
        
        # Generate a reply
        response = model.generate(
            inputs,
            max_length=1500,  # Adjust max_length as needed
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id  # Ensure EOS token is used
        )
        
        # Extract generated reply
        generated_reply = tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        
        return generated_reply
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in generate_reply: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    history.append(f"User: {user_input}")
    ai_reply = generate_reply(user_input, history)
    if ai_reply:
        history.append(f"AI: {ai_reply}")
        print(ai_reply)
        return jsonify({'reply': ai_reply})
    else:
        return jsonify({'reply': "Failed to generate a reply."})

if __name__ == '__main__':
    app.run(debug=True)
