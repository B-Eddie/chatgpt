from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize the DialoGPT-large model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Initialize chat history and conversation history
chat_history_ids = torch.tensor([]).to(device)
history = []

# In-memory user database (for demonstration purposes)
users = {}

def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def generate_reply(prompt):
    """Generates a conversational reply based on the user's input and conversation history."""
    global chat_history_ids, history
    try:
        # Encode the new user input and add the eos_token
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Append the new user input tokens to the chat history
        if chat_history_ids.size(0) > 0:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the generated response
        generated_reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Update chat history and conversation history
        chat_history_ids = chat_history_ids[:, -1000:]  # Keep only the last 1000 tokens
        history.append(f"User: {prompt}")
        history.append(f"AI: {generated_reply}")
        
        return generated_reply
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in generate_reply: {e}")
        return "Sorry, I couldn't process that request."

@app.route('/')
@authenticate
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return "Username already exists", 400
        users[username] = generate_password_hash(password)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    ai_reply = generate_reply(user_input)
    if ai_reply:
        return jsonify({'reply': ai_reply, 'history': history})
    else:
        return jsonify({'reply': "Failed to generate a reply."})

@app.route('/export')
@authenticate
def export_chat():
    try:
        # Ensure the chat history file exists and write it
        file_path = 'chat_history.txt'
        with open(file_path, 'w') as f:
            for line in history:
                f.write(f"{line}\n")
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error exporting chat history: {e}")
        return "Failed to export chat history", 500

if __name__ == '__main__':
    app.run(debug=True)
