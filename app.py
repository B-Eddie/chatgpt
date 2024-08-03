from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "mistralai/Mathstral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def solve_math_problem(problem: str) -> str:
    inputs = tokenizer.encode(problem, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    problem = data.get('problem')
    result = solve_math_problem(problem)
    return jsonify({'solution': result})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
