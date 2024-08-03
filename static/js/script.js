async function sendProblem() {
    const inputBox = document.getElementById('input-box');
    const chatBox = document.getElementById('chat-box');
    const problem = inputBox.value;

    // Display user message
    const userMessageElement = document.createElement('div');
    userMessageElement.textContent = 'You: ' + problem;
    chatBox.appendChild(userMessageElement);

    // Send message to backend
    const response = await fetch('/solve', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ problem: problem })
    });
    const data = await response.json();

    // Display model response
    const botMessageElement = document.createElement('div');
    botMessageElement.textContent = 'AI: ' + data.solution;
    chatBox.appendChild(botMessageElement);

    // Clear input
    inputBox.value = '';

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}