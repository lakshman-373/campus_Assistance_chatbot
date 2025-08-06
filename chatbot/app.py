from flask import Flask, request, jsonify, render_template
from inference import chatbot_response  # Make sure this works
from pyngrok import ngrok
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"response": "⚠️ No message received."}), 400
    response = chatbot_response(user_input)
    return jsonify({"response": response})

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("🚀 Ngrok URL:", public_url)

# Start Flask
app.run(port=5000)
