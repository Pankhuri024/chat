from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/send-text', methods=['POST'])
def receive_text():
    data = request.get_json()
    received_text = data.get('text', '')  # Get 'text' field from JSON data
    return jsonify({'received_text': received_text, 'message': 'Text received successfully'})

if __name__ == '__main__':
    app.run(debug=True)
