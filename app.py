from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    response_data = {'message': 'Hello, this is data from Flask!'}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
