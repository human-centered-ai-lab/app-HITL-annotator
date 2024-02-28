import json
import numpy as np
from flask import Flask, request
import server as server
from utils import is_authorized

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return 'Hello, I am a client', 200

# get the current status of the client
@app.route('/server/status', methods=['GET'])
def get_client_status():
    headers = request.headers
    authorized = is_authorized(headers, server.get_instance().verify_server_token)
    if not authorized:
        return json.dumps({ 'error': 'not authorized' }), 401
    status = server.get_instance().get_status()
    return json.dumps({'status': status}), 200

#reset this client
@app.route('/server/reset', methods=['DELETE'])
def reset_this_client():
    headers = request.headers
    authorized = is_authorized(headers, server.get_instance().verify_server_token)
    if not authorized:
        return json.dumps({ 'error': 'not authorized' }), 401
    server.get_instance().reset_server()
    status = server.get_instance().get_status()
    return json.dumps({'status': status}), 200

# endpoint to receive message from server
# intent (subject) of the message is in X-Intent header
@app.route('/server/message', methods=['POST'])
def receive_server_message():
    headers = request.headers
    authorized = is_authorized(headers, server.get_instance().verify_server_token)
    if not authorized:
        print("[CLIENT     ] received unauthorized message")
        return json.dumps({ 'error': 'not authorized' }), 401
    intent = None
    try:
        intent = headers['X-Intent']
        if not intent: raise "No intent provided"
    except:
        print(f"[CLIENT     ] received message from server without intent")
        return json.dumps({'error': 'no valid intent provided'}), 400
    print(f"[CLIENT     ] received message from server with intent {intent}")
    if intent == 'DEBUG':
        print(f"[CLIENT     ] message: {request.get_data()}")
        return json.dumps({'status': 'ok'}), 200
    print(f"[CLIENT     ] received message from server without valid intent")
    return json.dumps({'error': 'invalid intent'}), 400

if __name__ == "__main__":
    try:   
        server.get_instance().connect('http://localhost:5000')
        server.get_instance().send_message_to_server('DEBUG', 'Hello')
    except:
        print('not connected to server')

    app.run(debug=True, port=5001)
