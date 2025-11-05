from flask import Flask, request, jsonify, Response
import requests
import os
import json

app = Flask(__name__)

# NVIDIA NIM API configuration
NIM_API_KEY = os.environ.get('NIM_API_KEY', '')
NIM_BASE_URL = os.environ.get('NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-405b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream
        }
        
        headers = {
            'Authorization': f'Bearer {NIM_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Make request to NVIDIA NIM
        nim_response = requests.post(
            f'{NIM_BASE_URL}/chat/completions',
            headers=headers,
            json=nim_payload,
            stream=stream
        )
        
        if stream:
            def generate():
                for line in nim_response.iter_lines():
                    if line:
                        yield line + b'\n'
            
            return Response(generate(), content_type='text/event-stream')
        else:
            return jsonify(nim_response.json()), nim_response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    models = {
        'object': 'list',
        'data': [
            {
                'id': 'meta/llama-3.1-405b-instruct',
                'object': 'model',
                'owned_by': 'nvidia'
            },
            {
                'id': 'meta/llama-3.1-70b-instruct',
                'object': 'model',
                'owned_by': 'nvidia'
            },
            {
                'id': 'mistralai/mixtral-8x7b-instruct-v0.1',
                'object': 'model',
                'owned_by': 'nvidia'
            }
        ]
    }
    return jsonify(models)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
