from flask import Flask, request, jsonify, Response
import requests
import os
import time

app = Flask(__name__)

# NVIDIA NIM API configuration
NIM_API_KEY = os.environ.get('NIM_API_KEY', '')
NIM_BASE_URL = os.environ.get('NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')

print(f"Server starting... NIM_BASE_URL: {NIM_BASE_URL}")

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        print(f"Received request from Janitor AI")
        
        # Extract parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-70b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False  # Force non-streaming for compatibility
        }
        
        headers = {
            'Authorization': f'Bearer {NIM_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        print(f"Calling NVIDIA with model: {model}")
        
        # Make request to NVIDIA NIM
        nim_response = requests.post(
            f'{NIM_BASE_URL}/chat/completions',
            headers=headers,
            json=nim_payload,
            timeout=120
        )
        
        print(f"NVIDIA Status: {nim_response.status_code}")
        
        if nim_response.status_code == 200:
            response_data = nim_response.json()
            
            # Ensure the response has all required OpenAI fields
            if 'choices' in response_data and len(response_data['choices']) > 0:
                # Make sure message content exists
                if 'message' in response_data['choices'][0]:
                    content = response_data['choices'][0]['message'].get('content', '')
                    print(f"Response content length: {len(content)}")
                    
                    # Ensure proper OpenAI format
                    formatted_response = {
                        'id': response_data.get('id', f'chatcmpl-{int(time.time())}'),
                        'object': 'chat.completion',
                        'created': response_data.get('created', int(time.time())),
                        'model': response_data.get('model', model),
                        'choices': [{
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': content
                            },
                            'finish_reason': response_data['choices'][0].get('finish_reason', 'stop')
                        }],
                        'usage': response_data.get('usage', {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0
                        })
                    }
                    
                    return jsonify(formatted_response), 200
            
            # If we get here, return the raw response
            print("Returning raw NVIDIA response")
            return jsonify(response_data), 200
        else:
            error_msg = f"NVIDIA API error: {nim_response.status_code}"
            print(error_msg)
            return jsonify({'error': error_msg, 'details': nim_response.text}), nim_response.status_code
            
    except requests.exceptions.Timeout:
        print("Request timeout")
        return jsonify({'error': 'Request timeout'}), 504
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return jsonify({'error': f'Request error: {str(e)}'}), 502
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
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
