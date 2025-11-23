"""
Flask Web Application for Multi-Agent System
"""

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from pathlib import Path
import json

# Import the orchestrator
from orchestrator import MultiAgentOrchestrator

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff',
    'txt', 'md', 'docx', 'xlsx', 'xls', 'csv'
}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the orchestrator (singleton)
orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = MultiAgentOrchestrator(verbose=False)
    return orchestrator


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def get_conversation_history():
    """Get conversation history for current session"""
    session_id = get_session_id()
    if 'conversations' not in session:
        session['conversations'] = []
    return session['conversations']


def add_to_history(query, response, files=None):
    """Add interaction to conversation history"""
    if 'conversations' not in session:
        session['conversations'] = []
    
    session['conversations'].append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'files': files or [],
        'agent': response.get('agent_used', 'unknown')
    })
    
    # Keep only last 50 conversations
    if len(session['conversations']) > 50:
        session['conversations'] = session['conversations'][-50:]
    
    session.modified = True


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """
    Handle query requests
    Accepts: JSON with 'query' field and optional files
    """
    try:
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'No query provided',
                'answer': 'No query provided'
            }), 400
        
        user_query = data['query'].strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty',
                'answer': 'Query cannot be empty'
            }), 400
        
        # Get orchestrator
        orch = get_orchestrator()
        
        # Process query (no files in this endpoint)
        result = orch.process_query(
            query=user_query,
            files=None,
            show_routing=False
        )
        
        # Add to conversation history
        add_to_history(user_query, result)
        
        # Ensure all required fields exist
        response_data = {
            'success': result.get('success', True),
            'answer': result.get('raw_answer', result.get('formatted_answer', 'No answer generated')),
            'formatted_answer': result.get('formatted_answer', result.get('raw_answer', 'No answer generated')),
            'agent_used': result.get('agent_used', 'unknown'),
            'sources': result.get('sources', []),
            'routing': result.get('routing_decision', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"Error in /api/query: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'answer': f'Error: {str(e)}'
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and query
    Accepts: multipart/form-data with 'files' and 'query'
    """
    try:
        print("\n" + "="*60)
        print("FILE UPLOAD REQUEST RECEIVED")
        print("="*60)
        
        # Check if files are present
        print(f"Files in request: {list(request.files.keys())}")
        print(f"Form data: {dict(request.form)}")
        
        if 'files' not in request.files:
            print("ERROR: No 'files' key in request.files")
            return jsonify({
                'success': False,
                'error': 'No files provided',
                'answer': 'No files provided'
            }), 400
        
        # Get query
        user_query = request.form.get('query', '').strip()
        print(f"Query: {user_query}")
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty',
                'answer': 'Query cannot be empty'
            }), 400
        
        # Process uploaded files
        files = request.files.getlist('files')
        print(f"Number of files: {len(files)}")
        
        if not files or files[0].filename == '':
            print("ERROR: No files selected or empty filename")
            return jsonify({
                'success': False,
                'error': 'No files selected',
                'answer': 'No files selected'
            }), 400
        
        # Save files
        saved_files = []
        session_folder = os.path.join(
            app.config['UPLOAD_FOLDER'],
            get_session_id()
        )
        os.makedirs(session_folder, exist_ok=True)
        print(f"Session folder: {session_folder}")
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                filename = f"{datetime.now().timestamp()}_{filename}"
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                saved_files.append(filepath)
                print(f"✅ Saved: {filepath} ({os.path.getsize(filepath)} bytes)")
            else:
                print(f"ERROR: File not allowed: {file.filename}")
                return jsonify({
                    'success': False,
                    'error': f'File type not allowed: {file.filename}',
                    'answer': f'File type not allowed: {file.filename}'
                }), 400
        
        print(f"\nTotal files saved: {len(saved_files)}")
        print(f"Files: {saved_files}")
        
        # Get orchestrator
        print("\nInitializing orchestrator...")
        orch = get_orchestrator()
        
        # Process query with files
        print(f"\nProcessing query with {len(saved_files)} file(s)...")
        print(f"Query: '{user_query}'")
        print(f"Files: {saved_files}")
        
        result = orch.process_query(
            query=user_query,
            files=saved_files,
            show_routing=True  # Enable to see routing decision
        )
        
        print(f"\n📊 Result received:")
        print(f"  - Success: {result.get('success')}")
        print(f"  - Agent: {result.get('agent_used')}")
        print(f"  - Has answer: {bool(result.get('raw_answer') or result.get('formatted_answer'))}")
        if result.get('routing_decision'):
            print(f"  - Routing: {result['routing_decision']}")
        
        # Add to conversation history
        add_to_history(
            user_query,
            result,
            files=[os.path.basename(f) for f in saved_files]
        )
        
        # Ensure all required fields exist
        response_data = {
            'success': result.get('success', True),
            'answer': result.get('raw_answer', result.get('formatted_answer', 'No answer generated')),
            'formatted_answer': result.get('formatted_answer', result.get('raw_answer', 'No answer generated')),
            'agent_used': result.get('agent_used', 'unknown'),
            'sources': result.get('sources', []),
            'routing': result.get('routing_decision', {}),
            'files_processed': [os.path.basename(f) for f in saved_files],
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n✅ Sending response to client")
        print("="*60 + "\n")
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("❌ ERROR IN /api/upload")
        print("="*60)
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'answer': f'Error: {str(e)}'
        }), 500


@app.route('/api/history', methods=['GET'])
def history():
    """Get conversation history for current session"""
    try:
        conversations = get_conversation_history()
        return jsonify({
            'success': True,
            'history': conversations,
            'count': len(conversations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        session['conversations'] = []
        session.modified = True
        return jsonify({
            'success': True,
            'message': 'History cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        orch = get_orchestrator()
        system_status = orch.get_system_status()
        
        return jsonify({
            'success': True,
            'status': system_status,
            'session_id': get_session_id(),
            'conversation_count': len(get_conversation_history())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test', methods=['GET'])
def test():
    """Run test cases"""
    try:
        orch = get_orchestrator()
        
        test_queries = [
            "What is the capital of France?",
            "Who won the last FIFA World Cup?",
            "What is the current weather in Tokyo?",
        ]
        
        results = []
        for query in test_queries:
            result = orch.process_query(query, show_routing=False)
            results.append({
                'query': query,
                'answer': result.get('raw_answer', ''),
                'agent': result.get('agent_used', 'unknown'),
                'success': result.get('success', True)
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Run the app
    print("=" * 80)
    print("🚀 Starting Multi-Agent Flask Web Application")
    print("=" * 80)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📦 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"📝 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 80)
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)