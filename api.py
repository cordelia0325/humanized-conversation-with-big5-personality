"""
Flask API Module for Humanized Conversation with Personality

This module provides the REST API for the personality conversation system.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import uuid
from typing import Dict, Optional
import json
import os

from config import api_config
from data_processing import DataLoader, Persona
from personality_model import (
    PersonalityConversationModel,
    PersonalityModelFactory,
    DialogueManager,
    RolePlayingAgent
)

# =============================================================================
# Flask App Setup
# =============================================================================

app = Flask(__name__)
CORS(app, origins=api_config.cors_origins)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================

class AppState:
    """Global application state"""
    
    def __init__(self):
        self.model: Optional[PersonalityConversationModel] = None
        self.dialogue_manager: Optional[DialogueManager] = None
        self.personas: Dict[int, Persona] = {}
        self.agents: Dict[str, RolePlayingAgent] = {}
        self.sessions: Dict[str, Dict] = {}
        self.initialized = False
    
    def initialize(
        self,
        load_llama: bool = False,
        load_bert: bool = False,
        personas_path: str = None
    ):
        """Initialize the application"""
        
        logger.info("Initializing application...")
        
        # Create conversation model
        self.model = PersonalityModelFactory.create_model(
            load_llama=load_llama,
            load_bert=load_bert
        )
        
        # Create dialogue manager
        self.dialogue_manager = DialogueManager(self.model)
        
        # Load personas if path provided
        if personas_path and os.path.exists(personas_path):
            loader = DataLoader()
            personas_list = loader.load_big5_personas(personas_path)
            self.personas = {p.index: p for p in personas_list}
            logger.info(f"Loaded {len(self.personas)} personas")
        
        self.initialized = True
        logger.info("Application initialized successfully")

# Global state instance
state = AppState()

# =============================================================================
# Middleware
# =============================================================================

@app.before_request
def check_initialization():
    """Check if app is initialized before processing requests"""
    
    # Skip for health check and init endpoints
    if request.endpoint in ['health_check', 'initialize', 'static']:
        return
    
    if not state.initialized:
        return jsonify({
            'error': 'Application not initialized',
            'message': 'Please call /api/init first'
        }), 503

# =============================================================================
# API Routes - System
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'initialized': state.initialized,
        'personas_loaded': len(state.personas),
        'active_sessions': len(state.sessions)
    })

@app.route('/api/init', methods=['POST'])
def initialize():
    """
    Initialize the application
    
    Request body:
    {
        "load_llama": false,
        "load_bert": false,
        "personas_path": "path/to/personas.json"
    }
    """
    data = request.get_json() or {}
    
    try:
        state.initialize(
            load_llama=data.get('load_llama', False),
            load_bert=data.get('load_bert', False),
            personas_path=data.get('personas_path')
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Application initialized',
            'personas_count': len(state.personas)
        })
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# =============================================================================
# API Routes - Personas
# =============================================================================

@app.route('/api/personas', methods=['GET'])
def list_personas():
    """
    List all available personas
    
    Query params:
    - big5: Filter by Big5 string (e.g., "{high, low, high, high, low}")
    - limit: Maximum number of results
    - offset: Pagination offset
    """
    big5_filter = request.args.get('big5')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    personas = list(state.personas.values())
    
    # Apply filter
    if big5_filter:
        personas = [p for p in personas if p.big5 == big5_filter]
    
    # Apply pagination
    total = len(personas)
    personas = personas[offset:offset + limit]
    
    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'personas': [
            {
                'index': p.index,
                'name': p.name,
                'big5': p.big5,
                'job': p.job,
                'age': p.age,
                'region': p.region
            }
            for p in personas
        ]
    })

@app.route('/api/personas/<int:persona_id>', methods=['GET'])
def get_persona(persona_id: int):
    """Get a specific persona by ID"""
    
    persona = state.personas.get(persona_id)
    
    if not persona:
        return jsonify({
            'error': 'Persona not found',
            'persona_id': persona_id
        }), 404
    
    return jsonify({
        'index': persona.index,
        'name': persona.name,
        'big5': persona.big5,
        'gender': persona.gender,
        'age': persona.age,
        'region': persona.region,
        'tone': persona.tone,
        'job': persona.job,
        'personality': persona.personality,
        'advantages_and_disadvantages': persona.advantages_and_disadvantages,
        'hobby': persona.hobby,
        'growth_experience': persona.growth_experience,
        'family_relationship': persona.family_relationship,
        'working_conditions': persona.working_conditions,
        'social_relationship': persona.social_relationship,
        'emotional_state': persona.emotional_state,
        'living_conditions': persona.living_conditions,
        'recent_worry_or_anxiety': persona.recent_worry_or_anxiety,
        'additional_information': persona.additional_information
    })

@app.route('/api/personas/random', methods=['GET'])
def get_random_persona():
    """Get a random persona"""
    import random
    
    if not state.personas:
        return jsonify({
            'error': 'No personas loaded'
        }), 404
    
    persona = random.choice(list(state.personas.values()))
    
    return jsonify({
        'index': persona.index,
        'name': persona.name,
        'big5': persona.big5,
        'job': persona.job
    })

# =============================================================================
# API Routes - Conversations
# =============================================================================

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """
    Create a new conversation with a persona
    
    Request body:
    {
        "persona_id": 0,
        "user_id": "optional_user_id"
    }
    
    Returns:
    {
        "session_id": "uuid",
        "persona": {...},
        "greeting": "Hello message"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    persona_id = data.get('persona_id')
    user_id = data.get('user_id', str(uuid.uuid4())[:8])
    
    if persona_id is None:
        return jsonify({'error': 'persona_id required'}), 400
    
    persona = state.personas.get(persona_id)
    
    if not persona:
        return jsonify({
            'error': 'Persona not found',
            'persona_id': persona_id
        }), 404
    
    # Create session
    session_id = str(uuid.uuid4())
    
    try:
        # Start conversation through dialogue manager
        greeting = state.dialogue_manager.start_conversation(user_id, persona)
        conversation_id = f"{user_id}_{persona.index}"
        
        # Create agent
        agent = PersonalityModelFactory.create_agent(persona, state.model)
        state.agents[session_id] = agent
        
        # Store session
        state.sessions[session_id] = {
            'user_id': user_id,
            'persona_id': persona_id,
            'conversation_id': conversation_id,
            'messages': [{'role': 'assistant', 'content': greeting}]
        }
        
        return jsonify({
            'session_id': session_id,
            'persona': {
                'index': persona.index,
                'name': persona.name,
                'big5': persona.big5,
                'job': persona.job
            },
            'greeting': greeting
        })
    
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return jsonify({
            'error': 'Failed to create conversation',
            'message': str(e)
        }), 500

@app.route('/api/conversations/<session_id>/message', methods=['POST'])
def send_message(session_id: str):
    """
    Send a message in a conversation
    
    Request body:
    {
        "message": "User message text"
    }
    
    Returns:
    {
        "response": "AI response text",
        "memory_used": "optional memory context"
    }
    """
    if session_id not in state.sessions:
        return jsonify({
            'error': 'Session not found',
            'session_id': session_id
        }), 404
    
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'message required'}), 400
    
    user_message = data['message']
    session = state.sessions[session_id]
    
    try:
        # Get response from dialogue manager
        response = state.dialogue_manager.send_message(
            session['conversation_id'],
            user_message
        )
        
        # Update session
        session['messages'].append({'role': 'user', 'content': user_message})
        session['messages'].append({'role': 'assistant', 'content': response})
        
        # Get conversation stats
        stats = state.dialogue_manager.get_conversation_stats(session['conversation_id'])
        
        return jsonify({
            'response': response,
            'memory_used': stats.get('last_memory_used'),
            'turn_count': stats.get('turn_count', 0)
        })
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({
            'error': 'Failed to process message',
            'message': str(e)
        }), 500

@app.route('/api/conversations/<session_id>', methods=['GET'])
def get_conversation(session_id: str):
    """Get conversation history and status"""
    
    if session_id not in state.sessions:
        return jsonify({
            'error': 'Session not found',
            'session_id': session_id
        }), 404
    
    session = state.sessions[session_id]
    persona = state.personas.get(session['persona_id'])
    
    stats = state.dialogue_manager.get_conversation_stats(session['conversation_id'])
    
    return jsonify({
        'session_id': session_id,
        'persona': {
            'index': persona.index if persona else None,
            'name': persona.name if persona else None,
            'big5': persona.big5 if persona else None
        },
        'messages': session['messages'],
        'stats': stats
    })

@app.route('/api/conversations/<session_id>', methods=['DELETE'])
def end_conversation(session_id: str):
    """End a conversation and clean up resources"""
    
    if session_id not in state.sessions:
        return jsonify({
            'error': 'Session not found',
            'session_id': session_id
        }), 404
    
    session = state.sessions[session_id]
    
    # Clean up
    state.dialogue_manager.end_conversation(session['conversation_id'])
    
    if session_id in state.agents:
        del state.agents[session_id]
    
    del state.sessions[session_id]
    
    return jsonify({
        'status': 'success',
        'message': 'Conversation ended'
    })

# =============================================================================
# API Routes - Personality Analysis
# =============================================================================

@app.route('/api/analysis/big5', methods=['GET'])
def get_big5_combinations():
    """Get all possible Big5 combinations"""
    
    from persona_generator import PersonaGenerator
    generator = PersonaGenerator()
    combinations = generator.generate_all_combinations()
    
    return jsonify({
        'total': len(combinations),
        'combinations': [
            generator.format_big5_string(combo)
            for combo in combinations
        ]
    })

@app.route('/api/analysis/persona/<int:persona_id>/profile', methods=['GET'])
def get_persona_profile(persona_id: int):
    """Get detailed personality profile for a persona"""
    
    persona = state.personas.get(persona_id)
    
    if not persona:
        return jsonify({
            'error': 'Persona not found'
        }), 404
    
    from persona_generator import BehaviorPresetGenerator
    
    big5_dict = persona.get_big5_dict()
    behavior_gen = BehaviorPresetGenerator()
    presets = behavior_gen.generate_behavior_presets(big5_dict)
    
    return jsonify({
        'persona_id': persona_id,
        'name': persona.name,
        'big5': big5_dict,
        'system_prompt': persona.to_system_prompt(),
        'behavior_presets': presets
    })

# =============================================================================
# API Routes - Evaluation (for testing)
# =============================================================================

@app.route('/api/evaluate/interview', methods=['POST'])
def conduct_interview():
    """
    Conduct a structured interview with a persona
    
    Request body:
    {
        "session_id": "existing_session_id",
        "num_questions": 5
    }
    """
    data = request.get_json()
    
    if not data or 'session_id' not in data:
        return jsonify({'error': 'session_id required'}), 400
    
    session_id = data['session_id']
    num_questions = data.get('num_questions', 5)
    
    if session_id not in state.sessions:
        return jsonify({
            'error': 'Session not found'
        }), 404
    
    # Note: Full interview implementation would require the evaluation module
    # This is a placeholder for the API structure
    
    return jsonify({
        'status': 'not_implemented',
        'message': 'Interview evaluation requires additional setup'
    })

# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# =============================================================================
# Main Entry Point
# =============================================================================

def create_app(
    personas_path: str = None,
    load_llama: bool = False,
    load_bert: bool = False
) -> Flask:
    """Create and configure the Flask app"""
    
    # Initialize state
    state.initialize(
        load_llama=load_llama,
        load_bert=load_bert,
        personas_path=personas_path
    )
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Personality Conversation API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--personas', help='Path to personas JSON file')
    parser.add_argument('--load-llama', action='store_true', help='Load Llama model')
    parser.add_argument('--load-bert', action='store_true', help='Load BERT model')
    
    args = parser.parse_args()
    
    # Initialize application
    state.initialize(
        load_llama=args.load_llama,
        load_bert=args.load_bert,
        personas_path=args.personas
    )
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
