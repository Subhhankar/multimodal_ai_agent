

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import all components
from router import QueryRouter
from web_search_agent import WebSearchAgent
from multimodal_agent import MultimodalAgent
from synthesizer import AnswerSynthesizer

load_dotenv()


class MultiAgentOrchestrator:
    """
    Main orchestrator that coordinates the entire multi-agent system.
    
    Architecture Flow:
    1. User Query → Router (determines which agent to use)
    2. Router → Agent (executes the task)
    3. Agent → Synthesizer (formats the response)
    4. Synthesizer → User (final answer)
    """
    
    def __init__(
        self, 
        google_api_key: str = None,
        enable_ocr: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the multi-agent orchestrator
        
        Args:
            google_api_key: Google API key for Gemini (optional if in env)
            enable_ocr: Whether to enable OCR for image text extraction
            verbose: Whether to show detailed processing information
        """
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.verbose = verbose
        
        # Initialize all components
        print("🚀 Initializing Multi-Agent System...")
        
        try:
            self.router = QueryRouter(google_api_key=google_api_key)
            print("  ✅ Router initialized")
        except Exception as e:
            print(f"  ❌ Router initialization failed: {e}")
            self.router = None
        
        try:
            self.web_agent = WebSearchAgent(google_api_key=google_api_key)
            print("  ✅ Web Search Agent initialized")
        except Exception as e:
            print(f"  ❌ Web Search Agent initialization failed: {e}")
            self.web_agent = None
        
        try:
            self.multimodal_agent = MultimodalAgent(
                google_api_key=google_api_key,
                enable_ocr=enable_ocr
            )
            print("  ✅ Multimodal Agent initialized")
        except Exception as e:
            print(f"  ❌ Multimodal Agent initialization failed: {e}")
            self.multimodal_agent = None
        
        self.synthesizer = AnswerSynthesizer(verbose=verbose)
        print("  ✅ Answer Synthesizer initialized")
        
        print("✨ System ready!\n")
        
        # Track conversation history (optional for future enhancements)
        self.conversation_history = []
    
    def process_query(
        self, 
        query: str, 
        files: Optional[List[str]] = None,
        show_routing: bool = True
    ) -> Dict:
        """
        Main method to process user queries
        
        Args:
            query: User's question or request
            files: List of file paths (optional)
            show_routing: Whether to display routing decision
        
        Returns:
            Dictionary with complete response:
            {
                'formatted_answer': str (ready to display),
                'raw_answer': str,
                'agent_used': str,
                'routing_decision': Dict,
                'sources': List,
                'success': bool
            }
        """
        if not self.router:
            return self._create_error_response("Router not initialized")
        
        # Step 1: Route the query
        if self.verbose:
            print(f"📝 Processing query: {query}")
            if files:
                print(f"📎 Files attached: {len(files)}")
        
        routing_decision = self.router.route(query, files)
        
        if show_routing and self.verbose:
            print(f"🎯 Routing to: {routing_decision['agent'].upper()} agent")
            print(f"   Confidence: {routing_decision['confidence']:.0%}")
            print(f"   Reason: {routing_decision['reason']}\n")
        
        # Step 2: Execute the appropriate agent
        agent_type = routing_decision['agent']
        
        if agent_type == 'web_search':
            agent_response = self._execute_web_search(query)
        elif agent_type == 'multimodal':
            agent_response = self._execute_multimodal(query, files)
        else:
            agent_response = {
                'answer': f"Unknown agent type: {agent_type}",
                'error': True,
                'agent_used': 'unknown'
            }
        
        # Step 3: Synthesize the response
        synthesized_response = self.synthesizer.synthesize(agent_response)
        
        # Step 4: Add routing information
        synthesized_response['routing_decision'] = routing_decision
        
        # Step 5: Store in conversation history
        self.conversation_history.append({
            'query': query,
            'files': files,
            'routing': routing_decision,
            'response': synthesized_response
        })
        
        return synthesized_response
    
    def _execute_web_search(self, query: str) -> Dict:
        """Execute web search agent"""
        if not self.web_agent:
            return {
                'answer': 'Error: Web Search Agent not initialized',
                'error': True,
                'agent_used': 'web_search'
            }
        
        try:
            return self.web_agent.execute(query)
        except Exception as e:
            return {
                'answer': f'Error executing web search: {str(e)}',
                'error': True,
                'agent_used': 'web_search'
            }
    
    def _execute_multimodal(self, query: str, files: Optional[List[str]] = None) -> Dict:
        """Execute multimodal agent"""
        if not self.multimodal_agent:
            return {
                'answer': 'Error: Multimodal Agent not initialized',
                'error': True,
                'agent_used': 'multimodal'
            }
        
        try:
            return self.multimodal_agent.execute(query, file_paths=files)
        except Exception as e:
            return {
                'answer': f'Error executing multimodal agent: {str(e)}',
                'error': True,
                'agent_used': 'multimodal'
            }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response"""
        return {
            'formatted_answer': f"❌ System Error\n{'=' * 60}\n\n{error_message}",
            'raw_answer': error_message,
            'agent_used': 'system',
            'routing_decision': None,
            'sources': [],
            'success': False
        }
    
    def chat(self, query: str, files: Optional[List[str]] = None) -> str:
        """
        Simplified chat interface that returns just the formatted answer
        
        Args:
            query: User's question
            files: Optional list of file paths
        
        Returns:
            Formatted answer string
        """
        result = self.process_query(query, files, show_routing=self.verbose)
        return result['formatted_answer']
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")
    
    def reset_multimodal_agent(self):
        """Reset the multimodal agent (clear processed documents)"""
        if self.multimodal_agent:
            self.multimodal_agent.reset()
    
    def get_system_status(self) -> Dict:
        """Get status of all system components"""
        return {
            'router': self.router is not None,
            'web_agent': self.web_agent is not None,
            'multimodal_agent': self.multimodal_agent is not None,
            'synthesizer': self.synthesizer is not None,
            'conversation_history_length': len(self.conversation_history)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the orchestrator
    orchestrator = MultiAgentOrchestrator(verbose=True)
    
    # Check system status
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)
    status = orchestrator.get_system_status()
    for component, is_active in status.items():
        status_icon = "✅" if is_active else "❌"
        print(f"{status_icon} {component}: {'Active' if is_active else 'Inactive'}")
    print("\n")
    
    # Test cases matching the assignment requirements
    test_cases = [
        {
            'query': "What is the capital of France?",
            'files': None,
            'description': "Web Search - Factual Question"
        },
        {
            'query': "Who won the last FIFA World Cup?",
            'files': None,
            'description': "Web Search - Sports Question"
        },
        {
            'query': "What is the current weather in Tokyo?",
            'files': None,
            'description': "Web Search - Real-time Information"
        },
        # Uncomment and modify with actual file paths to test multimodal
        # {
        #     'query': "Describe the main subject in this image",
        #     'files': ["cat.jpg"],
        #     'description': "Multimodal - Image Analysis"
        # },
        # {
        #     'query': "Summarize the key findings in this document",
        #     'files': ["research.pdf"],
        #     'description': "Multimodal - PDF Analysis"
        # },
    ]
    
    # Run test cases
    for i, test in enumerate(test_cases, 1):
        print("=" * 80)
        print(f"TEST CASE {i}: {test['description']}")
        print("=" * 80)
        print(f"Query: {test['query']}")
        if test['files']:
            print(f"Files: {test['files']}")
        print("-" * 80)
        print()
        
        result = orchestrator.process_query(
            query=test['query'],
            files=test['files'],
            show_routing=True
        )
        
        print(result['formatted_answer'])
        print("\n")
    
    # Show conversation history summary
    print("=" * 80)
    print("CONVERSATION HISTORY SUMMARY")
    print("=" * 80)
    history = orchestrator.get_conversation_history()
    print(f"Total queries processed: {len(history)}")
    for i, entry in enumerate(history, 1):
        agent = entry['routing']['agent']
        print(f"{i}. {entry['query'][:50]}... → {agent}")