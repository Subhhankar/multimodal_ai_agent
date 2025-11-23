"""
Answer Synthesizer
Formats and presents agent responses in a user-friendly way
"""

from typing import Dict, List
from datetime import datetime


class AnswerSynthesizer:
    """
    Synthesizes and formats responses from different agents into
    a consistent, user-friendly format
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the synthesizer
        
        Args:
            verbose: Whether to include detailed metadata in responses
        """
        self.verbose = verbose
    
    def synthesize(self, agent_response: Dict) -> Dict:
        """
        Main synthesis method that formats agent response
        
        Args:
            agent_response: Raw response from web_search or multimodal agent
        
        Returns:
            Formatted response dictionary with:
            {
                'formatted_answer': str (ready to display),
                'raw_answer': str,
                'agent_used': str,
                'sources': List,
                'metadata': Dict,
                'success': bool
            }
        """
        if agent_response.get('error'):
            return self._format_error_response(agent_response)
        
        agent_type = agent_response.get('agent_used', 'unknown')
        
        if agent_type == 'web_search':
            return self._format_web_search_response(agent_response)
        elif agent_type == 'multimodal':
            return self._format_multimodal_response(agent_response)
        else:
            return self._format_generic_response(agent_response)
    
    def _format_web_search_response(self, response: Dict) -> Dict:
        """Format response from web search agent"""
        answer = response.get('answer', 'No answer generated')
        sources = response.get('sources', [])
        
        # Build formatted answer
        formatted_parts = []
        
        # Add the main answer
        formatted_parts.append("🌐 Web Search Result")
        formatted_parts.append("=" * 60)
        formatted_parts.append("")
        formatted_parts.append(answer)
        formatted_parts.append("")
        
        # Add sources if verbose
        if self.verbose and sources:
            formatted_parts.append("📚 Sources:")
            for i, source in enumerate(sources, 1):
                formatted_parts.append(f"  {i}. {source}")
            formatted_parts.append("")
        
        formatted_answer = "\n".join(formatted_parts)
        
        return {
            'formatted_answer': formatted_answer,
            'raw_answer': answer,
            'agent_used': 'web_search',
            'sources': sources,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_count': len(sources)
            },
            'success': True
        }
    
    def _format_multimodal_response(self, response: Dict) -> Dict:
        """Format response from multimodal agent"""
        answer = response.get('answer', 'No answer generated')
        sources = response.get('sources', [])
        files_processed = response.get('files_processed', [])
        num_sources = response.get('num_sources', len(sources))
        num_images = response.get('num_images', 0)
        
        # Build formatted answer
        formatted_parts = []
        
        # Add header with file info
        formatted_parts.append("📁 Document Analysis Result")
        formatted_parts.append("=" * 60)
        
        if files_processed:
            formatted_parts.append("")
            formatted_parts.append(f"Analyzed files: {len(files_processed)}")
            for file in files_processed:
                formatted_parts.append(f"  • {file}")
        
        formatted_parts.append("")
        
        # Add the main answer
        formatted_parts.append(answer)
        formatted_parts.append("")
        
        # Add source details if verbose
        if self.verbose and sources:
            formatted_parts.append(f"📚 Sources: {num_sources} document chunks")
            if num_images > 0:
                formatted_parts.append(f"🖼️  Images analyzed: {num_images}")
            formatted_parts.append("")
            
            # Group sources by file
            sources_by_file = {}
            for source in sources:
                file_name = source.get('source', 'Unknown')
                if file_name not in sources_by_file:
                    sources_by_file[file_name] = []
                sources_by_file[file_name].append(source)
            
            for file_name, file_sources in sources_by_file.items():
                formatted_parts.append(f"  From: {file_name}")
                for src in file_sources:
                    page = src.get('page')
                    doc_type = src.get('type', 'unknown')
                    similarity = src.get('similarity', 0)
                    
                    detail = f"    - {doc_type}"
                    if page is not None:
                        detail += f" (page {page})"
                    detail += f" [relevance: {similarity:.2f}]"
                    formatted_parts.append(detail)
            formatted_parts.append("")
        
        formatted_answer = "\n".join(formatted_parts)
        
        return {
            'formatted_answer': formatted_answer,
            'raw_answer': answer,
            'agent_used': 'multimodal',
            'sources': sources,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'files_processed': files_processed,
                'source_count': num_sources,
                'image_count': num_images
            },
            'success': True
        }
    
    def _format_error_response(self, response: Dict) -> Dict:
        """Format error response"""
        error_message = response.get('answer', 'Unknown error occurred')
        agent_type = response.get('agent_used', 'unknown')
        
        formatted_parts = []
        formatted_parts.append("❌ Error")
        formatted_parts.append("=" * 60)
        formatted_parts.append("")
        formatted_parts.append(error_message)
        formatted_parts.append("")
        formatted_parts.append(f"Agent: {agent_type}")
        
        formatted_answer = "\n".join(formatted_parts)
        
        return {
            'formatted_answer': formatted_answer,
            'raw_answer': error_message,
            'agent_used': agent_type,
            'sources': [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'error': True
            },
            'success': False
        }
    
    def _format_generic_response(self, response: Dict) -> Dict:
        """Format generic response (fallback)"""
        answer = response.get('answer', 'No answer generated')
        
        formatted_parts = []
        formatted_parts.append("💬 Response")
        formatted_parts.append("=" * 60)
        formatted_parts.append("")
        formatted_parts.append(answer)
        
        formatted_answer = "\n".join(formatted_parts)
        
        return {
            'formatted_answer': formatted_answer,
            'raw_answer': answer,
            'agent_used': response.get('agent_used', 'unknown'),
            'sources': response.get('sources', []),
            'metadata': {
                'timestamp': datetime.now().isoformat()
            },
            'success': True
        }
    
    def synthesize_simple(self, agent_response: Dict) -> str:
        """
        Simplified synthesis that returns just the formatted answer string
        
        Args:
            agent_response: Raw agent response
        
        Returns:
            Formatted answer string
        """
        result = self.synthesize(agent_response)
        return result['formatted_answer']
    
    def create_comparison_response(self, responses: List[Dict]) -> str:
        """
        Create a response comparing multiple agent outputs
        (useful for scenarios where multiple agents are consulted)
        
        Args:
            responses: List of agent response dictionaries
        
        Returns:
            Formatted comparison string
        """
        formatted_parts = []
        formatted_parts.append("🔄 Multi-Agent Response")
        formatted_parts.append("=" * 60)
        formatted_parts.append("")
        
        for i, response in enumerate(responses, 1):
            agent = response.get('agent_used', 'unknown')
            answer = response.get('answer', 'No answer')
            
            formatted_parts.append(f"Agent {i}: {agent.upper()}")
            formatted_parts.append("-" * 40)
            formatted_parts.append(answer)
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    synthesizer = AnswerSynthesizer(verbose=True)
    
    # Test case 1: Web search response
    print("=" * 80)
    print("TEST CASE 1: Web Search Response")
    print("=" * 80)
    
    web_response = {
        'answer': 'Paris is the capital of France. It is located in the north-central part of the country.',
        'sources': ['Web search results from DuckDuckGo'],
        'agent_used': 'web_search',
        'error': False
    }
    
    result = synthesizer.synthesize(web_response)
    print(result['formatted_answer'])
    
    # Test case 2: Multimodal response
    print("\n" + "=" * 80)
    print("TEST CASE 2: Multimodal Response")
    print("=" * 80)
    
    multimodal_response = {
        'answer': 'The document discusses three main topics: artificial intelligence, machine learning, and neural networks. The key finding is that deep learning has significantly improved image recognition accuracy.',
        'sources': [
            {'source': 'research.pdf', 'page': 1, 'type': 'text', 'similarity': 0.89},
            {'source': 'research.pdf', 'page': 3, 'type': 'image', 'similarity': 0.75}
        ],
        'agent_used': 'multimodal',
        'files_processed': ['research.pdf'],
        'num_sources': 2,
        'num_images': 1,
        'error': False
    }
    
    result = synthesizer.synthesize(multimodal_response)
    print(result['formatted_answer'])
    
    # Test case 3: Error response
    print("\n" + "=" * 80)
    print("TEST CASE 3: Error Response")
    print("=" * 80)
    
    error_response = {
        'answer': 'File not found: document.pdf',
        'agent_used': 'multimodal',
        'error': True
    }
    
    result = synthesizer.synthesize(error_response)
    print(result['formatted_answer'])