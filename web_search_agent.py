"""
Web Search Agent - Modified for Multi-Agent System
Handles queries requiring real-time information from the internet
"""

import os
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()


class WebSearchAgent:
    """
    Agent specialized in retrieving and synthesizing information from the web.
    Uses DuckDuckGo search and Gemini for answer synthesis.
    """
    
    def __init__(self, google_api_key: str = None, max_search_results: int = 5):
        """
        Initialize the Web Search Agent
        
        Args:
            google_api_key: Google API key for Gemini (optional if in env)
            max_search_results: Maximum number of search results to consider
        """
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize search tool
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception as e:
            print(f"⚠️  Could not initialize DuckDuckGo search: {e}")
            self.search_tool = None
        
        # Initialize LLM for answer synthesis
        self.llm = None
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.3,
                    max_tokens=2048
                )
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini: {e}")
        
        self.max_search_results = max_search_results
    
    def execute(self, query: str, include_sources: bool = True) -> Dict:
        """
        Main execution method for web search agent
        
        Args:
            query: User's search query
            include_sources: Whether to include source information in response
        
        Returns:
            Dictionary with answer, sources, and metadata:
            {
                'answer': str,
                'sources': List[str],
                'agent_used': str,
                'search_results': str (raw results),
                'error': bool
            }
        """
        # Check if tools are initialized
        if not self.search_tool:
            return {
                'answer': 'Error: Search tool not initialized.',
                'sources': [],
                'agent_used': 'web_search',
                'error': True
            }
        
        if not self.llm:
            return {
                'answer': 'Error: LLM not initialized. Please set GOOGLE_API_KEY.',
                'sources': [],
                'agent_used': 'web_search',
                'error': True
            }
        
        try:
            # Step 1: Perform web search
            print(f"🔍 Searching the web for: {query}")
            search_results = self.search_tool.invoke(query)
            
            if not search_results or len(search_results.strip()) == 0:
                return {
                    'answer': 'No search results found. Please try rephrasing your query.',
                    'sources': [],
                    'agent_used': 'web_search',
                    'search_results': '',
                    'error': False
                }
            
            # Step 2: Synthesize answer using LLM
            answer = self._synthesize_answer(query, search_results)
            
            # Step 3: Extract sources (simplified - just mention web search was used)
            sources = ['Web search results from DuckDuckGo']
            
            return {
                'answer': answer,
                'sources': sources,
                'agent_used': 'web_search',
                'search_results': search_results[:500],  # Keep first 500 chars for reference
                'error': False
            }
        
        except Exception as e:
            return {
                'answer': f'Error during web search: {str(e)}',
                'sources': [],
                'agent_used': 'web_search',
                'error': True
            }
    
    def _synthesize_answer(self, query: str, search_results: str) -> str:
        """
        Use LLM to synthesize a clean answer from search results
        
        Args:
            query: Original user query
            search_results: Raw search results from DuckDuckGo
        
        Returns:
            Synthesized answer string
        """
        synthesis_prompt = f"""You are a helpful AI assistant. Based on the web search results below, provide a clear, accurate, and concise answer to the user's question.

USER QUESTION:
{query}

WEB SEARCH RESULTS:
{search_results}

INSTRUCTIONS:
- Provide a direct, informative answer based on the search results
- Be concise but complete
- If the search results don't fully answer the question, say so
- Use natural language, not bullet points unless listing items
- Don't mention "according to search results" - just provide the information naturally
- If dates or specific facts are mentioned, include them

ANSWER:"""

        try:
            response = self.llm.invoke(synthesis_prompt)
            return response.content.strip()
        except Exception as e:
            # Fallback: return raw search results if synthesis fails
            return f"Search results: {search_results[:500]}..."
    
    def quick_search(self, query: str) -> str:
        """
        Quick search that returns just the answer string (simplified interface)
        
        Args:
            query: Search query
        
        Returns:
            Answer string
        """
        result = self.execute(query)
        return result['answer']


# Standalone function for backward compatibility
def web_search_query(query: str, google_api_key: str = None) -> Dict:
    """
    Standalone function to perform web search and get answer
    
    Args:
        query: Search query
        google_api_key: Optional API key
    
    Returns:
        Result dictionary
    """
    agent = WebSearchAgent(google_api_key=google_api_key)
    return agent.execute(query)


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = WebSearchAgent()
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Who won the last FIFA World Cup?",
        "What is the current weather in Tokyo?",
        "Latest news about AI",
    ]
    
    print("=" * 80)
    print("WEB SEARCH AGENT TEST CASES")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        print("-" * 80)
        
        result = agent.execute(query)
        
        if result['error']:
            print(f"❌ Error: {result['answer']}")
        else:
            print(f"✅ Answer: {result['answer']}")
            print(f"\n📚 Sources: {', '.join(result['sources'])}")
        
        print("=" * 80)