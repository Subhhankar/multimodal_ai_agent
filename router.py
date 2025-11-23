

import os
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class QueryRouter:
    """
    Intelligent query router that determines which agent should handle the request.
    Uses a combination of rule-based and LLM-based classification.
    """
    
    def __init__(self, google_api_key: str = None):
        """
        Initialize the router with LLM for intelligent classification
        
        Args:
            google_api_key: Google API key for Gemini (optional if in env)
        """
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize LLM for intelligent classification
        self.llm = None
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.1  # Low temperature for consistent classification
                )
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini for routing: {e}")
        
        # Keywords that indicate web search
        self.web_search_keywords = [
            'current', 'latest', 'today', 'now', 'recent', 'news',
            'weather', 'price', 'stock', 'score', 'result',
            'who won', 'what happened', 'when did', 'breaking',
            'update', 'live', 'real-time', 'this week', 'this month',
            'yesterday', 'last night', 'trending'
        ]
        
        # Keywords that might indicate file analysis
        self.file_analysis_keywords = [
            'this document', 'this file', 'this image', 'this pdf',
            'uploaded', 'attached', 'in the file', 'from the document',
            'analyze', 'summarize this', 'describe this', 'extract from',
            'what does this', 'show me in', 'find in this'
        ]
        
        # Question types that typically need web search
        self.factual_question_patterns = [
            'what is the', 'who is', 'when was', 'where is',
            'how many', 'capital of', 'president of', 'population of',
            'define', 'meaning of', 'explain what'
        ]
    
    def route(self, query: str, files: Optional[List[str]] = None) -> Dict:
        """
        Main routing method that determines which agent should handle the query
        
        Args:
            query: User's query string
            files: List of file paths (if any files are attached)
        
        Returns:
            Dictionary with routing decision:
            {
                'agent': 'web_search' or 'multimodal',
                'confidence': float (0-1),
                'reason': str (explanation of routing decision)
            }
        """
        query_lower = query.lower()
        
        # RULE 1: If files are attached, route to multimodal agent
        if files and len(files) > 0:
            return {
                'agent': 'multimodal',
                'confidence': 1.0,
                'reason': f'Files attached ({len(files)} file(s)). Routing to file analysis agent.'
            }
        
        # RULE 2: Check for explicit file reference keywords
        file_keywords_found = [kw for kw in self.file_analysis_keywords if kw in query_lower]
        if file_keywords_found:
            return {
                'agent': 'multimodal',
                'confidence': 0.9,
                'reason': f'Query references files/documents (keywords: {file_keywords_found}). Routing to file analysis agent.'
            }
        
        # RULE 3: Check for web search keywords (real-time/current info)
        web_keywords_found = [kw for kw in self.web_search_keywords if kw in query_lower]
        if web_keywords_found:
            return {
                'agent': 'web_search',
                'confidence': 0.85,
                'reason': f'Query requires current/real-time information (keywords: {web_keywords_found}). Routing to web search agent.'
            }
        
        # RULE 4: Check for factual question patterns
        factual_patterns_found = [pattern for pattern in self.factual_question_patterns if pattern in query_lower]
        if factual_patterns_found:
            return {
                'agent': 'web_search',
                'confidence': 0.75,
                'reason': f'Factual question detected (patterns: {factual_patterns_found}). Routing to web search agent.'
            }
        
        # RULE 5: If ambiguous, use LLM-based classification
        if self.llm:
            return self._llm_based_routing(query)
        else:
            # Fallback: Default to web search for general queries
            return {
                'agent': 'web_search',
                'confidence': 0.5,
                'reason': 'No clear indicators. Defaulting to web search agent (LLM classifier unavailable).'
            }
    
    def _llm_based_routing(self, query: str) -> Dict:
        """
        Use LLM to classify ambiguous queries
        
        Args:
            query: User's query string
        
        Returns:
            Routing decision dictionary
        """
        classification_prompt = f"""You are a query classifier for a multi-agent system.

The system has two agents:
1. WEB_SEARCH agent: Handles queries requiring real-time information, current events, factual lookups from the internet, general knowledge questions, definitions, etc.
2. MULTIMODAL agent: Handles queries about uploaded files (PDFs, images, documents, spreadsheets), analyzing file content, extracting information from documents.

Analyze this query and determine which agent should handle it:
Query: "{query}"

IMPORTANT RULES:
- If the query asks about files, documents, images, or uploaded content → MULTIMODAL
- If the query asks for current information, news, weather, prices, scores → WEB_SEARCH
- If the query is a general knowledge or factual question → WEB_SEARCH
- If the query is ambiguous but doesn't mention files → WEB_SEARCH (default)

Respond with ONLY ONE WORD: either "WEB_SEARCH" or "MULTIMODAL"
Do not include any explanation or additional text."""

        try:
            response = self.llm.invoke(classification_prompt)
            decision = response.content.strip().upper()
            
            # Parse LLM response
            if "MULTIMODAL" in decision:
                return {
                    'agent': 'multimodal',
                    'confidence': 0.7,
                    'reason': 'LLM classified as requiring file analysis (ambiguous query).'
                }
            elif "WEB_SEARCH" in decision:
                return {
                    'agent': 'web_search',
                    'confidence': 0.7,
                    'reason': 'LLM classified as requiring web search (ambiguous query).'
                }
            else:
                # LLM gave unexpected response, default to web search
                return {
                    'agent': 'web_search',
                    'confidence': 0.5,
                    'reason': f'LLM gave unclear response: {decision}. Defaulting to web search.'
                }
        
        except Exception as e:
            print(f"⚠️  LLM routing failed: {e}")
            return {
                'agent': 'web_search',
                'confidence': 0.5,
                'reason': 'LLM classification failed. Defaulting to web search agent.'
            }
    
    def explain_routing(self, query: str, files: Optional[List[str]] = None) -> str:
        """
        Get detailed explanation of routing decision (for debugging/transparency)
        
        Args:
            query: User's query
            files: List of files (if any)
        
        Returns:
            Human-readable explanation string
        """
        decision = self.route(query, files)
        
        explanation = f"""
📍 ROUTING DECISION
==================
Query: "{query}"
Files attached: {len(files) if files else 0}

Selected Agent: {decision['agent'].upper()}
Confidence: {decision['confidence']:.0%}
Reason: {decision['reason']}
"""
        return explanation


# Example usage and testing
if __name__ == "__main__":
    # Initialize router
    router = QueryRouter()
    
    # Test cases
    test_queries = [
        ("What is the capital of France?", None),
        ("What is the current weather in Tokyo?", None),
        ("Summarize this document", ["document.pdf"]),
        ("Describe this image", ["cat.jpg"]),
        ("Who won the last FIFA World Cup?", None),
        ("Find mentions of quantum computing in this file", ["research.pdf"]),
        ("What is quantum computing?", None),
        ("Analyze the data in this spreadsheet", ["sales.xlsx"]),
    ]
    
    print("=" * 80)
    print("QUERY ROUTER TEST CASES")
    print("=" * 80)
    
    for query, files in test_queries:
        decision = router.route(query, files)
        print(f"\nQuery: {query}")
        if files:
            print(f"Files: {files}")
        print(f"→ Agent: {decision['agent']}")
        print(f"  Confidence: {decision['confidence']:.0%}")
        print(f"  Reason: {decision['reason']}")
        print("-" * 80)