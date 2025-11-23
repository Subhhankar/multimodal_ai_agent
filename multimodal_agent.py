"""
Multimodal Agent Wrapper
Provides standardized interface for the MultimodalDocumentProcessor
"""

import os
from typing import Dict, List, Optional
from pathlib import Path

# Import your existing multimodal processor
# Make sure multimodal.py is in the same directory
try:
    from multimodal import MultimodalDocumentProcessor
except ImportError:
    print("⚠️  Could not import MultimodalDocumentProcessor from multimodal.py")
    print("   Make sure multimodal.py is in the same directory")
    MultimodalDocumentProcessor = None


class MultimodalAgent:
    """
    Wrapper for MultimodalDocumentProcessor to provide standardized interface
    for the multi-agent system
    """
    
    def __init__(self, google_api_key: str = None, enable_ocr: bool = True):
        """
        Initialize the Multimodal Agent
        
        Args:
            google_api_key: Google API key for Gemini
            enable_ocr: Whether to enable OCR for image text extraction
        """
        if MultimodalDocumentProcessor is None:
            raise ImportError("MultimodalDocumentProcessor not available")
        
        # Initialize the document processor
        self.processor = MultimodalDocumentProcessor(
            google_api_key=google_api_key,
            enable_ocr=enable_ocr,
            store_as_base64=True,  # Store images as base64 for portability
            use_faiss_index=True   # Use FAISS for faster search
        )
        
        # Track processed files
        self.processed_files = []
    
    def execute(
        self, 
        query: str, 
        file_paths: Optional[List[str]] = None,
        top_k: int = 5,
        include_images: bool = True
    ) -> Dict:
        """
        Main execution method for multimodal agent
        
        Args:
            query: User's query about the files
            file_paths: List of file paths to process (if not already processed)
            top_k: Number of most relevant chunks to retrieve
            include_images: Whether to include images in the response
        
        Returns:
            Dictionary with answer, sources, and metadata:
            {
                'answer': str,
                'sources': List[Dict],
                'agent_used': str,
                'files_processed': List[str],
                'num_sources': int,
                'num_images': int,
                'error': bool
            }
        """
        try:
            # Step 1: Process files if provided
            if file_paths:
                print(f"📁 Processing {len(file_paths)} file(s)...")
                for file_path in file_paths:
                    file_path_obj = Path(file_path)
                    
                    if not file_path_obj.exists():
                        return {
                            'answer': f'Error: File not found: {file_path}',
                            'sources': [],
                            'agent_used': 'multimodal',
                            'files_processed': [],
                            'error': True
                        }
                    
                    # Process file
                    success = self.processor.process_file(str(file_path))
                    if success:
                        self.processed_files.append(str(file_path))
                    else:
                        return {
                            'answer': f'Error: Failed to process file: {file_path}',
                            'sources': [],
                            'agent_used': 'multimodal',
                            'files_processed': self.processed_files,
                            'error': True
                        }
            
            # Step 2: Check if any documents have been processed
            if len(self.processor.all_docs) == 0:
                return {
                    'answer': 'No documents have been processed yet. Please provide files to analyze.',
                    'sources': [],
                    'agent_used': 'multimodal',
                    'files_processed': self.processed_files,
                    'error': False
                }
            
            # Step 3: Answer the question using processed documents
            print(f"🤔 Answering query: {query}")
            result = self.processor.answer_question(
                question=query,
                top_k=top_k,
                include_images=include_images,
                max_images=3
            )
            
            # Step 4: Format response in standardized format
            if result.get('error'):
                return {
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'agent_used': 'multimodal',
                    'files_processed': self.processed_files,
                    'error': True
                }
            else:
                return {
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'agent_used': 'multimodal',
                    'files_processed': self.processed_files,
                    'num_sources': result.get('num_sources', 0),
                    'num_images': result.get('num_images', 0),
                    'error': False
                }
        
        except Exception as e:
            return {
                'answer': f'Error in multimodal agent: {str(e)}',
                'sources': [],
                'agent_used': 'multimodal',
                'files_processed': self.processed_files,
                'error': True
            }
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about processed documents
        
        Returns:
            Statistics dictionary
        """
        return self.processor.get_statistics()
    
    def reset(self):
        """
        Reset the agent (clear all processed documents)
        """
        self.processor.all_docs = []
        self.processor.all_embeddings = []
        self.processor.image_data_store = {}
        self.processor.faiss_index = None
        self.processed_files = []
        print("✅ Agent reset complete")
    
    def quick_query(self, query: str, file_paths: List[str] = None) -> str:
        """
        Simplified interface that returns just the answer string
        
        Args:
            query: Question to answer
            file_paths: Files to process (optional)
        
        Returns:
            Answer string
        """
        result = self.execute(query, file_paths)
        return result['answer']


# Standalone function for backward compatibility
def analyze_files(query: str, file_paths: List[str], google_api_key: str = None) -> Dict:
    """
    Standalone function to analyze files and answer query
    
    Args:
        query: Question about the files
        file_paths: List of file paths
        google_api_key: Optional API key
    
    Returns:
        Result dictionary
    """
    agent = MultimodalAgent(google_api_key=google_api_key)
    return agent.execute(query, file_paths)


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = MultimodalAgent()
    
    # Example 1: Process and query a PDF
    print("=" * 80)
    print("Example 1: PDF Analysis")
    print("=" * 80)
    
    # Replace with your actual file path
    pdf_file = "sample.pdf"
    
    if Path(pdf_file).exists():
        result = agent.execute(
            query="What are the main topics discussed in this document?",
            file_paths=[pdf_file]
        )
        
        print(f"\n📄 Answer: {result['answer']}")
        print(f"\n📚 Sources: {len(result['sources'])} sources used")
    else:
        print(f"⚠️  File not found: {pdf_file}")
    
    print("\n" + "=" * 80)
    
    # Example 2: Query without processing new files (using previously processed files)
    if len(agent.processed_files) > 0:
        print("Example 2: Follow-up query")
        print("=" * 80)
        
        result = agent.execute(
            query="Can you summarize the key points?"
        )
        
        print(f"\n📄 Answer: {result['answer']}")
    
    # Example 3: Get statistics
    print("\n" + "=" * 80)
    print("Document Statistics")
    print("=" * 80)
    stats = agent.get_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Document types: {stats['document_types']}")