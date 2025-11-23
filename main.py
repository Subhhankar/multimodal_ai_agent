"""
Main User Interface for Multi-Agent System
Provides interactive command-line interface for testing
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import the orchestrator
from orchestrator import MultiAgentOrchestrator

load_dotenv()


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           INTELLIGENT MULTI-AGENT ASSISTANT                 ║
    ║                                                              ║
    ║  A smart routing system that directs queries to the         ║
    ║  most appropriate specialized agent                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print command menu"""
    menu = """
    📋 AVAILABLE COMMANDS:
    ─────────────────────────────────────────────────────────────
    1. query <your question>          - Ask a question
    2. file <path> <question>         - Analyze a file
    3. multifile <path1,path2> <q>   - Analyze multiple files
    4. test                           - Run test cases
    5. status                         - Show system status
    6. history                        - Show conversation history
    7. clear                          - Clear conversation history
    8. reset                          - Reset multimodal agent
    9. help                           - Show this menu
    10. exit                          - Exit the system
    ─────────────────────────────────────────────────────────────
    """
    print(menu)


def run_test_cases(orchestrator: MultiAgentOrchestrator):
    """Run predefined test cases from the assignment"""
    
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
        {
            'query': "What is quantum computing?",
            'files': None,
            'description': "Web Search - Definition"
        },
    ]
    
    print("\n" + "=" * 80)
    print("🧪 RUNNING TEST CASES")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {test['description']}")
        print(f"{'─' * 80}")
        print(f"📝 Query: {test['query']}")
        
        if test['files']:
            print(f"📎 Files: {test['files']}")
        
        print()
        
        try:
            result = orchestrator.process_query(
                query=test['query'],
                files=test['files'],
                show_routing=True
            )
            
            print(result['formatted_answer'])
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Test cases completed!")
    print("=" * 80)


def interactive_mode(orchestrator: MultiAgentOrchestrator):
    """Run interactive command-line interface"""
    
    print_banner()
    print("\n🎉 System initialized successfully!")
    print("💡 Type 'help' to see available commands or 'test' to run demo queries\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            # Handle commands
            if command == 'exit' or command == 'quit':
                print("\n👋 Goodbye! Thank you for using the Multi-Agent Assistant.")
                break
            
            elif command == 'help':
                print_menu()
            
            elif command == 'test':
                run_test_cases(orchestrator)
            
            elif command == 'status':
                print("\n📊 SYSTEM STATUS")
                print("=" * 60)
                status = orchestrator.get_system_status()
                for component, is_active in status.items():
                    status_icon = "✅" if is_active else "❌"
                    print(f"{status_icon} {component}: {'Active' if is_active else 'Inactive'}")
                print()
            
            elif command == 'history':
                print("\n📜 CONVERSATION HISTORY")
                print("=" * 60)
                history = orchestrator.get_conversation_history()
                if not history:
                    print("No conversations yet.")
                else:
                    for i, entry in enumerate(history, 1):
                        agent = entry['routing']['agent']
                        query_preview = entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query']
                        print(f"{i}. [{agent}] {query_preview}")
                print()
            
            elif command == 'clear':
                orchestrator.clear_history()
                print("✅ Conversation history cleared.")
            
            elif command == 'reset':
                orchestrator.reset_multimodal_agent()
                print("✅ Multimodal agent reset (all processed documents cleared).")
            
            elif command == 'query':
                if len(parts) < 2:
                    print("❌ Usage: query <your question>")
                    continue
                
                query = parts[1]
                print()
                result = orchestrator.process_query(query, show_routing=True)
                print(result['formatted_answer'])
            
            elif command == 'file':
                if len(parts) < 2:
                    print("❌ Usage: file <file_path> <question>")
                    continue
                
                # Parse file path and query
                file_and_query = parts[1].split(maxsplit=1)
                if len(file_and_query) < 2:
                    print("❌ Usage: file <file_path> <question>")
                    continue
                
                file_path = file_and_query[0]
                query = file_and_query[1]
                
                # Check if file exists
                if not Path(file_path).exists():
                    print(f"❌ File not found: {file_path}")
                    continue
                
                print()
                result = orchestrator.process_query(
                    query=query,
                    files=[file_path],
                    show_routing=True
                )
                print(result['formatted_answer'])
            
            elif command == 'multifile':
                if len(parts) < 2:
                    print("❌ Usage: multifile <path1,path2,...> <question>")
                    continue
                
                # Parse file paths and query
                files_and_query = parts[1].split(maxsplit=1)
                if len(files_and_query) < 2:
                    print("❌ Usage: multifile <path1,path2,...> <question>")
                    continue
                
                file_paths = [f.strip() for f in files_and_query[0].split(',')]
                query = files_and_query[1]
                
                # Check if all files exist
                missing_files = [f for f in file_paths if not Path(f).exists()]
                if missing_files:
                    print(f"❌ Files not found: {', '.join(missing_files)}")
                    continue
                
                print()
                result = orchestrator.process_query(
                    query=query,
                    files=file_paths,
                    show_routing=True
                )
                print(result['formatted_answer'])
            
            else:
                # Treat as direct query
                print()
                result = orchestrator.process_query(user_input, show_routing=True)
                print(result['formatted_answer'])
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using the Multi-Agent Assistant.")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for available commands.")


def demo_mode(orchestrator: MultiAgentOrchestrator):
    """Run demo mode with predefined queries"""
    
    print_banner()
    print("\n🎬 DEMO MODE - Running sample queries...\n")
    
    run_test_cases(orchestrator)
    
    print("\n💡 Demo completed! Use 'python main.py' for interactive mode.")


def main():
    """Main entry point"""
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
        print("   Some features may not work. Please set it in .env file or environment.")
        print()
    
    # Initialize the orchestrator
    try:
        orchestrator = MultiAgentOrchestrator(verbose=True)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_mode(orchestrator)
        elif sys.argv[1] == '--test':
            run_test_cases(orchestrator)
        elif sys.argv[1] == '--help':
            print_menu()
        else:
            # Treat as query
            query = ' '.join(sys.argv[1:])
            result = orchestrator.process_query(query, show_routing=True)
            print(result['formatted_answer'])
    else:
        # Interactive mode
        interactive_mode(orchestrator)


if __name__ == "__main__":
    main()