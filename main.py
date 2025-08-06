import sys
import requests
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Multiple output schemas for different use cases
class ChatOutputSchema(BaseModel):
    thinking: str = Field(description="Your reasoning process and thought steps")
    message: str = Field(description="Your main response to the user")
    toolCalls: List[dict] = Field(default=[], description="Any tool calls you want to make")

class SearchOutputSchema(BaseModel):
    thinking: str = Field(description="Your reasoning about the search query")
    search_results: List[str] = Field(description="Relevant search results")
    summary: str = Field(description="Summary of the search results")
    sources: List[str] = Field(description="Source URLs")

class AnalysisOutputSchema(BaseModel):
    thinking: str = Field(description="Your analytical reasoning")
    key_points: List[str] = Field(description="Key points from the analysis")
    conclusion: str = Field(description="Your conclusion")
    confidence: float = Field(description="Confidence level (0-1)")

def create_chat_agent(model_name="qwen3:8b"):
    """Create a PydanticAI agent for general chat."""
    return Agent(
        model_name,
        system_prompt="""You are a helpful AI assistant. Always provide thoughtful responses that include:
1. Your reasoning process (thinking)
2. Your main response to the user (message)
3. Any tool calls if needed (toolCalls)

Be helpful, accurate, and engaging in your responses.""",
        output_type=ChatOutputSchema,
    )

def create_search_agent(model_name="qwen3:8b"):
    """Create a PydanticAI agent with search capabilities."""
    return Agent(
        model_name,
        tools=[duckduckgo_search_tool()],
        system_prompt="""You are a research assistant. When users ask for current information or need to search:
1. Use the search tool to find relevant information
2. Analyze the search results
3. Provide a comprehensive summary with sources

Always include your reasoning process.""",
        output_type=SearchOutputSchema,
    )

def create_analysis_agent(model_name="qwen3:8b"):
    """Create a PydanticAI agent for analytical tasks."""
    return Agent(
        model_name,
        system_prompt="""You are an analytical AI assistant. When analyzing topics:
1. Break down complex topics into key points
2. Provide evidence-based conclusions
3. Assess your confidence level
4. Explain your reasoning process

Be thorough and objective in your analysis.""",
        output_type=AnalysisOutputSchema,
    )

def chat_with_model(model_name="qwen3:8b"):
    """Start a chat dialog with multiple agent types."""
    print(f"Starting enhanced chat with {model_name}")
    print("Available modes: chat, search, analysis")
    print("Commands: 'mode <type>', 'quit', 'help'")
    print("-" * 50)
    
    # Initialize agents
    chat_agent = create_chat_agent(model_name)
    search_agent = create_search_agent(model_name)
    analysis_agent = create_analysis_agent(model_name)
    
    current_agent = chat_agent
    current_mode = "chat"
    
    try:
        while True:
            # Get user input
            user_input = input(f"\nYou ({current_mode}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower().startswith('mode '):
                mode = user_input[5:].lower()
                current_agent, current_mode = switch_mode(mode, chat_agent, search_agent, analysis_agent)
                continue
            
            if not user_input:
                continue
            
            # Get response from model using PydanticAI
            try:
                result = current_agent.run_sync(user_input)
                
                # Display structured output based on mode
                display_result(result, current_mode)
                
            except Exception as e:
                print(f"\nError getting response: {e}")
                print("Make sure Ollama is running and the model is available.")
                break
                
    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

def switch_mode(mode, chat_agent, search_agent, analysis_agent):
    """Switch between different agent modes."""
    if mode == "chat":
        print("Switched to chat mode")
        return chat_agent, "chat"
    elif mode == "search":
        print("Switched to search mode")
        return search_agent, "search"
    elif mode == "analysis":
        print("Switched to analysis mode")
        return analysis_agent, "analysis"
    else:
        print(f"Unknown mode: {mode}. Available modes: chat, search, analysis")
        return chat_agent, "chat"

def display_result(result, mode):
    """Display results based on the current mode."""
    if mode == "chat":
        print(f"\nAssistant Thinking: {result.output.thinking}")
        print(f"\nAssistant Message: {result.output.message}")
        if result.output.toolCalls:
            print(f"\nTool Calls: {result.output.toolCalls}")
    
    elif mode == "search":
        print(f"\nSearch Thinking: {result.output.thinking}")
        print(f"\nSearch Results: {result.output.search_results}")
        print(f"\nSummary: {result.output.summary}")
        print(f"\nSources: {result.output.sources}")
    
    elif mode == "analysis":
        print(f"\nAnalysis Thinking: {result.output.thinking}")
        print(f"\nKey Points: {result.output.key_points}")
        print(f"\nConclusion: {result.output.conclusion}")
        print(f"\nConfidence: {result.output.confidence:.2f}")

def print_help():
    """Print help information."""
    print("\n" + "="*50)
    print("ENHANCED CHAT WITH PYDANTICAI FEATURES")
    print("="*50)
    print("Commands:")
    print("  mode chat     - Switch to general chat mode")
    print("  mode search   - Switch to search mode (with web search)")
    print("  mode analysis - Switch to analytical mode")
    print("  help          - Show this help")
    print("  quit          - Exit the chat")
    print("\nFeatures:")
    print("  • Structured outputs with thinking process")
    print("  • Web search integration")
    print("  • Analytical reasoning with confidence scores")
    print("  • Multiple specialized agents")
    print("  • Tool calling capabilities")
    print("="*50)

if __name__ == "__main__":
    # Check if a model name was provided as command line argument
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    
    print("Enhanced Chat with Local Model using PydanticAI")
    print("=" * 55)
    
    # Start the chat
    chat_with_model(model_name)