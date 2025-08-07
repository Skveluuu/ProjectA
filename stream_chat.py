#!/usr/bin/env python3
"""
Chat - A simple, working AI chat with Ollama.
"""

import asyncio
import sys
import time
import subprocess
import os
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.models.openai import OpenAIModel, AsyncOpenAI
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings

import nest_asyncio
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

nest_asyncio.apply()


# ============================================================================
# Graphiti (formerly Zep) Configuration
# ============================================================================

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# Configure Ollama LLM client for Graphiti
OLLAMA_BASE_URL = "http://ollama:11434/v1"
LLM_MODEL_NAME = "qwen3:8b" # This should match the model pulled in Ollama
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Pull this model in Ollama: ollama pull nomic-embed-text
EMBEDDING_DIM = 768 # Dimension for nomic-embed-text

llm_config = LLMConfig(
    api_key="ollama",  # Ollama doesn't require a real API key
    model=LLM_MODEL_NAME,
    small_model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

llm_client_graphiti = OpenAIClient(config=llm_config)

embedder_graphiti = OpenAIEmbedder(
    config=OpenAIEmbedderConfig(
        api_key="ollama",
        embedding_model=EMBEDDING_MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        base_url=OLLAMA_BASE_URL,
    )
)

cross_encoder_graphiti = OpenAIRerankerClient(
    client=llm_client_graphiti, config=llm_config
)

# ============================================================================
# Ollama Setup
# ============================================================================

class OllamaProvider(Provider[AsyncOpenAI]):
    """Custom provider for Ollama."""
    
    def __init__(self, base_url: str = "http://ollama:11434/v1"):
        self._base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")
    
    @property
    def client(self) -> AsyncOpenAI:
        return self._client
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def base_url(self) -> str:
        return self._base_url


def create_model(max_tokens: int = 500, temperature: float = 0.3) -> OpenAIModel:
    """Create Ollama model with configurable settings."""
    provider = OllamaProvider()
    settings = ModelSettings(temperature=temperature, max_tokens=max_tokens)
    return OpenAIModel("qwen3:8b", provider=provider, settings=settings)


@dataclass
class ChatDeps:
    """Simple dependencies."""
    usage: Usage
    session_id: str = "chat-session"
    graphiti_memory: Optional[Graphiti] = None # Changed from zep_memory


# ============================================================================
# Simple Chat
# ============================================================================

class SimpleChat:
    """Simple chat interface."""
    
    def __init__(self, max_tokens: int = 500, temperature: float = 0.3):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.agents = {
            'chat': self._create_agent(
                "You are a helpful AI assistant. Be concise."
            ),
            'search': self._create_agent(
                "You are a search specialist. Find information efficiently.",
                tools=[self.search_web]
            ),
            'analysis': self._create_agent(
                "You are an analytical expert. Provide focused analysis.",
                tools=[self.analyze_topic]
            )
        }
        self.current_agent = 'chat'
        self.usage = Usage()
        self.graphiti_client = Graphiti(
            NEO4J_URI, 
            NEO4J_USER, 
            NEO4J_PASSWORD,
            llm_client=llm_client_graphiti,
            embedder=embedder_graphiti,
            cross_encoder=cross_encoder_graphiti
        ) 
    
    def _create_agent(self, system_prompt: str, tools: list = []):
        """Create an agent with a given system prompt and tools."""
        agent = Agent[ChatDeps, str](
            create_model(self.max_tokens, self.temperature),
            system_prompt=system_prompt
        )
        for tool in tools:
            agent.tool(tool)
        return agent
    
    @staticmethod
    async def search_web(ctx: RunContext[ChatDeps], query: str) -> str:
        """Search for information."""
        return f"Search results for '{query}': [This would connect to a real search API in production]"
        
    @staticmethod
    async def analyze_topic(ctx: RunContext[ChatDeps], topic: str) -> str:
        """Analyze a topic in depth."""
        return f"Deep analysis of '{topic}': [This would provide detailed analysis in production]"
    
    async def get_response(self, message: str) -> tuple[str, float]:
        """Get response from current agent."""
        start_time = time.time()
        
        print(f"\n🤖 {self.current_agent.title()} Agent")
        print("💬 ", end="", flush=True)
        
        agent = self.agents[self.current_agent]
        
        # Create a Zep memory session
        session_id = "chat-session"
        
        # Add user message as an episode
        await self.graphiti_client.add_episode(
            name=f"User message - {session_id}",
            episode_body=message,
            source=EpisodeType.text,
            source_description="User chat input",
            reference_time=time.time()
        )

        # Optionally, retrieve context from Graphiti and add to prompt
        # For now, we'll keep it simple and just add the message.
        # Later, we can add logic here to fetch relevant memories via search
        # and inject them into the agent's prompt.

        deps = ChatDeps(usage=self.usage, graphiti_memory=self.graphiti_client)
        
        try:
            result = await agent.run(
                message, 
                deps=deps,
                usage_limits=UsageLimits(request_limit=100)
            )
            
            # Print the complete response
            response = result.output
            print(response)
            
            # Add AI response as an episode
            await self.graphiti_client.add_episode(
                name=f"AI response - {session_id}",
                episode_body=response,
                source=EpisodeType.text,
                source_description="AI chat output",
                reference_time=time.time()
            )

            # Update usage
            self.usage = result.usage()
            
            response_time = time.time() - start_time
            return response, response_time
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            response_time = time.time() - start_time
            return error_msg, response_time
    
    def switch_agent(self, agent_name: str) -> bool:
        """Switch to different agent."""
        if agent_name in self.agents:
            self.current_agent = agent_name
            return True
        return False
    
    def set_speed_mode(self, mode: str) -> bool:
        """Set speed optimization mode."""
        if mode == "fast":
            self.max_tokens = 250
            self.temperature = 0.1
        elif mode == "balanced":
            self.max_tokens = 500
            self.temperature = 0.3
        elif mode == "quality":
            self.max_tokens = 1000
            self.temperature = 0.7
        else:
            return False
        
        # Recreate agents with new settings
        self.agents = {
            'chat': self._create_agent(
                "You are a helpful AI assistant. Be concise."
            ),
            'search': self._create_agent(
                "You are a search specialist. Find information efficiently.",
                tools=[self.search_web]
            ),
            'analysis': self._create_agent(
                "You are an analytical expert. Provide focused analysis.",
                tools=[self.analyze_topic]
            )
        }
        return True
    
    def show_info(self, topic: str):
        """Show help or status information."""
        print("\n" + "="*50)
        if topic == "help":
            print("🎯 CHAT COMMANDS")
            print("="*50)
            print("  /agent <name>  - Switch agent (chat, search, analysis)")
            print("  /speed <mode>  - Set speed mode (fast, balanced, quality)")
            print("  /status       - Show current status")
            print("  /help         - Show this help")
            print("  /quit         - Exit")
        elif topic == "status":
            print("📊 STATUS")
            print("="*50)
            print(f"  Current Agent: {self.current_agent}")
            print(f"  Max Tokens: {self.max_tokens}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Total Requests: {self.usage.requests or 0}")
            print(f"  Total Tokens: {self.usage.total_tokens or 0}")
        print("="*50)
    
    def _handle_command(self, user_input: str) -> bool:
        """Handle user commands."""
        parts = user_input.split()
        command = parts[0].lower()
        
        if command == '/quit':
            print("\n👋 Goodbye!")
            return False
        elif command == '/help':
            self.show_info("help")
        elif command == '/status':
            self.show_info("status")
        elif command == '/agent' and len(parts) > 1:
            agent_name = parts[1].lower()
            if self.switch_agent(agent_name):
                print(f"✅ Switched to {agent_name}")
            else:
                print(f"❌ Unknown agent. Available: {', '.join(self.agents.keys())}")
        elif command == '/speed' and len(parts) > 1:
            mode = parts[1].lower()
            if self.set_speed_mode(mode):
                print(f"⚡ Speed mode set to {mode}")
                print(f"   Max tokens: {self.max_tokens}, Temperature: {self.temperature}")
            else:
                print(f"❌ Unknown speed mode. Available: fast, balanced, quality")
        else:
            print(f"❌ Unknown command. Type /help for available commands.")
        return True

    def _print_welcome(self):
        """Print welcome message."""
        print("\n" + "="*50)
        print("🚀 CHAT")
        print("="*50)
        print("AI chat with local Ollama")
        print("Type /help for commands")
        print("="*50)

    async def run(self):
        """Run the chat."""
        self._print_welcome()

        # Ensure Graphiti indices are built on startup
        await self.graphiti_client.build_indices_and_constraints()
        
        try:
            while True:
                try:
                    user_input = input(f"\nYou ({self.current_agent}): ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # Get response and show metrics
                response, response_time = await self.get_response(user_input)
                print(f"\n\n⏱️  {response_time:.2f}s")
                if self.usage.total_tokens:
                    print(f"📊 Tokens: {self.usage.total_tokens}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")


# ============================================================================
# Ollama Check & Main
# ============================================================================

def check_ollama():
    """Check if Ollama is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("\n❌ Ollama not running!")
            print("\n1. Start Ollama: ollama serve")
            print("2. Pull model: ollama pull qwen3:8b")
            print("3. Run this chat again")
            return False
    except Exception:
        print("⚠️  Could not check Ollama status")
    
    # Check model
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        if 'qwen3:8b' not in result.stdout.lower():
            print("\n⚠️  Model not found. Pulling qwen3:8b...")
            subprocess.run(['ollama', 'pull', 'qwen3:8b'])
            print("✅ Model ready!")
    except Exception:
        print("⚠️  Could not verify model")
    
    return True


def main():
    """Main entry point."""
    if not check_ollama():
        sys.exit(1)
    
    chat = SimpleChat()
    asyncio.run(chat.run())


if __name__ == "__main__":
    main()