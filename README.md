# Graphiti Agent with Neo4j and Ollama

This project demonstrates how to build a sophisticated AI agent using Graphiti for long-term memory, Neo4j for graph-based data storage, and Ollama for local LLM inference, all orchestrated with Docker Compose.

## Architecture

The agent's architecture is designed to be modular and scalable, with the following key components:

-   **Graphiti**: An open-source temporal knowledge graph framework that serves as the memory foundation, learning from user interactions and building a knowledge graph.
-   **Neo4j**: A graph database that serves as the persistent storage layer for Graphiti, enabling complex relationship-based queries.
-   **Ollama**: A local LLM runner that powers the agent's conversational abilities.
-   **Docker Compose**: Containerizes and orchestrates all services, simplifying setup and deployment.
-   **uv**: A fast, modern Python package manager.

## Getting Started

1.  **Launch the Services**: Start all services using Docker Compose:

    ```bash
    docker compose up --build
    ```

    This will build the agent's Docker image and start the Neo4j and Ollama containers.

2.  **Pull the LLM Model**: Once the services are running, you'll need to pull the language model for Ollama. Open a new terminal and run:

    ```bash
    docker compose exec ollama ollama pull qwen3:8b
    ```

3.  **Interact with the Agent**: The agent is now running and accessible. You can interact with it through the terminal where you launched Docker Compose.

## Development

-   **View Logs**: To see the logs from all services, you can use:
    ```bash
    docker compose logs -f
    ```
-   **Rebuild the Agent**: If you make changes to the agent's Python code, you'll need to rebuild the Docker image:
    ```bash
    docker compose up --build -d chat-agent
    ```