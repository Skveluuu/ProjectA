# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the dependency files and README
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --no-cache

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["uv", "run", "stream-chat"]
