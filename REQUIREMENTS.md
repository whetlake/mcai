# MCAI (My Custom AI Inference) Requirements Specification

## 1. Project Overview

MCAI is a Rust-based CLI application that manages and runs inference on GGUF (GPT-Generated Unified Format) language models. The application provides interactive chat capabilities between multiple models and a server mode for local API access.

### 1.1 Core Features

- Scan and detect GGUF models in specified directories
- Load and manage multiple GGUF models simultaneously
- Enable multi-model conversations and reasoning
- Provide CLI interface for model interaction
- Support both interactive chat and server modes
- REST API server for external integrations

## 2. Functional Requirements

### 2.1 Model Management

- **Model Discovery**
  - Automatically scan specified directories for `.gguf` files
  - Validate GGUF file format and metadata
  - Maintain a registry of available models
  - Display model information (size, architecture, parameters)

- **Model Selection**
  - Allow users to specify multiple models via CLI arguments
  - Support dynamic model loading/unloading
  - Provide model listing functionality

### 2.2 CLI Interface

#### Commands


1. **Run Command**

   Starts both server and interactive chat
   
   ```bash
   mcai run [OPTIONS]
   ```
   - `--models <name1,name2>`: Specify models to use
   - `--dir <path>`: Directory containing GGUF models
   - `--temp <float>`: Temperature setting for inference

2. **Serve Command**

   Starts only the API server

   ```bash
   mcai serve [OPTIONS]
   ```
   - `--models <name1,name2>`: Specify models to use
   - `--port <number>`: Server port (default: 3000)
   - `--host <address>`: Host address (default: localhost)

3. **List Command**
   ```bash
   mcai list
   ```
   - Lists all available models in the configured directory
   - Shows model details and status

### 2.3 Interactive Multi-Model Chat Mode

- Provide command prompt interface (">")
- Support multi-line input
- Display model responses with proper formatting and attribution
- Enable back-and-forth reasoning between models
- Support chat commands:
  - `/exit` or `exit`: Exit the application
  - `/models <name1,name2>`: Switch active models
  - `/help`: Show available commands
  - `/info`: Show current models information

### 2.4 API Server

#### Endpoints

1. **Generate Multi-Model Discussion**
   ```
   POST /generate/discussion
   ```
   - Request body:
     ```json
     {
       "models": ["string", "string"],
       "rounds": integer,
       "temperature": float (optional)
     }
     ```

2. **Generate Single Response**
   ```
   POST /generate
   ```
   - Request body:
     ```json
     {
       "prompt": "string",
       "model": "string",
       "temperature": float (optional)
     }
     ```

3. **List Models**
   ```
   GET /models
   ```
   - Returns list of available models and their status

4. **Model Info**
   ```
   GET /models/{name}
   ```
   - Returns detailed information about specific model

## 3. Non-Functional Requirements

### 3.1 Performance

- Model loading time < 5 seconds per model
- Response generation < 2 seconds for typical prompts
- Efficient coordination between multiple models
- Support concurrent API requests
- Efficient memory management for multiple loaded models

### 3.2 Reliability

- Graceful error handling for invalid models
- Automatic recovery from model loading failures
- Proper cleanup of resources on shutdown
- Input validation for all API endpoints
- Handling of model conversation deadlocks

### 3.3 Security

- Input sanitization for all user inputs
- Rate limiting for API endpoints
- Validation of GGUF files before loading
- Optional API authentication mechanism

## 4. Technical Requirements

### 4.1 Dependencies

- Rust 1.70 or higher
- Required crates:
  - `clap` for CLI argument parsing
  - `tokio` for async runtime
  - `reqwest` for HTTP client
  - `axum` or `warp` for HTTP server
  - Custom GGUF parsing implementation

### 4.2 Configuration

- Support for configuration file (`config.toml` or similar)
- Environment variable overrides
- Command-line argument precedence over config file
- Model interaction rules configuration

### 4.3 Logging

- Structured logging with different levels
- Optional file-based logging
- Performance metrics logging
- Error tracking and reporting
- Model conversation logging

## 5. Future Considerations

- WebSocket support for streaming responses
- Model fine-tuning capabilities
- Model download/update functionality
- GPU acceleration support
- Distributed inference support
- Chat history persistence
- Prompt templates and management
- Advanced reasoning protocols between models
- Specialized models for different reasoning tasks
- Visualization of model reasoning paths

## 6. Development Guidelines

- Follow Rust best practices and idioms
- Comprehensive error handling
- Unit tests for core functionality
- Integration tests for API endpoints
- Documentation for all public APIs
- Semantic versioning