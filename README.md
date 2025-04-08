# MCAI-Rust: Local GGUF Model Chat & Server

**Run GGUF language models locally with a versatile server and interactive chat client.**

This project provides a robust solution for hosting and interacting with GGUF-formatted language models on your own machine, leveraging the power of Rust and the `llama-cpp-rs` library for efficient inference. It serves as the foundation for building custom AI applications.

---

## ⚠️ Disclaimer

**This is a personal project primarily meant for learning purposes.** While functional, it comes with **no guarantees** regarding:

*   **Maintenance:** Updates may be infrequent or cease altogether.
*   **Support:** No official support is provided.
*   **Stability:** Use at your own risk. Breaking changes may occur and will occur.

Feel free to fork, experiment, and learn from it, but please be aware of its nature as a personal educational endeavor.

However, if you do want to discuss, feel free to get in touch via github.

---

## Overview

MCAI-Rust consists of two core components:

1.  **🚀 `mcai-server`:** A backend server that dynamically scans a directory for GGUF models, loads them into memory (potentially GPU-accelerated), manages concurrent model instances, and exposes a RESTful API for interaction. It uses llama_cpp rust package underneath that connects to the great llama.cpp. The idea of the server is that any service can build on top of it and the mcai-chat is not actually required.
2.  **💬 `mcai-chat`:** An interactive command-line (REPL) client that connects to the `mcai-server`. It allows users to easily manage loaded model instances (attach, detach, rename, switch context), view model details, and engage in conversations with the AI models using a familiar chat interface.

This setup provides flexibility, allowing multiple clients or applications to potentially interact with the same backend server hosting the models.

## ✨ Features

*   **Local Model Hosting:** Run GGUF models entirely on your hardware. Make sure your hardware can handle it.
*   **Server/Client Architecture:** Decouples model serving from the user interface.
*   **Dynamic Model Registry:** Automatically discovers GGUF models in a specified directory.
*   **Concurrent Model Instances:** Load and interact with multiple models or multiple instances of the same model simultaneously.
*   **GPU Acceleration:** Supports offloading computation to NVIDIA GPUs (via CUDA), Apple Silicon GPUs (via Metal), and potentially others depending on `llama_cpp` build features. Configurable via `config/default.toml` and `config/local.toml`.
*   **Interactive Chat Client:** Feature-rich REPL for easy model management and interaction:
    *   List available and loaded models.
    *   Attach models from the registry by number.
    *   Assign custom, unique labels to loaded instances.
    *   Switch the active chat context between loaded instances.
    *   Rename loaded instances using labels or UUIDs.
    *   Drop instances (unload from memory) by identifier or implicitly drop the active one.
    *   Detach the active context without unloading the model.
    *   View detailed model metadata and tensor information.
    *   Streaming responses for smooth text generation.
    *   Interrupt ongoing generation (`Ctrl+C`) without exiting the client.
*   **Configuration:** Manage server, model directory, and inference settings via `config/default.toml` (defaults) and `config/local.toml` (user overrides).

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
# Make sure to replace <your-repo-url> with the actual URL
git clone <your-repo-url>
cd mcai-rust

# Build the project (use --release for optimized binaries)
# This might take a while, especially the first time, as it compiles llama-cpp-rs
cargo build --release

# Optional: Enable GPU features during build (example for CUDA)
# Refer to llama-cpp-rs docs for specific feature flags (cuda, metal, etc.)
# cargo build --release --features cuda
```

Binaries (`mcai-rust`) will be in the `./target/release/` directory.

### 2. Configuration

Configuration is handled via files in the `config/` directory:

*   `config/default.toml`: Contains the default settings for the application. **Do not edit this directly.**
*   `config/local.toml`: **Create this file** (you can copy `config/local.toml.example`) to override specific default settings. Settings in `local.toml` take precedence.

Key settings to potentially override in `config/local.toml`:

*   **`models_dir`**: **Crucial!** Set this to the path where you stored your GGUF models.
*   `server.host` / `server.port`: Adjust if needed.
*   `inference.n_gpu_layers`: Set the number of layers to offload to GPU (e.g., `35`, `99` for full offload if VRAM allows). Set to `0` for CPU-only inference.
*   Review other inference parameters (`use_mmap`, `use_mlock`, context size, etc.) in `config/default.toml` and override them in `config/local.toml` if necessary.

### 3. Run the Server

The server must be running before the chat client can connect.

```bash
# From the project root directory
./target/release/mcai-rust server
# Or via cargo: cargo run --release -- server
```

The server will scan the `models_dir`, report any found models, and listen for API requests on the configured host/port.

### 4. Run the Chat Client

Open another terminal in the same project root directory.

```bash
# From the project root directory
./target/release/mcai-rust chat
# Or via cargo: cargo run --release -- chat
```

You'll see the welcome message and the `>` prompt, indicating no model context is active yet.

## 💬 Using the Chat Interface

The chat interface is the primary way to interact with your models.

**Command Reference:**

*   **General:**
    *   `help` / `mcai help`: Shows available commands.
    *   `clear` / `mcai clear`: Clears the terminal screen.
    *   `exit` / `bye` / `quit`: Exits the chat client (when detached).
    *   `mcai exit` / `mcai bye` / `mcai quit`: Exits the chat client (when attached).

*   **Model Management:**
    *   `models` / `mcai models`: Lists models available in the registry.
    *   `attached` / `mcai attached`: Lists model instances currently loaded.
    *   `attach new <#> [label]` / `mcai attach new <#> [label]`: Loads model `<#>` from the registry, optionally assigns `[label]`, and activates its context.
    *   `attach <id>` (Detached only): Activates context for an already loaded instance (`<id>` = label or UUID).
    *   `mcai switch <id>` (Attached only): Switches active context to instance `<id>`.
    *   `drop <id>` (Detached only): Unloads instance `<id>` from memory.
    *   `mcai drop [id]` (Attached only): Unloads instance `<id>` (or active instance if `[id]` is omitted).
    *   `mcai detach` (Attached only): Deactivates the current context but keeps the model loaded.
    *   `rename <old_id> <new>` (Detached only): Renames instance `<old_id>` to `<new>` label.
    *   `mcai rename [old_id] <new>` (Attached only): Renames instance `<old_id>` (or active instance if `[old_id]` is omitted) to `<new>` label.

*   **Interaction & Info:**
    *   **(Direct Prompt)** (Attached only): Type your prompt and press Enter.
    *   `metadata <id>` / `mcai metadata [id]`: Shows metadata for instance `<id>` (or active instance if `[id]` is omitted when attached).
    *   `tensors <id>` / `mcai tensors [id]`: Shows tensors for instance `<id>` (or active instance if `[id]` is omitted when attached).

**(See `/docs` for more detailed usage examples - *Note: Docs folder is planned*)**

## Future Considerations

*   Implementing functionality to download models directly.
*   Developing a web-based user interface.
*   Integrating a database (e.g., PostgreSQL) for potential RAG system capabilities.

## Discussion

While this is a personal learning and testing project with no guarantees (see Disclaimer), feel free to open GitHub issues to discuss ideas, bugs, or potential improvements. No promises are made regarding implementation or response times. Its a hobby.
