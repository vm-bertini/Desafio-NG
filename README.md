# MCP-SOM TSP Solver

A sophisticated implementation that combines **Model Context Protocol (MCP)**, **Self-Organizing Maps (SOM)**, and **OpenAI GPT** integration to solve Traveling Salesman Problems (TSP) through natural language interaction.

## Overview

This project demonstrates an innovative approach to solving TSP problems by leveraging:
- **Model Context Protocol (MCP)** for seamless client-server communication
- **Self-Organizing Maps (SOM)** neural networks for TSP optimization
- **OpenAI GPT-4** integration for natural language processing and tool orchestration
- Interactive chat interface for problem-solving through conversation

## Architecture

### Core Components

1. **MCP Server** (`Class/MCP_Server_Class.py`)
   - Implements FastMCP server with TSP-specific tools
   - Provides SOM processing capabilities
   - Handles TSP file loading and city generation
   - Graph visualization functionality

2. **MCP Client** (`Class/MCP_Client_Class.py`)
   - Connects to MCP server via stdio transport
   - Integrates with OpenAI GPT-4 for intelligent tool usage
   - Manages interactive chat sessions
   - Handles tool calling and response processing

3. **SOM Neural Network** (`SOM/SOM.py`)
   - Custom PyTorch implementation of Self-Organizing Maps
   - Specialized for TSP route optimization
   - Supports both circular and linear topologies
   - Adaptive learning rate and neighborhood functions

4. **TSP File Reader** (`Functions/Read_TSP.py`)
   - Parses standard TSP file formats
   - Converts coordinate data to PyTorch tensors
   - Supports various TSP problem instances

## Features

### MCP Tools Available

- **`Process_Map`**: Processes city coordinates using SOM algorithm for TSP solving
- **`Load_TSP_File`**: Loads TSP instances from the Local_TSP directory
- **`gen_cities`**: Generates random city coordinates for testing
- **`graph_cities`**: Creates visualizations of city distributions and solutions

### SOM Capabilities

- Adaptive neural network topology for TSP optimization
- Configurable circular topology for closed tours
- Dynamic learning rate and neighborhood radius decay
- Efficient batch processing of city coordinates

### OpenAI Integration

- GPT-4 powered natural language interface
- Intelligent tool selection and parameter inference
- Conversational problem-solving workflow
- Automatic error handling and retry logic

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Quick Start

1. **Start the MCP Server** (in one terminal)
    ```bash
    python Server.py
    ```

2. **Run the main client interface** (in a separate terminal)
    ```bash
    python Main.py
    ```

3. **Example conversation**
   ```
   Query: Load the berlin52 TSP file and solve it using SOM
   
   Query: Generate 20 random cities and find an optimal tour
   
   Query: Show me a graph of the cities and solution
   ```

## Project Structure

```
NG/
├── Main.py                 # Entry point - runs integrated system
├── Server.py              # MCP server standalone runner
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── Class/                 # Core MCP implementation
│   ├── MCP_Client_Class.py    # MCP client with OpenAI integration
│   └── MCP_Server_Class.py    # MCP server with TSP tools
│
├── SOM/                   # Self-Organizing Map implementation
│   └── SOM.py                 # Custom PyTorch SOM for TSP
│
├── Functions/             # Utility functions
│   └── Read_TSP.py           # TSP file parser
│
├── Local_TSP/             # TSP problem instances
│   ├── berlin52.tsp          # Classic 52-city problem
│   ├── att48.tsp             # 48-city problem
│   └── ...                   # 100+ standard TSP instances
│
└── PNGs/                  # Generated visualizations
    ├── cities_graph.png      # City distribution plots
    ├── tsp_solution.png      # Solution visualizations
    └── ...
```

## Dependencies

- `mcp>=1.0.0` - Model Context Protocol framework
- `torch>=2.0.0` - PyTorch for SOM neural networks
- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable management
- `pandas>=2.0.0` - Data manipulation for TSP files
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization and plotting

## Contributing

This project demonstrates advanced integration of neural networks, protocol communication, and AI-powered interfaces for optimization problems.

