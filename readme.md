# Clothing Search Application with LangChain and Ollama

This project is a Python-based application that uses **LangChain**, **LangChain-Ollama**, and the **Serper API** to analyze an image of a clothing item, describe it, generate a search query, and recommend online stores where similar items can be purchased. The application leverages local vision and text language models (via Ollama) and web search capabilities to provide a seamless clothing search experience.

## Features

- **Image Analysis**: Uses a vision model (`llava:13b`) to describe clothing items in an uploaded image.
- **Search Query Generation**: Generates concise search queries (up to three words) based on the image description using a text model (`qwen2.5`).
- **Web Search Integration**: Fetches relevant search results from the web using the Serper API.
- **Store Recommendations**: Recommends up to five online stores with links to purchase similar clothing items.
- **Workflow Orchestration**: Utilizes LangGraph to manage the multi-step process of image processing, query generation, searching, and recommending.

## Prerequisites

Before running the application, ensure you have the following:

- **Python 3.8+** installed.
- **[Ollama](https://ollama.ai/)** installed and running locally with the following models:
  - `qwen2.5:latest` (text model)
  - `llava:13b` (vision model)
- A **[Serper API key](https://serper.dev/)** for web search functionality.
- A GPU is recommended for faster processing with Ollama models.
- An image file (e.g., `red_skirt2.jpg`) in the project directory for testing.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/armanjscript/Clothing-Search-Application.git
   cd Clothing-Search-Application
