# SoulMate.AGI

A personalized, evolving AI companion trained by you, for you. This project was developed for the XHorizon 2025 Hackathon at MBU.

## Overview

SoulMate.AGI is an AI-native, user-trained general intelligence system that builds its personality, preferences, tone, and knowledge base directly from its owner's daily interactions. The system begins with a foundation language model and evolves into a deeply personalized companion over time via daily fine-tuning, embedding learning, and cloud-based incremental model adaptation.

## Features

- **Foundation AI Model**: Built on open-source LLMs with a context-persistent memory module
- **Personalized Interactions**: Adapts conversation style and emotional responses based on user interaction
- **Night-time Learning**: Simulates incremental training while you sleep
- **Emotional Intelligence**: Analyzes and responds to user emotions appropriately
- **Self-Evolving Personality**: Grows more personalized over time

## Technology Stack

- **Backend**: Python, FastAPI
- **AI/ML**: Hugging Face Transformers, PyTorch, PEFT/LoRA
- **Memory**: FAISS vector database
- **Frontend**: Streamlit

## Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/soulmate-agi.git
   cd soulmate-agi
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the backend API:
   ```
   cd app
   python main.py
   ```

4. In a new terminal, start the frontend:
   ```
   cd frontend
   streamlit run app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Project Structure

```
soulmate-agi/
├── app/
│   ├── main.py           # Main application entry point
│   ├── model/            # Foundation model and fine-tuning
│   ├── memory/           # Vector storage and retrieval
│   ├── emotion/          # Sentiment analysis 
│   └── learning/         # Incremental learning simulation
├── frontend/             # Streamlit UI
├── data/                 # Sample data and user data storage
└── requirements.txt      # Dependencies
```

## Hackathon Notes

This project is a prototype developed for the XHorizon 2025 Hackathon. Due to the 24-hour constraint, some features are simulated rather than fully implemented:

- **Incremental Learning**: The night-time learning process is simulated for demonstration purposes
- **Model Personalization**: Uses lightweight adaptation techniques rather than full fine-tuning
- **Cloud Processing**: Designed for cloud but runs locally for the hackathon demo

## Future Work

- **Multimodal Capabilities**: Extend to support voice and image inputs
- **Improved Emotional Intelligence**: More nuanced emotional understanding and response
- **Full Cloud Integration**: Implement proper cloud-based training pipeline
- **Enhanced Security**: Add encryption for user data and memory storage
- **Mobile App**: Develop companion mobile application for on-the-go access

## Team

- @Yaswanth-Koppanathi

## License

This project is licensed under the MIT License
