# SoulMate.AGI - Quick Start Guide

This guide will help you quickly set up and demonstrate the SoulMate.AGI project for your hackathon presentation.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Running the Tests

To ensure everything is working correctly, run the test script:

```bash
python test.py
```

This will test all core components individually and then simulate a conversation flow.

## Step 3: Starting the API

Start the FastAPI backend service:

```bash
cd app
python main.py
```

The API will be available at `http://localhost:8000`.

## Step 4: Starting the Frontend

In a new terminal window (make sure the virtual environment is activated), start the Streamlit frontend:

```bash
cd frontend
streamlit run app.py
```

This will automatically open a browser window with the SoulMate.AGI interface at `http://localhost:8501`.

## Step 5: Demonstration Flow

Here's a suggested flow for demonstrating SoulMate.AGI during your presentation:

1. **Introduction**:
   - Explain the concept of a personalized AI companion that learns from interactions
   - Highlight the key features: emotional understanding, continuous learning, personalization

2. **Basic Interaction**:
   - Start with a simple greeting: "Hi, I'm [Your Name]"
   - Ask about the AI's capabilities: "What can you help me with?"
   - Share an interest: "I really enjoy hiking in the mountains"

3. **Emotional Response**:
   - Express an emotion: "I'm feeling a bit stressed about this presentation"
   - Note how SoulMate responds with empathy

4. **Simulate Learning**:
   - Navigate to the Night-time Learning section in the sidebar
   - Set the slider to 30 days and click "Simulate Learning"
   - Point out how the adaptation level increases and new traits appear

5. **Emotional Analysis**:
   - Click on the "Emotional Insights" tab
   - Show how SoulMate has tracked emotional patterns
   - Explain how this helps the AI better understand the user over time

6. **Personalized Response**:
   - Return to the chat and ask something related to the interest you mentioned earlier
   - Point out how responses are now tailored to your established interests

7. **Conclusion**:
   - Summarize how SoulMate.AGI demonstrates continuous learning and personalization
   - Explain the potential future applications and enhancements

## Technical Demo Points

If you're presenting to a technical audience, highlight these aspects:

- **Foundation Model**: Explain the use of transformer-based LLMs as the foundation
- **Vector Memory**: Point out how the system uses vector embeddings to recall relevant past interactions
- **Incremental Learning**: Describe the LoRA/PEFT approach for efficient fine-tuning
- **Emotion Analysis**: Showcase the multi-dimensional emotion classification

## Troubleshooting

If you encounter any issues:

- **Model loading errors**: Make sure you have enough RAM (at least 8GB recommended)
- **API connection errors**: Check that both the API and frontend are running
- **Slow responses**: The first few responses might be slow as models load into memory

Remember, this is a prototype built in 24 hours - focus on demonstrating the concept rather than perfect functionality!