import os
import sys
import json
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our components
from app.model.model_handler import SoulMateModel
from app.memory.memory_handler import MemorySystem
from app.emotion.emotion_analyzer import EmotionAnalyzer
from app.learning.learning_engine import IncrementalLearner

def test_model():
    """Test the foundation model."""
    print("\n--- Testing Foundation Model ---")
    model = SoulMateModel("gpt2")
    
    test_input = "I'm feeling a bit down today."
    print(f"Input: {test_input}")
    
    response = model.generate_response(test_input)
    print(f"Response: {response}")
    
    print("Model test completed")
    return model

def test_memory(model):
    """Test the memory system."""
    print("\n--- Testing Memory System ---")
    memory = MemorySystem()
    
    # Add some sample memories
    memory.add_memory(
        "<User> I love hiking in the mountains. <SoulMate> That sounds wonderful! What's your favorite trail?",
        {"type": "conversation", "emotion": "joy"}
    )
    
    memory.add_memory(
        "<User> I'm feeling really stressed about my exam tomorrow. <SoulMate> I understand exam stress can be tough. Have you tried taking short breaks between study sessions?",
        {"type": "conversation", "emotion": "fear"}
    )
    
    # Update user profile
    memory.update_user_profile("personality_traits", "adventurous")
    memory.update_user_profile("interests", "hiking")
    
    # Test retrieval
    query = "I'm worried about my test"
    print(f"Query: {query}")
    
    relevant, profile = memory.get_relevant_context(query)
    print(f"Relevant memories: {len(relevant)}")
    print(f"First relevant memory: {relevant[0] if relevant else 'None'}")
    print(f"User profile: {profile}")
    
    # Test with the model
    response = model.generate_response(query, relevant, profile)
    print(f"Response with context: {response}")
    
    print("Memory test completed")
    return memory

def test_emotion():
    """Test the emotion analyzer."""
    print("\n--- Testing Emotion Analyzer ---")
    analyzer = EmotionAnalyzer()
    
    test_inputs = [
        "I'm so happy today!",
        "I'm feeling really anxious about my presentation.",
        "I'm really angry about what happened."
    ]
    
    for text in test_inputs:
        print(f"\nInput: {text}")
        sentiment = analyzer.analyze_sentiment(text)
        emotion = analyzer.analyze_emotion(text)
        
        print(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
        print(f"Primary emotion: {emotion['primary_emotion']}")
        print(f"Emotion scores: {emotion['emotions']}")
    
    # Test trends
    trends = analyzer.get_emotion_trends()
    print(f"\nEmotion trends: {trends}")
    
    current = analyzer.get_current_emotional_state()
    print(f"Current emotional state: {current}")
    
    print("Emotion test completed")
    return analyzer

def test_learning(model):
    """Test the incremental learning system."""
    print("\n--- Testing Incremental Learning ---")
    learner = IncrementalLearner(model.model_name)
    
    # Sample conversations for training
    conversations = [
        {
            "user_message": "I had a really tough day at work.",
            "ai_response": "I'm sorry to hear that. What happened?",
            "timestamp": datetime.now().isoformat(),
            "sentiment": "negative"
        },
        {
            "user_message": "I love going for walks in the park.",
            "ai_response": "That sounds lovely! What's your favorite park?",
            "timestamp": datetime.now().isoformat(),
            "sentiment": "positive"
        }
    ]
    
    # Prepare training data
    df = learner.prepare_training_data(conversations)
    print(f"Prepared {len(df)} training examples")
    
    # Simulate learning
    result = learner.perform_incremental_learning(conversations)
    print(f"Learning result: {result}")
    
    # Simulate personality evolution
    user_profile = {
        "personality_traits": ["kind", "thoughtful"],
        "interests": ["nature", "books"]
    }
    
    evolution = learner.simulate_personality_evolution(user_profile, days=30)
    print(f"Personality evolution after 30 days: {evolution}")
    
    print("Learning test completed")

def full_conversation_flow():
    """Test the full conversation flow."""
    print("\n--- Testing Full Conversation Flow ---")
    
    # Initialize components
    model = SoulMateModel("gpt2")
    memory = MemorySystem()
    emotion = EmotionAnalyzer()
    
    # Simulate a conversation
    conversation = [
        "Hi, I'm new here. My name is Alex.",
        "I've been feeling a bit lonely lately.",
        "I enjoy reading science fiction books.",
        "What do you think about meditation for stress relief?"
    ]
    
    for i, user_input in enumerate(conversation):
        print(f"\nUser: {user_input}")
        
        # Analyze emotion
        emotion_data = emotion.analyze_emotion(user_input)
        print(f"Detected emotion: {emotion_data['primary_emotion']}")
        
        # Get context
        relevant, profile = memory.get_relevant_context(user_input)
        
        # Generate response
        response = model.generate_response(user_input, relevant, profile)
        print(f"SoulMate: {response}")
        
        # Add to memory
        memory_entry = f"<User> {user_input}\n<SoulMate> {response}"
        memory.add_memory(memory_entry, {
            "type": "conversation", 
            "emotion": emotion_data["primary_emotion"],
            "turn": i + 1
        })
        
        # Update profile (simplified logic for demo)
        if "name is" in user_input.lower():
            # Extract name
            name = user_input.split("name is")[1].strip().split()[0].strip(".,")
            memory.update_user_profile("name", name)
            
        if "enjoy" in user_input.lower() or "like" in user_input.lower():
            # Extract potential interest
            memory.update_user_profile("interests", user_input.split("enjoy")[1].strip() if "enjoy" in user_input.lower() else user_input.split("like")[1].strip())
    
    # Print final profile
    print("\nFinal user profile:")
    print(json.dumps(memory.user_profile, indent=2))
    
    print("Full conversation test completed")

if __name__ == "__main__":
    print("SoulMate.AGI Test Suite")
    print("======================")
    
    model = test_model()
    memory = test_memory(model)
    test_emotion()
    test_learning(model)
    full_conversation_flow()
    
    print("\nAll tests completed successfully!")