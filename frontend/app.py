import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px

# API endpoint
API_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="SoulMate.AGI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "personality_traits": [],
        "interests": [],
        "emotional_patterns": {}
    }
    
if "emotional_data" not in st.session_state:
    st.session_state.emotional_data = []
    
if "adaptation_level" not in st.session_state:
    st.session_state.adaptation_level = 0.0

# Function to send chat message
def send_message(message):
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error communicating with API: {str(e)}")
        return {"response": "I'm having trouble connecting to my brain right now."}

# Function to update profile
def update_profile(trait, value):
    try:
        response = requests.post(
            f"{API_URL}/profile/update",
            json={"trait": trait, "value": value}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error updating profile: {str(e)}")
        return {"status": "error"}

# Function to trigger learning
def trigger_learning(days=30):
    try:
        response = requests.post(
            f"{API_URL}/learn",
            json={"days_to_simulate": days}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error triggering learning: {str(e)}")
        return {"status": "error"}

# Function to get profile
def get_profile():
    try:
        response = requests.get(f"{API_URL}/profile")
        return response.json()["profile"]
    except Exception as e:
        st.error(f"Error getting profile: {str(e)}")
        return st.session_state.user_profile

# Function to get emotion analysis
def get_emotion_analysis():
    try:
        response = requests.get(f"{API_URL}/emotion/analysis")
        return response.json()
    except Exception as e:
        st.error(f"Error getting emotion analysis: {str(e)}")
        return {"trends": {}, "current_state": {}, "emotional_shifts": []}

# Sidebar
with st.sidebar:
    st.title("SoulMate.AGI")
    st.caption("Your personalized AI companion")
    
    # Display adaptation level
    st.subheader("Adaptation Level")
    st.progress(st.session_state.adaptation_level)
    st.caption(f"{st.session_state.adaptation_level*100:.1f}% personalized to you")
    
    # Profile section
    st.subheader("Your Profile")
    profile_traits = st.session_state.user_profile.get("personality_traits", [])
    profile_interests = st.session_state.user_profile.get("interests", [])
    
    if profile_traits:
        st.write("Personality traits:")
        for trait in profile_traits:
            st.write(f"- {trait}")
    
    if profile_interests:
        st.write("Interests:")
        for interest in profile_interests:
            st.write(f"- {interest}")
    
    if not profile_traits and not profile_interests:
        st.write("Your profile is still developing. Keep chatting!")
    
    # Learning section
    st.subheader("Night-time Learning")
    days = st.slider("Days to simulate", 1, 90, 30)
    if st.button("Simulate Learning"):
        with st.spinner("Learning in progress..."):
            result = trigger_learning(days)
            if result.get("status") == "success":
                evolution = result.get("evolution_projection", {})
                st.session_state.adaptation_level = evolution.get("adaptation_level", 0.0)
                st.success("Learning completed successfully!")
                
                # Update profile display
                new_profile = get_profile()
                st.session_state.user_profile = new_profile
                st.experimental_rerun()

# Main area
st.title("SoulMate.AGI")
st.subheader("Your personalized, evolving AI companion")

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="❤️"):
                st.write(message["content"])
                if "emotion" in message:
                    st.caption(f"Detected emotion: {message['emotion']}")

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response
    with st.spinner("Thinking..."):
        result = send_message(user_input)
        ai_response = result.get("response", "Sorry, I'm having trouble thinking right now.")
        emotion_detected = result.get("emotion_detected", "neutral")
    
    # Add AI response to chat
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_response,
        "emotion": emotion_detected
    })
    
    # Display AI response
    with st.chat_message("assistant", avatar="❤️"):
        st.write(ai_response)
        st.caption(f"Detected emotion: {emotion_detected}")
    
    # Track emotional data
    st.session_state.emotional_data.append({
        "emotion": emotion_detected,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Update profile
    new_profile = get_profile()
    st.session_state.user_profile = new_profile

# Tabs for additional features
tab1, tab2 = st.tabs(["Emotional Insights", "About SoulMate.AGI"])

with tab1:
    st.subheader("Your Emotional Journey")
    
    # Check if we have emotional data
    if st.session_state.emotional_data:
        # Create dataframe for visualization
        df = pd.DataFrame(st.session_state.emotional_data)
        
        # Count emotions
        emotion_counts = df["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        
        # Create pie chart
        fig = px.pie(emotion_counts, values="Count", names="Emotion", 
                    title="Emotions in Conversation")
        st.plotly_chart(fig)
        
        # Emotional timeline
        st.subheader("Emotional Timeline")
        timeline_fig = px.scatter(df, x="timestamp", y="emotion", 
                                color="emotion", title="Emotional Journey")
        st.plotly_chart(timeline_fig)
    else:
        st.write("Start chatting to see emotional insights!")

with tab2:
    st.subheader("About SoulMate.AGI")
    st.write("""
    SoulMate.AGI is a personalized, evolving companion that learns from your interactions.
    
    Key features:
    - 💬 Natural conversation with emotional understanding
    - 🧠 Continuous learning and adaptation to your personality
    - 🔄 Daily fine-tuning to better understand your needs
    - 🔒 Privacy-focused with encrypted memory storage
    
    This project was developed for the XHorizon 2025 Hackathon at MBU.
    """)
    
    st.subheader("How It Works")
    st.image("https://via.placeholder.com/800x400?text=SoulMate.AGI+Architecture", 
            caption="SoulMate.AGI System Architecture")