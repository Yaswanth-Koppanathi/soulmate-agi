from typing import Dict, Any, List, Tuple
from transformers import pipeline
import numpy as np
from datetime import datetime

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analysis system."""
        # Load sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Load emotion classification pipeline 
        # This will detect more specific emotions
        self.emotion_classifier = pipeline("text-classification", 
                                          model="j-hartmann/emotion-english-distilroberta-base", 
                                          return_all_scores=True)
        
        # Emotion history tracking
        self.emotion_history = []
        
        print("Emotion analyzer initialized")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment information
        """
        result = self.sentiment_analyzer(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze the emotional content of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with emotion scores
        """
        # Get emotion scores
        emotions = self.emotion_classifier(text)[0]
        
        # Convert to a cleaner format
        emotion_scores = {item["label"]: float(item["score"]) for item in emotions}
        
        # Create the result with timestamp
        result = {
            "emotions": emotion_scores,
            "primary_emotion": max(emotion_scores.items(), key=lambda x: x[1])[0],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.emotion_history.append(result)
        if len(self.emotion_history) > 100:  # Limit history size
            self.emotion_history.pop(0)
            
        return result
    
    def get_emotion_trends(self, n_entries: int = 10) -> Dict[str, Any]:
        """Analyze trends in user emotions over time.
        
        Args:
            n_entries: Number of recent entries to analyze
            
        Returns:
            Dictionary with emotion trend analysis
        """
        if not self.emotion_history:
            return {"error": "No emotion history available"}
            
        # Get recent history
        recent = self.emotion_history[-min(n_entries, len(self.emotion_history)):]
        
        # Extract primary emotions
        primary_emotions = [entry["primary_emotion"] for entry in recent]
        
        # Count occurrences
        emotion_counts = {}
        for emotion in primary_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # Find most frequent emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        
        # Calculate emotional stability (lower value means more fluctuation)
        stability = 1.0
        if len(primary_emotions) > 1:
            changes = sum(1 for i in range(1, len(primary_emotions)) if primary_emotions[i] != primary_emotions[i-1])
            stability = 1.0 - (changes / (len(primary_emotions) - 1))
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "emotional_stability": stability,
            "analyzed_entries": len(recent)
        }
    
    def detect_emotional_shifts(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect significant shifts in emotional state.
        
        Args:
            threshold: The threshold for considering a shift significant
            
        Returns:
            List of detected emotional shifts
        """
        if len(self.emotion_history) < 2:
            return []
            
        shifts = []
        for i in range(1, len(self.emotion_history)):
            prev = self.emotion_history[i-1]
            curr = self.emotion_history[i]
            
            # Check if primary emotion changed
            if prev["primary_emotion"] != curr["primary_emotion"]:
                # Calculate the magnitude of the shift
                prev_score = prev["emotions"][prev["primary_emotion"]]
                curr_score = curr["emotions"][curr["primary_emotion"]]
                
                # If the shift is significant enough
                if abs(prev_score - curr_score) > threshold:
                    shifts.append({
                        "from_emotion": prev["primary_emotion"],
                        "to_emotion": curr["primary_emotion"],
                        "magnitude": abs(prev_score - curr_score),
                        "timestamp": curr["timestamp"]
                    })
                    
        return shifts
    
    def get_current_emotional_state(self) -> Dict[str, Any]:
        """Get the current emotional state based on recent history.
        
        Returns:
            Dictionary with current emotional state assessment
        """
        if not self.emotion_history:
            return {"state": "unknown", "confidence": 0.0}
            
        # Get the most recent emotion
        latest = self.emotion_history[-1]
        
        # Check for emotional consistency
        consistency = 1.0
        if len(self.emotion_history) > 1:
            recent = self.emotion_history[-min(5, len(self.emotion_history)):]
            primary_emotions = [entry["primary_emotion"] for entry in recent]
            dominant = max(set(primary_emotions), key=primary_emotions.count)
            consistency = primary_emotions.count(dominant) / len(primary_emotions)
            
        return {
            "current_emotion": latest["primary_emotion"],
            "confidence": float(latest["emotions"][latest["primary_emotion"]]),
            "emotional_consistency": consistency,
            "timestamp": latest["timestamp"]
        }