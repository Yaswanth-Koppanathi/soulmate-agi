import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SoulMateModel:
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the SoulMate foundation model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add a special token for the AI
        special_tokens = {"additional_special_tokens": ["<SoulMate>", "<User>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        
    def generate_response(self, user_input: str, conversation_history: List[str] = None, user_profile: Dict[str, Any] = None) -> str:
        """Generate a response based on user input and conversation history.
        
        Args:
            user_input: The latest user message
            conversation_history: List of previous exchanges
            user_profile: User preferences and personality traits
            
        Returns:
            Generated response as string
        """
        if conversation_history is None:
            conversation_history = []
            
        # Format the prompt with history and user input
        formatted_prompt = self._format_prompt(user_input, conversation_history, user_profile)
        
        # Tokenize and generate
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"],
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract just the response part
        ai_response = self._extract_response(generated_text, formatted_prompt)
        
        return ai_response
    
    def _format_prompt(self, user_input: str, conversation_history: List[str], user_profile: Optional[Dict[str, Any]]) -> str:
        """Format the prompt with conversation history and user profile information."""
        # Start with an empty prompt
        prompt = ""
        
        # Add conversation history
        if conversation_history:
            for entry in conversation_history[-5:]:  # Limit to last 5 exchanges to avoid context length issues
                prompt += entry + "\n"
                
        # Add user profile context if available
        if user_profile:
            # Extract key personality traits and preferences
            traits = user_profile.get("personality_traits", [])
            interests = user_profile.get("interests", [])
            
            # Add a subtle hint about user preferences to guide the model
            if traits or interests:
                prompt += "<SoulMate> (Remember: "
                if traits:
                    prompt += f"The user tends to be {', '.join(traits)}. "
                if interests:
                    prompt += f"The user enjoys {', '.join(interests)}. "
                prompt += "Adapt your responses accordingly) \n"
        
        # Add the current exchange
        prompt += f"<User> {user_input}\n<SoulMate>"
        
        return prompt
    
    def _extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract just the model's response from the generated text."""
        # Remove the original prompt to get just the new generated content
        response = generated_text[len(original_prompt):]
        
        # Find the end of the response (next user turn or end of text)
        end_marker = "<User>"
        if end_marker in response:
            response = response.split(end_marker)[0]
            
        return response.strip()
    
    def save_model(self, path: str) -> None:
        """Save the current model state."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a saved model state."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")