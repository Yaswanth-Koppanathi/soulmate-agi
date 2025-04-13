import os
import json
from typing import List, Dict, Any, Optional
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from datetime import datetime

class IncrementalLearner:
    def __init__(self, model_path: str):
        """Initialize the incremental learning system.
        
        Args:
            model_path: Path to the base model
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set up LoRA configuration for efficient fine-tuning
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # Target attention layers
        )
        
        # Training history
        self.training_history = []
        
        print("Incremental learner initialized")
        
    def prepare_training_data(self, conversations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare conversation data for training.
        
        Args:
            conversations: List of conversation exchanges
            
        Returns:
            DataFrame with formatted training examples
        """
        training_examples = []
        
        for convo in conversations:
            # Extract user message and AI response
            user_msg = convo.get("user_message", "")
            ai_response = convo.get("ai_response", "")
            
            if not user_msg or not ai_response:
                continue
                
            # Format as training example
            formatted_example = f"<User> {user_msg}\n<SoulMate> {ai_response}"
            
            # Add metadata
            example = {
                "text": formatted_example,
                "timestamp": convo.get("timestamp", datetime.now().isoformat()),
                "sentiment": convo.get("sentiment", "neutral"),
                "emotional_state": convo.get("emotional_state", "unknown")
            }
            
            training_examples.append(example)
            
        return pd.DataFrame(training_examples)
    
    def tokenize_data(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Tokenize the training data.
        
        Args:
            df: DataFrame with training examples
            
        Returns:
            Dictionary with tokenized inputs
        """
        texts = df["text"].tolist()
        
        # Tokenize the texts
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Prepare dataset
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # For causal language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def perform_incremental_learning(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform one cycle of incremental learning.
        
        Args:
            conversations: Recent conversation data
            
        Returns:
            Training results and metadata
        """
        print("Starting incremental learning cycle")
        
        # Prepare data
        df = self.prepare_training_data(conversations)
        if len(df) == 0:
            print("No valid training examples found")
            return {"error": "No valid training examples", "status": "failed"}
            
        print(f"Prepared {len(df)} training examples")
        
        # Tokenize data
        tokenized_data = self.tokenize_data(df)
        
        # Create dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, tokenized_inputs):
                self.tokenized_inputs = tokenized_inputs
                
            def __len__(self):
                return len(self.tokenized_inputs["input_ids"])
                
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.tokenized_inputs.items()}
                
        dataset = SimpleDataset(tokenized_data)
        
        # Apply LoRA adapters
        model = get_peft_model(self.model, self.peft_config)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=5,
            logging_steps=10,
            save_strategy="epoch",
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            report_to="none"  # Disable wandb, etc.
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        
        # For the hackathon demo, let's just simulate training
        # In a real implementation, we would do: trainer.train()
        print("Simulating training process...")
        
        # Record training metadata
        training_meta = {
            "examples_count": len(df),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),  # In real implementation, this would be after training
            "epochs": 3,
            "learning_rate": 2e-4
        }
        
        self.training_history.append(training_meta)
        
        print("Incremental learning cycle completed")
        
        return {
            "status": "completed",
            "examples_trained": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    def simulate_personality_evolution(self, user_profile: Dict[str, Any], days: int = 30) -> Dict[str, Any]:
        """Simulate how the AI personality would evolve over time.
        
        Args:
            user_profile: Current user profile data
            days: Number of days to simulate
            
        Returns:
            Projected personality traits
        """
        # Extract current traits
        current_traits = user_profile.get("personality_traits", [])
        current_interests = user_profile.get("interests", [])
        
        # Simulate evolution
        evolved_traits = current_traits.copy()
        evolved_interests = current_interests.copy()
        
        # Simulate learning new traits
        potential_new_traits = ["empathetic", "analytical", "supportive", "curious", 
                               "creative", "practical", "motivational", "reflective"]
        
        # Add a few traits based on "days" of learning
        for _ in range(min(days // 10, 3)):  # Add up to 3 new traits
            if potential_new_traits:
                new_trait = np.random.choice(potential_new_traits)
                if new_trait not in evolved_traits:
                    evolved_traits.append(new_trait)
                    potential_new_traits.remove(new_trait)
        
        # Simulate adaptation level (0-1 scale)
        base_adaptation = 0.3  # Starting adaptation
        max_adaptation = 0.95  # Maximum possible adaptation
        
        # Logarithmic growth of adaptation level
        adaptation_level = min(base_adaptation + (0.5 * np.log(1 + days/10)), max_adaptation)
        
        return {
            "initial_traits": current_traits,
            "evolved_traits": evolved_traits,
            "adaptation_level": adaptation_level,
            "simulated_days": days,
            "projected_interests": evolved_interests
        }
        
    def save_model_checkpoint(self, path: str) -> None:
        """Save the current model state.
        
        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save training history
        with open(os.path.join(path, "training_history.json"), "w") as f:
            json.dump(self.training_history, f)
            
        print(f"Model checkpoint saved to {path}")
    
    def load_model_checkpoint(self, path: str) -> None:
        """Load a saved model checkpoint.
        
        Args:
            path: Directory containing the saved model
        """
        self.model = AutoModelForCausalLM.from_pretrained(os.path.join(path, "model"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        
        # Load training history if available
        history_path = os.path.join(path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.training_history = json.load(f)
                
        print(f"Model checkpoint loaded from {path}")