import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer

class MemorySystem:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the memory system with a vector database.
        
        Args:
            embedding_model: The sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Memory storage
        self.memories = []
        self.metadata = []
        
        # User profile
        self.user_profile = {
            "personality_traits": [],
            "interests": [],
            "emotional_patterns": {},
            "conversation_preferences": {},
            "created_at": datetime.now().isoformat()
        }
        
        print(f"Memory system initialized with embedding dimension {self.embedding_dim}")
    
    def add_memory(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a new memory with metadata.
        
        Args:
            text: The text content to remember
            metadata: Additional information about this memory
        """
        # Create embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store the text and metadata
        self.memories.append(text)
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
            
        self.metadata.append(metadata)
    
    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories based on a query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of memory entries with their metadata
        """
        if not self.memories:
            return []
            
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search the index
        k = min(k, len(self.memories))  # Don't request more results than we have
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):  # Safety check
                results.append({
                    "text": self.memories[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })
                
        return results
    
    def update_user_profile(self, attribute: str, value: Any) -> None:
        """Update an attribute in the user profile.
        
        Args:
            attribute: The profile attribute to update
            value: The new value
        """
        if attribute in self.user_profile:
            if isinstance(self.user_profile[attribute], list):
                if value not in self.user_profile[attribute]:
                    self.user_profile[attribute].append(value)
            elif isinstance(self.user_profile[attribute], dict):
                self.user_profile[attribute].update(value)
            else:
                self.user_profile[attribute] = value
        else:
            self.user_profile[attribute] = value
    
    def get_relevant_context(self, query: str, k: int = 3) -> Tuple[List[str], Dict[str, Any]]:
        """Get relevant conversation history and the current user profile.
        
        Args:
            query: The current query to find relevant context for
            k: Number of relevant memories to retrieve
            
        Returns:
            Tuple of (relevant_memories, user_profile)
        """
        relevant_memories = self.search_memories(query, k)
        return [m["text"] for m in relevant_memories], self.user_profile
    
    def save_memory(self, path: str) -> None:
        """Save the memory system to disk.
        
        Args:
            path: Directory to save memory data
        """
        os.makedirs(path, exist_ok=True)
        
        # Save user profile
        with open(os.path.join(path, "user_profile.json"), "w") as f:
            json.dump(self.user_profile, f)
        
        # Save memories and metadata
        memory_data = {
            "texts": self.memories,
            "metadata": self.metadata
        }
        with open(os.path.join(path, "memories.json"), "w") as f:
            json.dump(memory_data, f)
            
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "memory_index.faiss"))
        
        print(f"Memory saved to {path}")
    
    def load_memory(self, path: str) -> None:
        """Load memory system from disk.
        
        Args:
            path: Directory to load memory data from
        """
        # Load user profile
        profile_path = os.path.join(path, "user_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                self.user_profile = json.load(f)
        
        # Load memories and metadata
        memories_path = os.path.join(path, "memories.json")
        if os.path.exists(memories_path):
            with open(memories_path, "r") as f:
                memory_data = json.load(f)
                self.memories = memory_data["texts"]
                self.metadata = memory_data["metadata"]
        
        # Load FAISS index
        index_path = os.path.join(path, "memory_index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        print(f"Memory loaded from {path}")