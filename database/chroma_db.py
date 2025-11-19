"""
ChromaDB Configuration and Collection Management
This handles the "AI Memory" database for storing conversation logs
"""

import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB client configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")


def get_chroma_client():
    """
    Initialize and return ChromaDB client
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    return client


def get_or_create_relationship_collection(relationship_id: int, reset: bool = False):
    """
    Get or create a collection for a specific relationship
    Collection name format: relationship_chats_{relationship_id}
    
    Args:
        relationship_id: The unique ID of the relationship
        reset: If True, delete existing collection and create new one
        
    Returns:
        ChromaDB collection object
    """
    client = get_chroma_client()
    collection_name = f"relationship_chats_{relationship_id}"
    
    # If reset requested, delete existing collection first
    if reset:
        try:
            client.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist yet
    
    # Create or get collection with metadata schema
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",  # Use cosine similarity for semantic search
            "description": f"Conversation logs for relationship ID {relationship_id}"
        }
    )
    
    return collection


def add_conversation_to_memory(
    relationship_id: int,
    chat_id: str,
    text: str,
    speaker: str,
    timestamp: str,
    topic: str = None,
    **additional_metadata
):
    """
    Add a conversation chunk to ChromaDB memory
    
    Args:
        relationship_id: The relationship this chat belongs to
        chat_id: Unique ID for this chat chunk (e.g., "chat_chunk_105")
        text: The actual conversation text
        speaker: "self" or "partner"
        timestamp: ISO format timestamp (e.g., "2025-09-28T17:00:00")
        topic: Optional topic tag (e.g., "meeting_avoidance", "trust_issue")
        **additional_metadata: Any additional metadata (e.g., lie_detected=True)
    
    Returns:
        Success status
    """
    collection = get_or_create_relationship_collection(relationship_id)
    
    # Prepare metadata
    metadata = {
        "speaker": speaker,
        "timestamp": timestamp,
        "text": text,  # Store original text in metadata for citation
    }
    
    if topic:
        metadata["topic"] = topic
    
    # Add any additional metadata
    metadata.update(additional_metadata)
    
    # Add to collection
    collection.add(
        ids=[chat_id],
        documents=[text],  # This gets vectorized automatically
        metadatas=[metadata]
    )
    
    return True


def search_conversation_memory(
    relationship_id: int = None,
    collection = None,
    query: str = "",
    n_results: int = 5,
    speaker_filter: str = None
):
    """
    Search conversation memory for relevant context with speaker filtering (v3.1)
    
    Args:
        relationship_id: The relationship to search in (optional if collection provided)
        collection: ChromaDB collection object (optional, will create if not provided)
        query: Search query (e.g., "avoid ignore distance")
        n_results: Number of results to return
        speaker_filter: Filter by speaker - "self", "partner", "other", or None for all
        
    Returns:
        List of dictionaries with metadata (text, speaker, timestamp, topic)
    """
    if collection is None:
        collection = get_or_create_relationship_collection(relationship_id)
    
    # Build where filter for speaker
    where_filter = None
    if speaker_filter:
        where_filter = {"speaker": speaker_filter}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    # Format results for easier use
    formatted_results = []
    if results and results['documents']:
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                'text': doc,
                'metadata': metadata
            })
    
    return formatted_results


def get_context_messages(
    relationship_id: int,
    message_index: int,
    context_window: int = 5
):
    """
    Get surrounding context messages for a specific message (v3.0 Context Retrieval)
    
    Args:
        relationship_id: The relationship to search in
        message_index: The message_index to get context for
        context_window: Number of messages before and after (default 5)
        
    Returns:
        List of messages with metadata in chronological order
    """
    collection = get_or_create_relationship_collection(relationship_id)
    
    # Get messages in the index range
    start_idx = max(0, message_index - context_window)
    end_idx = message_index + context_window
    
    all_messages = collection.get(
        where={
            "$and": [
                {"message_index": {"$gte": start_idx}},
                {"message_index": {"$lte": end_idx}}
            ]
        },
        include=["metadatas", "documents"]
    )
    
    return all_messages


# Example usage structure (commented out):
"""
# Adding a conversation:
add_conversation_to_memory(
    relationship_id=1,
    chat_id="chat_chunk_105",
    text="굳이",
    speaker="partner",
    timestamp="2025-09-28T17:00:00",
    topic="meeting_avoidance"
)

# Searching for evidence:
results = search_conversation_memory(
    relationship_id=1,
    query="Jo, I'm the only one trying, lack of effort",
    n_results=5
)

# Results will contain:
# - ids: ["chat_chunk_105", ...]
# - documents: ["굳이", ...]
# - metadatas: [{"speaker": "partner", "timestamp": "2025-09-28T17:00:00", "text": "굳이", "topic": "meeting_avoidance"}, ...]
"""
