"""
Stage 2 Database Test Script
Tests SQLite and ChromaDB functionality
"""

import sys
import os
from datetime import datetime

print("=" * 50)
print("🧪 STAGE 2 DATABASE TEST")
print("=" * 50)

# Test 1: SQLite Models
print("\n1️⃣ Testing SQLite Models...")
try:
    from database.models import Base, User, Relationship, Participant, Hobby
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory test database
    engine = create_engine('sqlite:///:memory:', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("   ✅ All 4 models imported successfully")
    print("   ✅ Database tables created")
    
    # Test 2: Insert Test Data
    print("\n2️⃣ Testing Data Insertion...")
    
    # Create test user
    test_user = User(
        email="jo@test.com",
        password_hash="hashed_password_123",
        created_at=datetime.utcnow()
    )
    session.add(test_user)
    session.commit()
    print(f"   ✅ User created: {test_user}")
    
    # Create test relationship
    test_relationship = Relationship(
        user_id=test_user.user_id,
        status="Dating",
        start_date=datetime.utcnow(),
        total_duration_days=130
    )
    session.add(test_relationship)
    session.commit()
    print(f"   ✅ Relationship created: {test_relationship}")
    
    # Create test participants
    test_self = Participant(
        relationship_id=test_relationship.relationship_id,
        role="self",
        age=26,
        gender="male",
        mbti="INFP",
        notes="First relationship"
    )
    test_partner = Participant(
        relationship_id=test_relationship.relationship_id,
        role="partner",
        age=21,
        gender="female",
        mbti="ENFJ"
    )
    session.add_all([test_self, test_partner])
    session.commit()
    print(f"   ✅ Participants created: self={test_self}, partner={test_partner}")
    
    # Create test hobby
    test_hobby = Hobby(
        user_id=test_user.user_id,
        hobby_name="Screen Baseball",
        category="Sports"
    )
    session.add(test_hobby)
    session.commit()
    print(f"   ✅ Hobby created: {test_hobby}")
    
    # Test 3: Query Data
    print("\n3️⃣ Testing Data Queries...")
    
    # Query user
    queried_user = session.query(User).filter_by(email="jo@test.com").first()
    assert queried_user.email == "jo@test.com"
    print(f"   ✅ User query successful: {queried_user.email}")
    
    # Query relationship with participants
    queried_rel = session.query(Relationship).filter_by(user_id=test_user.user_id).first()
    assert queried_rel.status == "Dating"
    assert len(queried_rel.participants) == 2
    print(f"   ✅ Relationship query successful: {queried_rel.status}, {len(queried_rel.participants)} participants")
    
    # Query hobbies
    hobbies = session.query(Hobby).filter_by(user_id=test_user.user_id).all()
    assert len(hobbies) == 1
    print(f"   ✅ Hobby query successful: {hobbies[0].hobby_name}")
    
    session.close()
    
except Exception as e:
    print(f"   ❌ SQLite test failed: {str(e)}")
    sys.exit(1)

# Test 4: ChromaDB
print("\n4️⃣ Testing ChromaDB Setup...")
try:
    from database.chroma_db import (
        get_chroma_client,
        get_or_create_relationship_collection,
        add_conversation_to_memory,
        search_conversation_memory
    )
    
    # Test client connection
    client = get_chroma_client()
    print(f"   ✅ ChromaDB client created")
    
    # Test collection creation
    collection = get_or_create_relationship_collection(relationship_id=999)
    print(f"   ✅ Collection created: {collection.name}")
    
    # Test 5: Add Conversation
    print("\n5️⃣ Testing Conversation Storage...")
    
    add_conversation_to_memory(
        relationship_id=999,
        chat_id="test_chat_1",
        text="굳이",
        speaker="partner",
        timestamp="2025-09-28T17:00:00",
        topic="meeting_avoidance"
    )
    print("   ✅ Conversation added to memory")
    
    add_conversation_to_memory(
        relationship_id=999,
        chat_id="test_chat_2",
        text="I love you",
        speaker="partner",
        timestamp="2025-09-29T10:00:00",
        topic="affection"
    )
    print("   ✅ Second conversation added")
    
    # Test 6: Search Conversation
    print("\n6️⃣ Testing Semantic Search...")
    
    results = search_conversation_memory(
        relationship_id=999,
        query="meeting avoidance lack of effort",
        n_results=2
    )
    
    assert len(results['ids'][0]) > 0
    print(f"   ✅ Search returned {len(results['ids'][0])} results")
    print(f"   ✅ Top result: {results['metadatas'][0][0]['text']} (topic: {results['metadatas'][0][0].get('topic', 'N/A')})")
    
    # Cleanup test collection
    client.delete_collection("relationship_chats_999")
    print("   ✅ Test collection cleaned up")
    
except Exception as e:
    print(f"   ❌ ChromaDB test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 50)
print("🎉 ALL STAGE 2 TESTS PASSED!")
print("=" * 50)
print("\n✅ Database design is working correctly!")
print("✅ SQLite: 4 tables functional")
print("✅ ChromaDB: Vector storage functional")
print("✅ Ready for Stage 3!")
