"""
Database Models for (R)Evolution App
Uses SQLAlchemy for relational database (SQLite)
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """
    Users table - stores user account information
    """
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)  # Will store hashed password
    is_admin = Column(Boolean, default=False)  # Admin flag
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    relationships = relationship('Relationship', back_populates='user')
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, email='{self.email}', is_admin={self.is_admin})>"


class Relationship(Base):
    """
    Relationships table - stores relationship records
    Each user can have multiple relationships (dating or breakup)
    """
    __tablename__ = 'relationships'
    
    relationship_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    status = Column(String(20), nullable=False)  # "Dating" or "Breakup"
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)  # Nullable (only for breakups)
    total_duration_days = Column(Integer, nullable=True)  # Calculated field
    
    # Relationships
    user = relationship('User', back_populates='relationships')
    participants = relationship('Participant', back_populates='relationship')
    
    def __repr__(self):
        return f"<Relationship(relationship_id={self.relationship_id}, status='{self.status}', user_id={self.user_id})>"


class Participant(Base):
    """
    Participants table - stores detailed info about "self" and "partner"
    """
    __tablename__ = 'participants'
    
    participant_id = Column(Integer, primary_key=True, autoincrement=True)
    relationship_id = Column(Integer, ForeignKey('relationships.relationship_id'), nullable=False)
    role = Column(String(10), nullable=False)  # "self" or "partner"
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    mbti = Column(String(4), nullable=True)  # e.g., "INFP"
    occupation = Column(String(100), nullable=True)  # e.g., "Software Engineer", "Student"
    notes = Column(String(500), nullable=True)  # Special notes for AI reference
    
    # Relationships
    relationship = relationship('Relationship', back_populates='participants')
    
    def __repr__(self):
        return f"<Participant(participant_id={self.participant_id}, role='{self.role}', age={self.age})>"


class Hobby(Base):
    """
    Hobbies table - stores user hobbies for AI coach recommendations
    Used when suggesting personalized actions (e.g., "suggest screen baseball")
    """
    __tablename__ = 'hobbies'
    
    hobby_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    hobby_name = Column(String(100), nullable=False)  # e.g., "Screen Baseball", "Gaming", "Karaoke"
    category = Column(String(50), nullable=True)  # e.g., "Sports", "Indoor", "Instant Relief"
    
    # Relationships
    usr = relationship('User')
    
    def __repr__(self):
        return f"<Hobby(hobby_id={self.hobby_id}, hobby_name='{self.hobby_name}', category='{self.category}')>"


class AnalysisHistory(Base):
    """
    Analysis History table - stores all AI analysis results
    Allows users and admins to review past analyses
    """
    __tablename__ = 'analysis_history'
    
    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    relationship_id = Column(Integer, ForeignKey('relationships.relationship_id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # e.g., "emotion_cause", "partner_behavior", "self_behavior", "effort_ratio", "relationship_pattern"
    query_input = Column(Text, nullable=True)  # User's input query (for emotion cause analysis)
    ai_response = Column(Text, nullable=False)  # AI's complete response
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships (renamed to avoid conflict with relationship() function)
    rel = relationship('Relationship')
    usr = relationship('User')
    
    def __repr__(self):
        return f"<AnalysisHistory(analysis_id={self.analysis_id}, type='{self.analysis_type}', created_at='{self.created_at}')>"
