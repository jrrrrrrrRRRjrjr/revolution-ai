"""
Database Migration Script - Add occupation column to participants table
Run this once to update existing database schema
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Get database URL
database_url = os.getenv('DATABASE_URL')

if not database_url:
    print("ERROR: DATABASE_URL not found in .env file")
    exit(1)

print(f"Connecting to database...")

try:
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT COUNT(*) as count 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = 'revolution_ai' 
            AND TABLE_NAME = 'participants' 
            AND COLUMN_NAME = 'occupation'
        """))
        
        exists = result.fetchone()[0] > 0
        
        if exists:
            print("✅ Column 'occupation' already exists in participants table")
        else:
            # Add occupation column
            conn.execute(text("""
                ALTER TABLE participants 
                ADD COLUMN occupation VARCHAR(100) NULL 
                AFTER mbti
            """))
            conn.commit()
            print("✅ Successfully added 'occupation' column to participants table")
        
        # Verify the column
        result = conn.execute(text("DESCRIBE participants"))
        print("\n📋 Current participants table schema:")
        for row in result:
            print(f"  - {row[0]}: {row[1]}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("\n✅ Migration completed successfully!")
