"""
Stage 1 Environment Test Script
Tests all components installed in Stage 1
"""

import sys
print("=" * 50)
print("🧪 STAGE 1 ENVIRONMENT TEST")
print("=" * 50)

# Test 1: Python Version
print("\n1️⃣ Testing Python Version...")
print(f"   ✅ Python {sys.version}")
if sys.version_info >= (3, 9):
    print("   ✅ Python version is compatible (3.9+)")
else:
    print("   ❌ Python version too old!")
    sys.exit(1)

# Test 2: Required Packages
print("\n2️⃣ Testing Package Imports...")
required_packages = {
    "streamlit": "Streamlit (UI Framework)",
    "langchain": "LangChain (AI Pipeline)",
    "langchain_google_genai": "LangChain-Google-GenAI",
    "google.generativeai": "Google Gemini API",
    "chromadb": "ChromaDB (Vector Database)",
    "sqlalchemy": "SQLAlchemy (SQL Database)",
    "dotenv": "Python-dotenv (Environment Variables)"
}

failed_imports = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {description}")
    except ImportError as e:
        print(f"   ❌ {description} - FAILED")
        failed_imports.append(package)

if failed_imports:
    print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
    sys.exit(1)

# Test 3: Environment Variables
print("\n3️⃣ Testing Environment Variables...")
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"   ✅ GOOGLE_API_KEY loaded")
    print(f"   ✅ Key format: {api_key[:8]}...{api_key[-4:]}")
else:
    print("   ❌ GOOGLE_API_KEY not found in .env file")
    sys.exit(1)

# Test 4: Gemini API Connection
print("\n4️⃣ Testing Gemini API Connection...")
try:
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Try to list models (lightweight test)
    models = genai.list_models()
    model_count = sum(1 for _ in models)
    
    print(f"   ✅ Successfully connected to Gemini API")
    print(f"   ✅ Available models: {model_count}")
    
    # Test a simple generation
    print("\n5️⃣ Testing Gemini Text Generation...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say 'Hello' in one word.")
    
    print(f"   ✅ Gemini response: {response.text.strip()}")
    
except Exception as e:
    print(f"   ❌ Gemini API test failed: {str(e)}")
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 50)
print("🎉 ALL STAGE 1 TESTS PASSED!")
print("=" * 50)
print("\n✅ Environment is ready for Stage 2!")
