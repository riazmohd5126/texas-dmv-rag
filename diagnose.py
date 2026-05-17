#!/usr/bin/env python3
"""
Diagnostic Script for RAG Web App
==================================
Checks all requirements and configurations
"""

import os
import sys
from pathlib import Path

print("="*80)
print("RAG WEB APP - DIAGNOSTIC CHECK")
print("="*80)
print()

# Get script directory
script_dir = Path(__file__).parent.absolute()
print(f"📁 Script directory: {script_dir}")
print()

# Check 1: Python version
print("1️⃣  Checking Python version...")
print(f"   Python: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required!")
else:
    print("   ✅ Python version OK")
print()

# Check 2: Required files
print("2️⃣  Checking required files...")
required_files = [
    'backend_api.py',
    'fast_rag_universal.py',
    'universal_extractor.py',
    'chunking_structure_aware.py',
    'advanced_retrieval.py',
    'frontend/index.html'
]

all_files_exist = True
for file in required_files:
    file_path = script_dir / file
    if file_path.exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} NOT FOUND")
        all_files_exist = False

if not all_files_exist:
    print("\n   ⚠️  Some files are missing!")
    print("   Make sure all files are in the same directory")
print()

# Check 3: Python dependencies
print("3️⃣  Checking Python dependencies...")
dependencies = [
    ('flask', 'Flask'),
    ('flask_cors', 'Flask-CORS'),
    ('groq', 'Groq'),
    ('sentence_transformers', 'Sentence Transformers'),
    ('chromadb', 'ChromaDB')
]

missing_deps = []
for module, name in dependencies:
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} NOT INSTALLED")
        missing_deps.append(name)

if missing_deps:
    print(f"\n   ⚠️  Install missing dependencies:")
    print(f"   pip install flask flask-cors groq sentence-transformers chromadb")
print()

# Check 4: Environment variables
print("4️⃣  Checking environment variables...")
groq_key = os.getenv('GROQ_API_KEY')
if groq_key:
    print(f"   ✅ GROQ_API_KEY is set ({groq_key[:10]}...)")
else:
    print(f"   ❌ GROQ_API_KEY not set")
    print(f"   Run: export GROQ_API_KEY='your-key'")
print()

# Check 5: Port availability
print("5️⃣  Checking port 5000...")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 5000))
sock.close()

if result == 0:
    print("   ⚠️  Port 5000 is already in use!")
    print("   Stop existing server or use different port")
else:
    print("   ✅ Port 5000 is available")
print()

# Check 6: Frontend directory
print("6️⃣  Checking frontend directory...")
frontend_dir = script_dir / 'frontend'
index_file = frontend_dir / 'index.html'

if frontend_dir.exists():
    print(f"   ✅ Frontend directory exists: {frontend_dir}")
    if index_file.exists():
        size = index_file.stat().st_size
        print(f"   ✅ index.html exists ({size:,} bytes)")
    else:
        print(f"   ❌ index.html NOT FOUND")
else:
    print(f"   ❌ Frontend directory NOT FOUND")
    print(f"   Expected: {frontend_dir}")
print()

# Check 7: Database
print("7️⃣  Checking database...")
db_path = script_dir / 'chroma_universal'
if db_path.exists():
    print(f"   ✅ Database exists: {db_path}")
else:
    print(f"   ℹ️  Database not found (will be created on first run)")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)

issues = []
if sys.version_info < (3, 8):
    issues.append("Python version too old")
if not all_files_exist:
    issues.append("Missing required files")
if missing_deps:
    issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
if not groq_key:
    issues.append("GROQ_API_KEY not set")
if not frontend_dir.exists() or not index_file.exists():
    issues.append("Frontend files not found")

if issues:
    print("\n❌ Issues found:")
    for issue in issues:
        print(f"   • {issue}")
    print("\n🔧 Fix these issues before running the server")
else:
    print("\n✅ All checks passed!")
    print("\n🚀 Ready to start the server:")
    print(f"   cd {script_dir}")
    print(f"   python3 backend_api.py")
    print(f"\n   Then open: http://localhost:5000")

print()
print("="*80)
