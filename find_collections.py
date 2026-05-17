#!/usr/bin/env python3
"""
Find Existing RAG Collections
==============================
Shows all ChromaDB collections and their document counts
"""

import os
import chromadb
from pathlib import Path

print("="*70)
print("FINDING EXISTING RAG COLLECTIONS")
print("="*70)

# Check for chroma databases
chroma_dirs = []
for item in os.listdir('.'):
    if item.startswith('chroma') and os.path.isdir(item):
        chroma_dirs.append(item)

if not chroma_dirs:
    print("\n❌ No ChromaDB databases found in current directory!")
    print(f"\nCurrent directory: {os.getcwd()}")
    print("\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    print("\n💡 You need to create a new database or copy existing one here.")
    exit(1)

print(f"\n✅ Found {len(chroma_dirs)} ChromaDB database(s):")
for db_dir in chroma_dirs:
    print(f"  - {db_dir}")

print("\n" + "="*70)
print("COLLECTIONS IN EACH DATABASE")
print("="*70)

for db_dir in chroma_dirs:
    print(f"\n📁 Database: {db_dir}")
    print("-" * 70)
    
    try:
        client = chromadb.PersistentClient(path=f'./{db_dir}')
        collections = client.list_collections()
        
        if not collections:
            print("  ⚠️  No collections found (empty database)")
        else:
            print(f"  Found {len(collections)} collection(s):")
            for col in collections:
                count = col.count()
                print(f"\n  Collection: '{col.name}'")
                print(f"    Documents: {count:,}")
                print(f"    Metadata: {col.metadata}")
                
                if count > 0:
                    print(f"\n    💡 To use this collection, update simple_server.py:")
                    print(f"       collection_name=\"{col.name}\"")
                    print(f"       persist_directory=\"./{db_dir}\"")
    
    except Exception as e:
        print(f"  ❌ Error reading database: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Find collections with documents
has_data = []
for db_dir in chroma_dirs:
    try:
        client = chromadb.PersistentClient(path=f'./{db_dir}')
        collections = client.list_collections()
        for col in collections:
            if col.count() > 0:
                has_data.append({
                    'db': db_dir,
                    'collection': col.name,
                    'count': col.count()
                })
    except:
        pass

if has_data:
    print("\n✅ Collections with data:")
    for item in has_data:
        print(f"\n  Database: {item['db']}")
        print(f"  Collection: {item['collection']}")
        print(f"  Documents: {item['count']:,}")
        print(f"\n  📝 Update simple_server.py with:")
        print(f"     collection_name=\"{item['collection']}\"")
        print(f"     persist_directory=\"./{item['db']}\"")
else:
    print("\n⚠️  No collections with data found!")
    print("\n💡 You need to either:")
    print("   1. Load documents into a collection")
    print("   2. Copy your existing database here")
    print("   3. Update DOCUMENTS_FOLDER path and restart server")

print("\n" + "="*70)
