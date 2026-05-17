#!/usr/bin/env python3
"""
Patch 2 for fast_rag_universal.py
Adds smart top_k — automatically increases chunk count for
aggregation and comparison queries.
Only touches 2 lines. Everything else stays the same.
"""
import shutil
from datetime import datetime

FILE = '/Users/riazmohd/Downloads/TexasRAGF 2/fast_rag_universal.py'

# Backup
backup = FILE.replace('.py', f'_backup_patch2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py')
shutil.copy2(FILE, backup)
print(f"✅ Backup: {backup}")

with open(FILE, 'r') as f:
    content = f.read()

# ── Change 1: Add get_smart_top_k to existing import ────────────────────────
OLD_IMPORT = "from document_categories import get_category, get_filter_categories"
NEW_IMPORT = "from document_categories import get_category, get_filter_categories, get_smart_top_k"

if "get_smart_top_k" in content:
    print("⏭️  Change 1: get_smart_top_k already imported, skipping")
elif OLD_IMPORT in content:
    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("✅ Change 1: Added get_smart_top_k to import")
else:
    print("⚠️  Change 1: Could not find import line — add manually:")
    print("   from document_categories import get_category, get_filter_categories, get_smart_top_k")

# ── Change 2: Use smart top_k inside query() ─────────────────────────────────
# Find the auto-classify block we added in patch 1 and insert top_k logic after it

OLD_CLASSIFY = """            # ── Auto category filter (added by patch) ──────────────────────
            if metadata_filter is None:
                _auto_cats = get_filter_categories(question)
                if _auto_cats:
                    metadata_filter = {"category": {"$in": _auto_cats}}
                    print(f"   🏷️  Auto filter: {_auto_cats}")
                else:
                    print(f"   🏷️  No filter (broad query)")
            # ─────────────────────────────────────────────────────────────"""

NEW_CLASSIFY = """            # ── Auto category filter (added by patch) ──────────────────────
            if metadata_filter is None:
                _auto_cats = get_filter_categories(question)
                if _auto_cats:
                    metadata_filter = {"category": {"$in": _auto_cats}}
                    print(f"   🏷️  Auto filter: {_auto_cats}")
                else:
                    print(f"   🏷️  No filter (broad query)")
            # ── Smart top_k (added by patch 2) ───────────────────────────────
            top_k = get_smart_top_k(question, default=top_k)
            # ─────────────────────────────────────────────────────────────"""

if "Smart top_k" in content:
    print("⏭️  Change 2: smart top_k already present, skipping")
elif OLD_CLASSIFY in content:
    content = content.replace(OLD_CLASSIFY, NEW_CLASSIFY, 1)
    print("✅ Change 2: Added smart top_k inside query()")
else:
    print("⚠️  Change 2: Could not find auto-classify block — add manually:")
    print("   Add this line right before your collection.query() call:")
    print("   top_k = get_smart_top_k(question, default=top_k)")

# Write
with open(FILE, 'w') as f:
    f.write(content)

# Verify
print("\n── VERIFICATION ───────────────────────────────────────────────────────")
with open(FILE, 'r') as f:
    final = f.read()

checks = [
    ("get_smart_top_k imported",  "get_smart_top_k" in final),
    ("Smart top_k in query()",    "Smart top_k" in final),
]

all_ok = True
for label, ok in checks:
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok:
        all_ok = False

if all_ok:
    print("""
🎉 Done! No re-index needed — this only changes query behaviour.

Restart the server:
  lsof -ti:8080 | xargs kill -9 2>/dev/null
  export GROQ_API_KEY='your-key'
  python3 backend_api.py

Then test:
  "What are all the costs involved in opening a new independent dealership?"
  → should now return top_k=10 and find the right fee chunks
""")
else:
    print(f"\n⚠️  Apply missing changes manually. Backup at: {backup}")
