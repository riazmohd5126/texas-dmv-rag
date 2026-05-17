"""
Texas DMV RAG — Retrieval Accuracy Test
========================================
Tests chunking + BM25 retrieval against real documents from Google Drive.
No GROQ_API_KEY needed — this validates the retrieval layer only.

For each test question, we check:
  1. Is the correct chunk retrieved in top-3?
  2. Is it ranked #1?
  3. Does the retrieved chunk contain the expected answer keywords?
"""

import sys, types, os, json

# Mock sentence_transformers (not needed for BM25 tests)
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
_st.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", _st)

# Suppress noisy prints during import
import io
_null = io.StringIO()
sys.stdout = _null
from chunking_structure_aware import StructureAwareSplitter
from advanced_retrieval import BM25Retriever
from document_categories import classify_query, get_smart_top_k
sys.stdout = sys.__stdout__

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

results = {"pass": 0, "fail": 0, "warn": 0, "details": []}

# ── Load and chunk documents ──────────────────────────────────────────────────
DOCS_DIR = "./docs"
splitter  = StructureAwareSplitter(chunk_size=700, chunk_overlap=120)

all_chunks  = []   # list of TextChunk
all_texts   = []   # parallel list of plain strings
all_sources = []   # source filenames

print(f"\n{BOLD}Loading documents from {DOCS_DIR}...{RESET}")
for fname in sorted(os.listdir(DOCS_DIR)):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(DOCS_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    chunks = splitter.split(content, metadata={"source_file": fname})
    for c in chunks:
        all_chunks.append(c)
        all_texts.append(c.text)
        all_sources.append(fname)

print(f"  {len(all_chunks)} chunks from {len(os.listdir(DOCS_DIR))} documents\n")

# ── Build BM25 index ──────────────────────────────────────────────────────────
sys.stdout = _null
bm25 = BM25Retriever()
bm25.index(all_texts)
sys.stdout = sys.__stdout__
print(f"  BM25 index built over {len(all_texts)} chunks\n")

# ── Test harness ──────────────────────────────────────────────────────────────
def search(query, top_k=5):
    """Return list of (chunk_text, source_file, score) for top_k results."""
    hits = bm25.search(query, top_k=top_k)
    return [(all_texts[i], all_sources[i], score) for i, score in hits]

def check_retrieval(query, expected_keywords, label, top_k=5):
    """
    Pass if ANY of the top_k results contains ALL expected_keywords (case-insensitive).
    Also report if the match is rank #1 vs rank 2-N.
    """
    hits = search(query, top_k=top_k)
    for rank, (text, src, score) in enumerate(hits, start=1):
        text_lower = text.lower()
        if all(kw.lower() in text_lower for kw in expected_keywords):
            if rank == 1:
                print(f"  {PASS}  [{label}] — found at rank #{rank} (score={score:.3f})")
                print(f"         Source: {src}")
                results["pass"] += 1
                results["details"].append({"q": query, "status": "pass", "rank": rank})
            else:
                print(f"  {WARN}  [{label}] — found at rank #{rank} (not #1, score={score:.3f})")
                print(f"         Source: {src}")
                results["warn"] += 1
                results["details"].append({"q": query, "status": "warn", "rank": rank})
            return
    print(f"  {FAIL}  [{label}]")
    print(f"         Query: {query!r}")
    print(f"         Expected keywords: {expected_keywords}")
    print(f"         Top result: {hits[0][0][:120]!r}..." if hits else "         No results")
    results["fail"] += 1
    results["details"].append({"q": query, "status": "fail", "rank": -1})

def section(title):
    print(f"\n{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{CYAN}{'─'*70}{RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST QUESTIONS — grouped by topic
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. GDN Definition & Basics ────────────────────────────────────────────────
section("1. GDN Definition & Basics")

check_retrieval(
    "What is a GDN license?",
    ["gdn", "basic dealer license", "buy, sell"],
    "GDN definition"
)
check_retrieval(
    "What does GDN stand for?",
    ["general distinguishing number"],
    "GDN full name"
)
check_retrieval(
    "Who needs a GDN in Texas?",
    ["gdn", "buy, sell"],
    "Who needs GDN"
)
check_retrieval(
    "How long is a GDN license valid?",
    ["2 years", "two-year"],
    "GDN license term"
)
check_retrieval(
    "What is the GDN license fee?",
    ["700", "$700"],
    "GDN license fee"
)
check_retrieval(
    "What are the different types of GDN licenses?",
    ["motor vehicle", "motorcycle", "travel trailer"],
    "GDN license types"
)

# ── 2. Surety Bond ────────────────────────────────────────────────────────────
section("2. Surety Bond Requirements")

check_retrieval(
    "What is the surety bond requirement for a dealer?",
    ["50,000", "surety bond"],
    "Surety bond amount"
)
check_retrieval(
    "Do travel trailer dealers need a surety bond?",
    ["travel trailer", "bond"],
    "Travel trailer bond exemption"
)
check_retrieval(
    "How long must the surety bond be valid?",
    ["2 years", "bond"],
    "Surety bond term"
)
check_retrieval(
    "Who requires a surety bond for GDN?",
    ["motor vehicle dealer", "motorcycle", "surety bond"],
    "Who needs bond"
)

# ── 3. Premises & Signage ─────────────────────────────────────────────────────
section("3. Premises, Display Area & Signage")

check_retrieval(
    "What are the dealership office requirements?",
    ["office", "desk", "telephone"],
    "Office requirements"
)
check_retrieval(
    "What is the display area requirement for a dealer?",
    ["display area", "five", "vehicles"],
    "Display area requirement"
)
check_retrieval(
    "What are the sign requirements for a dealership?",
    ["sign", "6 inches", "permanent"],
    "Signage requirement"
)
check_retrieval(
    "How many dealers can share the same business location?",
    ["four retail", "eight wholesale"],
    "Sharing location limit"
)

# ── 4. Hours of Operation ─────────────────────────────────────────────────────
section("4. Hours of Operation")

check_retrieval(
    "How many hours must a dealer be open per week?",
    ["4 consecutive hours", "4 days"],
    "Retail dealer hours"
)
check_retrieval(
    "Can a dealer be open on both Saturday and Sunday?",
    ["saturday", "sunday", "not both"],
    "Blue law Saturday Sunday"
)
check_retrieval(
    "What is the Blue Law for dealers?",
    ["saturday", "sunday"],
    "Blue law reference"
)

# ── 5. Franchise License ──────────────────────────────────────────────────────
section("5. Franchise License")

check_retrieval(
    "What is the difference between a franchise license and a GDN?",
    ["franchise", "gdn"],
    "Franchise vs GDN"
)
check_retrieval(
    "Can a franchise dealer sell on both Saturday and Sunday?",
    ["saturday", "sunday"],
    "Franchise Saturday Sunday"
)
check_retrieval(
    "What is the fee for a service-only franchise facility?",
    ["200", "service-only"],
    "Service-only fee"
)
check_retrieval(
    "Can franchised dealers lease vehicles?",
    ["franchised dealer", "lease"],
    "Franchise lease"
)

# ── 6. HB 718 Metal Plates ────────────────────────────────────────────────────
section("6. House Bill 718 / Metal Plates")

check_retrieval(
    "When did HB 718 take effect?",
    ["july 1, 2025", "july 1"],
    "HB 718 effective date"
)
check_retrieval(
    "What replaced paper buyer temporary tags?",
    ["general issue", "metal"],
    "Metal plate replacement"
)
check_retrieval(
    "How long is an out-of-state buyer plate valid?",
    ["60 days", "out-of-state"],
    "Out-of-state plate validity"
)
check_retrieval(
    "What should dealers do with incoming trade-in plates?",
    ["10 days", "reassign", "destroy"],
    "Trade-in plate handling"
)
check_retrieval(
    "Where must dealers store metal plates?",
    ["locked", "secure"],
    "Plate storage"
)

# ── 7. Salvage Dealer ─────────────────────────────────────────────────────────
section("7. Salvage Dealer License")

check_retrieval(
    "When do I need a salvage dealer license?",
    ["five", "salvage", "calendar year"],
    "Salvage license threshold"
)
check_retrieval(
    "What is the salvage dealer license fee?",
    ["190", "$190"],
    "Salvage dealer fee"
)
check_retrieval(
    "Can a salvage dealer operate from their home?",
    ["residence", "salvage"],
    "Salvage home-based restriction"
)

# ── 8. Enforcement / Penalties ────────────────────────────────────────────────
section("8. Enforcement & Penalties")

check_retrieval(
    "What is the penalty for fraudulently issuing temporary tags?",
    ["class a misdemeanor", "4,000"],
    "Tag fraud first offense"
)
check_retrieval(
    "What happens if a dealer issues 25 or more fraudulent tags?",
    ["felony", "25"],
    "Tag fraud felony threshold"
)
check_retrieval(
    "Can TxDMV revoke a dealer license?",
    ["revocation", "revoke"],
    "License revocation"
)
check_retrieval(
    "What is the administrative penalty per violation per day?",
    ["10,000", "per violation"],
    "Administrative penalty"
)

# ── 9. Leasing ────────────────────────────────────────────────────────────────
section("9. Leasing Licenses")

check_retrieval(
    "What is the difference between a lessor and a lease facilitator?",
    ["lessor", "facilitator"],
    "Lessor vs facilitator"
)
check_retrieval(
    "Do franchised dealers need a lessor license?",
    ["franchised dealer", "lessor"],
    "Franchise lessor exemption"
)

# ── 10. Converter ─────────────────────────────────────────────────────────────
section("10. Converter License")

check_retrieval(
    "What is a converter license used for?",
    ["convert", "assembles", "chassis"],
    "Converter definition"
)
check_retrieval(
    "Can a converter sell vehicles directly to consumers?",
    ["converter", "consumer", "franchised"],
    "Converter sales restriction"
)
check_retrieval(
    "Does a converter need to be located in Texas?",
    ["located in texas", "converter"],
    "Converter location requirement"
)

# ── 11. Query Classifier cross-check ─────────────────────────────────────────
section("11. Query Classifier + Smart top_k alignment")

classifier_cases = [
    ("What is the surety bond requirement?",       "licensing_elicensing_guides", 5),
    ("What is a GDN?",                             "licensing_elicensing_guides", 5),
    ("When did HB 718 take effect?",               "hb718_metal_plates",          5),
    ("What is the penalty for tag fraud?",         "enforcement_compliance_penalties", 5),
    ("What are all the costs to open a dealer?",   "licensing_elicensing_guides", 10),
    ("Difference between lessor and facilitator",  None,                          12),
]

for query, exp_cat, exp_k in classifier_cases:
    cat = classify_query(query)
    k   = get_smart_top_k(query)
    cat_ok = (cat == exp_cat) or (exp_cat is None and cat is None)
    k_ok   = (k == exp_k)
    if cat_ok and k_ok:
        print(f"  {PASS}  classify+topk: {query!r}")
        print(f"         category={cat!r}, top_k={k}")
        results["pass"] += 1
    elif not cat_ok:
        print(f"  {WARN}  classify: {query!r}")
        print(f"         expected category={exp_cat!r}, got={cat!r}")
        results["warn"] += 1
    elif not k_ok:
        print(f"  {FAIL}  top_k: {query!r}")
        print(f"         expected={exp_k}, got={k}")
        results["fail"] += 1

# ── Summary ───────────────────────────────────────────────────────────────────
total = results["pass"] + results["fail"] + results["warn"]
print(f"\n{'='*70}")
print(f"{BOLD}  RETRIEVAL ACCURACY RESULTS{RESET}")
print(f"{'='*70}")
print(f"  Total tests : {total}")
print(f"  {GREEN}Passed{RESET}      : {results['pass']}  ({100*results['pass']//total if total else 0}%)")
print(f"  {YELLOW}Warnings{RESET}    : {results['warn']}  (retrieved but not rank #1)")
print(f"  {RED}Failed{RESET}      : {results['fail']}")
print(f"{'='*70}")

# Accuracy = pass + warn (answer was found, just not rank 1)
found = results["pass"] + results["warn"]
print(f"\n  Top-5 recall (answer found in top-5): {found}/{total} = {100*found//total if total else 0}%")
print(f"  Precision@1  (answer ranked #1):      {results['pass']}/{total} = {100*results['pass']//total if total else 0}%")

# Save JSON report
with open("retrieval_report.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Full report saved to retrieval_report.json")

if results["fail"] > 0:
    sys.exit(1)
