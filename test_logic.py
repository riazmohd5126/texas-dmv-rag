"""
Texas DMV RAG — Logic Accuracy Test Suite
==========================================
Tests query classification, category routing, smart top_k, get_category,
BM25 scoring, heading detection, and chunk junk filtering.
No external API keys or ChromaDB needed.
"""

import sys
import re

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

results = {"pass": 0, "fail": 0, "warn": 0}


def check(label, actual, expected, warn_only=False):
    if actual == expected:
        print(f"  {PASS}  {label}")
        results["pass"] += 1
    else:
        tag = WARN if warn_only else FAIL
        key = "warn" if warn_only else "fail"
        print(f"  {tag}  {label}")
        print(f"         expected: {expected!r}")
        print(f"         got:      {actual!r}")
        results[key] += 1


def check_in(label, actual, valid_set, warn_only=False):
    if actual in valid_set:
        print(f"  {PASS}  {label}  → {actual!r}")
        results["pass"] += 1
    else:
        tag = WARN if warn_only else FAIL
        key = "warn" if warn_only else "fail"
        print(f"  {tag}  {label}")
        print(f"         expected one of: {valid_set}")
        print(f"         got:             {actual!r}")
        results[key] += 1


def section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. document_categories — classify_query (Licensing & GDN focus)
# ──────────────────────────────────────────────────────────────────────────────

section("1. classify_query — Licensing & GDN questions")

from document_categories import classify_query, get_filter_categories, get_smart_top_k, get_category

LICENSING = "licensing_elicensing_guides"
HB718     = "hb718_metal_plates"
ENF       = "enforcement_compliance_penalties"
FORMS     = "forms_applications"
QS        = "elicensing_quickstart"
GEN       = "general_reference"

licensing_queries = [
    ("What is a GDN?",                                              LICENSING),
    ("What does GDN stand for?",                                    LICENSING),
    ("How do I apply for an independent dealer license?",           LICENSING),
    ("What is the surety bond requirement?",                        LICENSING),
    ("How much does a surety bond cost?",                           LICENSING),
    ("I want to open a dealership — what are the requirements?",    LICENSING),
    ("What is the cost to open an independent dealership?",         LICENSING),
    ("What premises requirements do I need for a dealer license?",  LICENSING),
    ("Do I need a criminal background check for a dealer license?", LICENSING),
    ("How do I renew my dealer license?",                           LICENSING),
    ("What are the signage requirements for a dealership?",         LICENSING),
    ("How do I get a franchise dealer license?",                    LICENSING),
    ("What is a converter license?",                                LICENSING),
    ("How do I apply for a lessor license?",                        LICENSING),
    ("What is required for a salvage dealer license?",              LICENSING),
    ("How do I get a manufacturer license?",                        LICENSING),
    ("What is an in-transit license?",                              LICENSING),
    ("What are the startup costs for opening a dealership?",        LICENSING),
]

for query, expected in licensing_queries:
    result = classify_query(query)
    check(f'classify_query("{query}")', result, expected)


# ──────────────────────────────────────────────────────────────────────────────
# 2. classify_query — should NOT misfire on non-licensing queries
# ──────────────────────────────────────────────────────────────────────────────

section("2. classify_query — non-licensing queries should NOT return licensing")

non_licensing = [
    ("What is a metal plate?",               HB718),
    ("How does webDEALER work?",             HB718),
    ("What is the penalty for tag fraud?",   ENF),
    ("What is form LF610?",                  FORMS),
    ("How do I create an eLICENSING account?", QS),
]

for query, expected in non_licensing:
    result = classify_query(query)
    check(f'classify_query("{query}")', result, expected, warn_only=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. classify_query — None for ambiguous / broad queries
# ──────────────────────────────────────────────────────────────────────────────

section("3. classify_query — ambiguous/broad queries should return None")

broad_queries = [
    "Tell me about Texas motor vehicle dealers",
    "What are the rules?",
    "Help me understand Texas DMV",
]

for query in broad_queries:
    result = classify_query(query)
    check(f'classify_query("{query}")', result, None, warn_only=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. get_filter_categories — always includes general_reference alongside primary
# ──────────────────────────────────────────────────────────────────────────────

section("4. get_filter_categories — includes general_reference fallback")

filter_queries = [
    ("What is the surety bond requirement?",        LICENSING),
    ("What is a GDN?",                              LICENSING),
    ("How do I renew my dealer license?",           LICENSING),
]

for query, primary in filter_queries:
    cats = get_filter_categories(query)
    if cats is None:
        print(f"  {FAIL}  get_filter_categories({query!r}) → None (expected list)")
        results["fail"] += 1
        continue
    has_primary = primary in cats
    has_gen_ref = GEN in cats
    check(f'  has "{primary}"  in filter for: "{query}"', has_primary, True)
    check(f'  has "general_reference" in filter for: "{query}"', has_gen_ref, True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. get_smart_top_k
# ──────────────────────────────────────────────────────────────────────────────

section("5. get_smart_top_k — returns appropriate chunk counts")

top_k_cases = [
    ("What is the surety bond amount?",                  5,  "simple"),
    ("What is a GDN?",                                   5,  "simple"),
    ("What are all the costs to open a dealership?",     10, "aggregation"),
    ("Give me the complete list of license requirements",10, "aggregation"),
    ("What is the difference between a GDN and a franchise license?", 12, "comparison"),
    ("Compare independent vs franchise dealer license",  12, "comparison"),
    ("What changed after July 2025?",                    10, "aggregation"),
    ("Full process to get a dealer license from start to finish", 10, "aggregation"),
]

for query, expected_k, qtype in top_k_cases:
    k = get_smart_top_k(query)
    check(f'top_k={expected_k} ({qtype}) for: "{query}"', k, expected_k)


# ──────────────────────────────────────────────────────────────────────────────
# 6. get_category — filename → category mapping
# ──────────────────────────────────────────────────────────────────────────────

section("6. get_category — filename-to-category mapping")

file_cases = [
    ("elicensing-userguide_independent-gdn-licensees.pdf", LICENSING),
    ("elicensing_userguide_franchise_dealer_licenses.pdf",  LICENSING),
    ("elicensing-user-guide-for-salvage-dealer-licenses",   LICENSING),
    ("elicensing-quickstart_newlicense.pdf",                QS),
    ("elicensing-quick-start-guide-protest-dealer.pdf",     QS),
    ("dealer-plate-faqs_hb718.pdf",                         HB718),
    ("webdealer_4.1.1_dealer_user_guide_0.pdf",             HB718),
    ("texas_tag_penalties.pdf",                             ENF),
    ("_motorists_consumer-protection_lemon-law.pdf",        ENF),
    ("dmv_lf131ab.pdf",                                     FORMS),
    ("lf610.pdf",                                           FORMS),
    ("txdmv_compact_with_texans.pdf",                       GEN),
    ("unknown_random_document.pdf",                         GEN),  # fallback
]

for filename, expected_cat in file_cases:
    cat = get_category(filename)
    check(f'get_category("{filename}")', cat, expected_cat)


# ──────────────────────────────────────────────────────────────────────────────
# 7. BM25 — scoring logic
# ──────────────────────────────────────────────────────────────────────────────

section("7. BM25Retriever — scoring and ranking")

# Mock sentence_transformers so BM25 (pure-Python) can be imported without ML deps
import types, sys as _sys
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
_st.CrossEncoder = object
_sys.modules.setdefault("sentence_transformers", _st)

import sys
sys.stdout = open("/dev/null", "w")  # suppress noisy BM25 init print
from advanced_retrieval import BM25Retriever
sys.stdout = sys.__stdout__

docs = [
    "A General Distinguishing Number (GDN) is required for all independent motor vehicle dealers in Texas.",
    "The surety bond requirement for most independent dealers is $50,000.",
    "To apply for a franchise dealer license, you must submit the eLICENSING application.",
    "House Bill 718 introduced metal license plates to replace temporary paper tags.",
    "Violations of the Texas dealer licensing law may result in fines or license revocation.",
    "The eLICENSING system allows dealers to submit license applications online.",
    "A converter license is required if you substantially alter the body, chassis, or interior of a vehicle.",
]

bm25 = BM25Retriever()
bm25.index(docs)

# Test 1: "GDN" query should rank doc 0 highest
results_gdn = bm25.search("What is a GDN general distinguishing number?", top_k=3)
top_gdn_idx = results_gdn[0][0]
check("BM25: 'GDN' query → doc about GDN ranked #1", top_gdn_idx, 0)

# Test 2: "surety bond" query should rank doc 1 highest
results_bond = bm25.search("surety bond requirement dealer", top_k=3)
top_bond_idx = results_bond[0][0]
check("BM25: 'surety bond' query → bond doc ranked #1", top_bond_idx, 1)

# Test 3: "converter license" query should rank doc 6 highest
results_conv = bm25.search("converter license alter vehicle body", top_k=3)
top_conv_idx = results_conv[0][0]
check("BM25: 'converter license' query → converter doc ranked #1", top_conv_idx, 6)

# Test 4: all scores are non-negative
all_non_neg = all(score >= 0 for _, score in results_gdn + results_bond + results_conv)
check("BM25: all scores are non-negative", all_non_neg, True)

# Test 5: scores are sorted descending
gdn_scores = [s for _, s in results_gdn]
check("BM25: results are sorted descending", gdn_scores, sorted(gdn_scores, reverse=True))

# Test 6: search with unknown term returns zeros or near-zero
results_unk = bm25.search("xyzzy foobar nonexistent term", top_k=3)
all_zero_or_low = all(score < 0.1 for _, score in results_unk)
check("BM25: unknown query returns near-zero scores", all_zero_or_low, True, warn_only=True)


# ──────────────────────────────────────────────────────────────────────────────
# 8. HeadingDetector
# ──────────────────────────────────────────────────────────────────────────────

section("8. HeadingDetector — detects headings correctly")

from chunking_structure_aware import HeadingDetector

hd = HeadingDetector()

text_with_headings = """
1. Introduction

Some introductory text.

1.1 GDN Requirements

A GDN is required for all dealers.

1.1.1 Background Check

You must pass a background check.

## Markdown Heading

Some markdown content.

SECTION ALL CAPS HEADING HERE

Some caps content.
"""

headings = hd.find_headings(text_with_headings)
heading_texts = [h['text'] for h in headings]

check("HeadingDetector: finds at least 4 headings", len(headings) >= 4, True)
has_numbered = any(re.match(r'^\d+\.', h) for h in heading_texts)
check("HeadingDetector: finds numbered headings (1.x)", has_numbered, True)
has_markdown = any(h.startswith('#') for h in heading_texts)
check("HeadingDetector: finds markdown headings", has_markdown, True)

# Level detection
level_cases = [
    ("1. Introduction",    1),
    ("1.1 Sub Section",    2),
    ("1.1.1 Sub Sub",      3),
]
for heading_text, expected_level in level_cases:
    level = hd._detect_level(heading_text)
    check(f"HeadingDetector._detect_level({heading_text!r}) == {expected_level}", level, expected_level)


# ──────────────────────────────────────────────────────────────────────────────
# 9. StructureAwareSplitter — junk detection
# ──────────────────────────────────────────────────────────────────────────────

section("9. StructureAwareSplitter — junk chunk filtering")

from chunking_structure_aware import StructureAwareSplitter

splitter = StructureAwareSplitter()

junk_cases = [
    ("8.3 Closure Application ................................ 95", True,  "TOC with dots"),
    ("Figure 37: Previous Held Licenses .............. 36",          True,  "Figure TOC entry"),
    ("1.2 Requirements",                                              True,  "Heading-only, too short"),
    ("A GDN (General Distinguishing Number) is the license required for all independent motor vehicle dealers operating in Texas. You must obtain a GDN before selling vehicles.", False, "Real content"),
    ("The surety bond requirement for independent dealers is $50,000. This bond protects consumers from dealer fraud and must be maintained throughout the license period.", False, "Real content 2"),
]

for text, expected_junk, label in junk_cases:
    is_junk = splitter._is_junk_chunk(text)
    check(f"_is_junk_chunk — {label}", is_junk, expected_junk)


# ──────────────────────────────────────────────────────────────────────────────
# 10. StructureAwareSplitter — produces valid chunks
# ──────────────────────────────────────────────────────────────────────────────

section("10. StructureAwareSplitter — chunk production")

sample_doc = """
1. Overview

This guide explains the Texas dealer licensing process.

1.1 Who Needs a GDN

Any person or business that buys and sells more than 5 vehicles per year in Texas
must obtain a General Distinguishing Number (GDN). This applies to independent dealers,
franchise dealers, wholesale dealers, and others as defined by the Texas Transportation Code.

1.2 Surety Bond Requirements

Most independent motor vehicle dealers must obtain a $50,000 surety bond before their
license application can be approved. The bond must be issued by a surety company authorized
to do business in Texas. The bond protects consumers from fraudulent dealer practices
and must remain active throughout the license period.

1.3 Premises Requirements

Every licensed dealer must maintain a permanent business location that meets the following
requirements: a permanent building with an office, adequate display area for vehicles,
a permanent sign visible from the road with the dealership name, and the dealer must be
open during regular business hours.
"""

chunks = splitter.split(sample_doc, metadata={"source": "test.pdf"})

check("StructureAwareSplitter: produces at least 1 chunk", len(chunks) >= 1, True)
check("StructureAwareSplitter: all chunks have text", all(c.text.strip() for c in chunks), True)
check("StructureAwareSplitter: all chunks have metadata", all(c.metadata for c in chunks), True)

max_chars = splitter.char_limit
oversized = [c for c in chunks if len(c.text) > max_chars * 1.1]  # 10% tolerance
check("StructureAwareSplitter: no chunk exceeds size limit", len(oversized) == 0, True)

# At least one chunk should contain GDN content
gdn_chunks = [c for c in chunks if "gdn" in c.text.lower() or "general distinguishing" in c.text.lower()]
check("StructureAwareSplitter: GDN content preserved in chunks", len(gdn_chunks) >= 1, True)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  RESULTS: {results['pass']} passed  |  {results['fail']} failed  |  {results['warn']} warnings")
print(f"{'='*70}")

if results["fail"] > 0:
    sys.exit(1)
