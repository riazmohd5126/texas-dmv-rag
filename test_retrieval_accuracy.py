"""
Texas DMV RAG — Expanded Retrieval Accuracy Test (v2)
======================================================
Tests BM25 retrieval against 13 real TxDMV documents using 100 questions
across 16 topic categories. No GROQ_API_KEY needed.

Metrics:
  Precision@1  — correct chunk is rank #1
  Top-5 Recall — correct chunk appears anywhere in top 5
  Top-3 Recall — correct chunk appears anywhere in top 3
"""

import sys, types, os, json, textwrap

# Mock sentence_transformers (not needed for BM25 tests)
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
_st.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", _st)

import io
_null = io.StringIO()
sys.stdout = _null
from chunking_structure_aware import StructureAwareSplitter
from advanced_retrieval import BM25Retriever
from document_categories import classify_query, get_smart_top_k
sys.stdout = sys.__stdout__

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"
PASS   = f"{GREEN}✓ PASS{RESET}"
FAIL   = f"{RED}✗ FAIL{RESET}"
WARN   = f"{YELLOW}⚠ WARN{RESET}"   # found but not rank #1

results = {"pass": 0, "fail": 0, "warn": 0, "details": []}

# ── Load + chunk all docs ─────────────────────────────────────────────────────
DOCS_DIR = "./docs"
splitter  = StructureAwareSplitter(chunk_size=700, chunk_overlap=120)

all_texts, all_sources = [], []
print(f"\n{BOLD}Loading documents from {DOCS_DIR}...{RESET}")
for fname in sorted(os.listdir(DOCS_DIR)):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(DOCS_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    chunks = splitter.split(content, metadata={"source_file": fname})
    for c in chunks:
        all_texts.append(c.text)
        all_sources.append(fname)

print(f"  {len(all_texts)} chunks from {sum(1 for f in os.listdir(DOCS_DIR) if f.endswith('.txt'))} documents\n")

# ── Build BM25 ────────────────────────────────────────────────────────────────
sys.stdout = _null
bm25 = BM25Retriever()
bm25.index(all_texts)
sys.stdout = sys.__stdout__
print(f"  BM25 index built over {len(all_texts)} chunks\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def search(query, top_k=5):
    hits = bm25.search(query, top_k=top_k)
    return [(all_texts[i], all_sources[i], score) for i, score in hits]

def check(query, expected_keywords, label, top_k=5, note=""):
    """
    PASS  — keywords found at rank #1
    WARN  — keywords found at rank 2-top_k
    FAIL  — keywords not found in top_k
    """
    hits = search(query, top_k=top_k)
    for rank, (text, src, score) in enumerate(hits, start=1):
        text_lower = text.lower()
        if all(kw.lower() in text_lower for kw in expected_keywords):
            tag = PASS if rank == 1 else WARN
            state = "pass" if rank == 1 else "warn"
            suffix = f" — {note}" if note else ""
            print(f"  {tag}  [{label}] rank=#{rank} score={score:.2f} src={src}{suffix}")
            results[state] += 1
            results["details"].append({"q": query, "status": state, "rank": rank, "src": src})
            return
    print(f"  {FAIL}  [{label}]")
    print(f"         Query   : {query!r}")
    print(f"         Expect  : {expected_keywords}")
    top_snippet = textwrap.shorten(hits[0][0], 120) if hits else "—"
    print(f"         Top hit : {top_snippet!r}")
    results["fail"] += 1
    results["details"].append({"q": query, "status": "fail", "rank": -1, "src": ""})

def section(title, n=""):
    spacer = f" ({n} tests)" if n else ""
    print(f"\n{CYAN}{'─'*72}{RESET}")
    print(f"{BOLD}  {title}{spacer}{RESET}")
    print(f"{CYAN}{'─'*72}{RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GDN Definition & Basics  (10 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("1. GDN Definition & Basics", 10)

check("What is a GDN license?",
      ["gdn", "buy, sell"], "GDN definition")

check("What does GDN stand for in Texas motor vehicle law?",
      ["general distinguishing number"], "GDN acronym expansion")

check("Who is required to have a GDN in Texas?",
      ["gdn", "buy, sell"], "GDN who needs it")

check("How long does a GDN license last?",
      ["gdn", "2 years"], "GDN 2-year term")

check("What is the application fee for each GDN category?",
      ["700", "$700"], "GDN $700 fee")

check("What are all the types of independent GDN licenses?",
      ["motor vehicle", "motorcycle", "travel trailer"], "GDN license types list")

check("What letter do all GDN license numbers start with?",
      ["letter p", "p number"], "GDN P-number format")

check("What suffix does a wholesale dealer GDN number end with?",
      ["wholesale", "ending in w"], "GDN wholesale suffix")

check("What suffix does a travel trailer GDN license end with?",
      ["travel trailer", "ending in x"], "GDN travel trailer suffix")

check("Can I have more than one type of GDN license?",
      ["multiple", "application"], "Multiple GDN types")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Surety Bond  (8 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("2. Surety Bond Requirements", 8)

check("What is the required surety bond amount for a motor vehicle dealer?",
      ["50,000", "surety bond"], "Bond $50,000 amount")

check("Do travel trailer dealers need a surety bond?",
      ["travel trailer", "bond"], "Travel trailer bond exempt")

check("Does a trailer/semi-trailer GDN require a surety bond?",
      ["trailer", "bond"], "Semi-trailer bond exempt")

check("How long must the dealer surety bond be valid?",
      ["2 years", "bond"], "Bond 2-year term")

check("Who needs a surety bond for a GDN license?",
      ["motor vehicle dealer", "motorcycle", "surety bond"], "Bond who needs it")

check("Can a surety bond cover multiple GDN locations?",
      ["physical address", "bond"], "Bond multi-location")

check("What happens if there is a typo on the surety bond document?",
      ["rider", "typographic"], "Bond typo fix")

check("Where can a dealer find a bonding company for the surety bond?",
      ["bonding company", "department of insurance"], "Bond where to find")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Premises, Display Area & Signage  (8 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("3. Premises, Display Area & Signage", 8)

check("What are the minimum office equipment requirements for a dealership?",
      ["desk", "telephone"], "Office equipment minimum")

check("How many vehicle spaces must a dealer's display area have?",
      ["five", "display area"], "Display area 5 spaces")

check("What are the minimum sign height requirements for a dealer?",
      ["6 inches", "sign"], "Sign 6-inch height")

check("How many retail dealers can be at the same business address?",
      ["four retail"], "Max 4 retail dealers per address")

check("How many wholesale dealers can share one business structure?",
      ["eight wholesale"], "Max 8 wholesale dealers")

check("Can a dealership office be located inside an apartment or house?",
      ["apartment", "house", "office"], "Office not in apartment")

check("Does the dealer need to own or lease the dealership property?",
      ["lease or own", "physical address"], "Lease or own property")

check("Can two dealer display areas be separated by a temporary barricade?",
      ["temporary barricade", "not acceptable"], "Temporary barricade not acceptable", top_k=7)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Hours of Operation / Blue Law  (6 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("4. Hours of Operation & Blue Law", 6)

check("How many hours per day must a retail dealer be open?",
      ["4 consecutive hours"], "Retail 4-hour minimum")

check("How many days per week must a wholesale dealer be open?",
      ["2 consecutive hours", "2 days"], "Wholesale 2-day minimum")

check("Can a dealer sell vehicles on both Saturday and Sunday?",
      ["saturday", "sunday", "not both"], "Blue law both days")

check("Does the Blue Law apply to travel trailer dealers?",
      ["travel trailer", "saturday", "sunday"], "Blue law trailer exemption")

check("Does the dealer's phone need to be answered during certain hours?",
      ["8 a.m.", "telephone"], "Phone hours requirement")

check("What must a dealer post if temporarily away from the office?",
      ["date and time", "resume"], "Away from office sign")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Franchise License  (7 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("5. Franchise Dealer License", 7)

check("What two licenses does a franchise dealer need?",
      ["franchise license", "gdn"], "Franchise + GDN both needed")

check("Is a franchise dealer license transferable to another person?",
      ["not transferrable", "franchise"], "Franchise not transferable")

check("How many showrooms can share one franchise license?",
      ["separate franchise license", "showroom"], "One franchise per showroom")

check("What is the fee for a service-only franchise facility license?",
      ["200", "service-only"], "Service-only $200 fee")

check("Can a franchised dealer use the word 'leasing' in their DBA?",
      ["lease", "facilitator", "franchised dealer"], "Franchise leasing DBA exception")

check("Can multiple vehicle makes be sold in the same showroom?",
      ["line-make", "showroom"], "Multiple makes one showroom")

check("Can franchised dealers be open on both Saturday and Sunday?",
      ["saturday", "sunday"], "Franchise Saturday Sunday blue law")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — House Bill 718 / Metal Plates  (9 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("6. House Bill 718 & Metal License Plates", 9)

check("When did House Bill 718 take effect?",
      ["july 1, 2025"], "HB718 effective date")

check("What replaced paper buyer temporary tags?",
      ["general issue", "metal"], "General issue replaces buyer tags")

check("How long is an out-of-state buyer plate valid?",
      ["60 days", "out-of-state"], "Out-of-state plate 60 days")

check("What must dealers do with trade-in plates that are not reassigned?",
      ["10 days", "destroy"], "Trade-in plate 10-day destroy rule")

check("Where must dealers securely store metal license plates?",
      ["locked", "safe"], "Plate storage security")

check("What system do dealers use to order and track plate inventory?",
      ["inventory management system", "ims"], "IMS ordering system")

check("What is ePLATE and what was it formerly called?",
      ["eplate", "etag", "july 1, 2025"], "ePLATE formerly eTAG")

check("Can dealer temporary plates be used to deliver vehicles to buyers?",
      ["dealer temporary", "deliver"], "Dealer temp plate delivery prohibition")

check("Which paper tags are still valid after July 1, 2025?",
      ["vehicle transit permit", "72/144"], "Remaining valid paper tags")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Salvage Dealer License  (7 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("7. Salvage Dealer License", 7)

check("When is a salvage dealer license required?",
      ["five", "salvage", "calendar year"], "Salvage license threshold >5")

check("What is the initial fee for a salvage dealer license?",
      ["190", "$190"], "Salvage dealer $190 fee")

check("Can a salvage dealer operate from a residential address?",
      ["residence", "salvage"], "Salvage no residence")

check("What is a nonrepairable motor vehicle?",
      ["nonrepairable", "parts or scrap"], "Nonrepairable vehicle definition")

check("Can a salvage vehicle be driven on public roads?",
      ["salvage", "public roads", "not be operated"], "Salvage no public roads")

check("What system must all salvage dealers register with under federal law?",
      ["nmvtis", "national motor vehicle title"], "NMVTIS federal requirement")

check("What GDN license is needed to sell rebuilt salvage vehicles?",
      ["gdn", "rebuilt"], "GDN for rebuilt vehicle sales")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Enforcement & Penalties  (8 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("8. Enforcement & Penalties", 8)

check("What is the criminal penalty for fraudulently issuing a temporary tag?",
      ["class a misdemeanor", "4,000"], "Tag fraud Class A first offense")

check("What is the penalty for issuing 25 or more fraudulent tags?",
      ["felony", "25"], "25+ tags third-degree felony")

check("What is the administrative penalty per violation per day?",
      ["10,000", "per violation"], "$10,000/day admin penalty")

check("What is the penalty for using dealer plates for personal transportation?",
      ["dealer plate", "personal", "administrative penalty"], "Personal use of dealer plates")

check("What criminal charge applies if a dealer fails to return plates after cancellation?",
      ["class c misdemeanor", "return plates"], "Plate return Class C")

check("What is the penalty for issuing tags for vehicles not in dealer inventory?",
      ["not in dealer inventory", "class a misdemeanor"], "Inventory tag fraud")

check("Who can a dealer report tag fraud violations to?",
      ["enforcement division", "888"], "Reporting tag violations")

check("What enforcement actions can TxDMV take against a violating dealer?",
      ["suspension", "revocation", "warning letter"], "TxDMV enforcement actions")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Leasing Licenses  (5 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("9. Leasing Licenses", 5)

check("What is the difference between a lessor and a lease facilitator?",
      ["lessor", "facilitator"], "Lessor vs facilitator difference")

check("Do franchised dealers need a separate lessor license?",
      ["franchised dealer", "lessor"], "Franchise lessor exemption")

check("How long must a vehicle lease exceed to require a lessor license?",
      ["180 days"], "180-day lease threshold")

check("Can a lessor also act as a lease facilitator?",
      ["lessor", "facilitator", "own leases"], "Lessor dual role")

check("Does a lease facilitator own the vehicle they are leasing?",
      ["facilitator", "not the lessor", "owner"], "Facilitator not owner")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — Converter License  (6 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("10. Converter License", 6)

check("What is the definition of a converter?",
      ["assembles", "chassis"], "Converter definition")

check("Can a converter sell converted vehicles directly to consumers?",
      ["converter", "consumer", "franchised"], "Converter direct sales ban")

check("Does a converter need to be physically located in Texas?",
      ["located in texas", "converter"], "Converter out-of-state allowed")

check("What happens if a converter produces a vehicle that meets the definition of an ambulance?",
      ["ambulance", "manufacturer license"], "Converter ambulance → manufacturer")

check("Who pays the entire purchase price when selling a converted vehicle?",
      ["franchised dealer", "purchase price", "invoic"], "Converter dealer invoices full price")

check("Are after-market conversions on already-sold vehicles regulated by TxDMV?",
      ["after-market", "retail sale"], "After-market conversion not regulated")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — Manufacturer License  (5 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("11. Manufacturer License", 5)

check("What is the definition of a manufacturer under Texas law?",
      ["manufactures or assembles", "new motor vehicles"], "Manufacturer definition")

check("Can a manufacturer sell vehicles directly to consumers in Texas?",
      ["manufacturer", "consumer", "franchised"], "Manufacturer direct sales ban")

check("What does Texas law prohibit a manufacturer from doing with a dealership?",
      ["manufacturer", "operate", "control", "dealer"], "Manufacturer dealership prohibition")

check("Must a manufacturer be located in Texas to need a TxDMV license?",
      ["manufacturer", "regardless", "located"], "Manufacturer out-of-state must license")

check("What license must a manufacturer get if their product does not fully meet the ambulance definition?",
      ["converter license", "ambulance"], "Manufacturer ambulance → converter")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — Distributor License  (4 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("12. Distributor License", 4)

check("What is a distributor in Texas motor vehicle law?",
      ["distributor", "franchised dealer", "manufacturer"], "Distributor definition")

check("Can a distributor sell new vehicles directly to the public?",
      ["distributor", "consumer", "franchised"], "Distributor direct sales ban")

check("Can a distributor operate a dealership?",
      ["distributor", "operate or control", "dealer"], "Distributor dealership prohibition")

check("Must a distributor be licensed even if located outside Texas?",
      ["distributor", "regardless", "located"], "Distributor out-of-state must license")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — In-Transit License  (3 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("13. In-Transit License", 3)

check("Who needs an in-transit license?",
      ["drive-a-way operator", "transports"], "In-transit who needs it")

check("What plates can drive-a-way operators use?",
      ["in-transit license plates", "metal"], "In-transit metal plates")

check("What methods can a drive-a-way operator use to transport a vehicle?",
      ["full-mount", "saddle-mount", "tow-bar"], "Drive-a-way transport methods")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — Texas Lemon Law  (10 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("14. Texas Lemon Law", 10)

check("What vehicles are covered by the Texas Lemon Law?",
      ["new vehicles", "manufacturer's written warranty"], "Lemon Law covered vehicles")

check("Does the Texas Lemon Law cover used vehicles?",
      ["used vehicle", "manufacturer's original warranty"], "Lemon Law used vehicles")

check("Does the Lemon Law cover boats or farm equipment?",
      ["boats", "farm equipment", "does not cover"], "Lemon Law exclusions")

check("How many repair attempts trigger the four-times test?",
      ["four times", "same defect", "24 months"], "Lemon Law 4-times test")

check("What is the serious safety hazard test under the Lemon Law?",
      ["serious safety hazard", "twice", "24 months"], "Lemon Law safety hazard test")

check("How many total days out of service triggers the 30-day test?",
      ["30 days", "out of service"], "Lemon Law 30-day test")

check("How long does a consumer have to file a Lemon Law complaint?",
      ["six (6)", "24 months"], "Lemon Law 6-month filing deadline")

check("How much is the filing fee for a Texas Lemon Law complaint?",
      ["35", "$35"], "Lemon Law $35 filing fee")

check("What remedies can a consumer receive if they win a Lemon Law case?",
      ["refund", "replacement", "repair"], "Lemon Law remedies")

check("How many days does a hearing examiner have to issue a decision?",
      ["60 days", "hearing examiner"], "Lemon Law 60-day hearing decision")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — eLICENSING System  (5 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("15. eLICENSING System", 5)

check("What is eLICENSING used for?",
      ["apply", "renew", "online"], "eLICENSING purpose")

check("How can a dealer contact TxDMV about eLICENSING?",
      ["mvdlicensing@txdmv.gov", "888-368-4689"], "eLICENSING contact info")

check("What must licensees keep current to use eLICENSING?",
      ["email address", "elicensing"], "eLICENSING email requirement")

check("What replaced the paper-based licensing system in Texas?",
      ["elicensing", "paper-based"], "eLICENSING replaces paper system")

check("Can a dealer track their license application status through eLICENSING?",
      ["track", "progress", "submitted applications"], "eLICENSING application tracking")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — IMMV (Mobility Motor Vehicle) Dealer  (5 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("16. Independent Mobility Motor Vehicle (IMMV) Dealer", 5)

check("What is a mobility motor vehicle?",
      ["mobility motor vehicle", "disability", "wheelchair"], "IMMV definition")

check("What additional license must an IMMV dealer have beyond the GDN?",
      ["converter license", "immv"], "IMMV needs Converter license")

check("What insurance must an IMMV dealer maintain?",
      ["garage keeper", "50,000", "products-completed"], "IMMV insurance requirements")

check("What welding certification do IMMV dealers need?",
      ["welder", "american welding society"], "IMMV welder certification")

check("What federal agency must IMMV dealers register with?",
      ["national highway traffic", "nhtsa"], "IMMV NHTSA registration")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — Wholesale Dealer  (4 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("17. Wholesale Dealer", 4)

check("Can a wholesale dealer sell vehicles to individual consumers?",
      ["wholesale", "not", "retail"], "Wholesale no retail sales")

check("Does a wholesale dealer need a five-vehicle display area?",
      ["wholesale", "not required", "display"], "Wholesale no display area required")

check("What vehicles can a wholesale dealer buy and sell?",
      ["wholesale", "motor vehicles", "motorcycles", "travel trailer"], "Wholesale vehicle types")

check("How many hours per week must a wholesale dealer be open?",
      ["wholesale", "2 consecutive hours", "2 days"], "Wholesale 2-day hours")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 18 — Paraphrase / Synonym Robustness  (8 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("18. Paraphrase & Synonym Robustness", 8)

check("What bond amount is a Texas auto dealer required to maintain?",
      ["50,000", "surety bond"], "Paraphrase: auto dealer bond")

check("My new car keeps breaking down after multiple repairs — what law protects me?",
      ["lemon law", "defect", "warranty"], "Paraphrase: lemon law protection")

check("I want to sell used cars in Texas — what license do I need?",
      ["gdn", "buy, sell"], "Paraphrase: GDN for used cars")

check("What penalty applies to a car dealer who fraudulently issues paper tags?",
      ["class a misdemeanor", "tag"], "Paraphrase: tag fraud penalty")

check("When does a dealer have to stop using temporary paper tags?",
      ["july 1, 2025", "temporary"], "Paraphrase: HB718 paper tags end")

check("How does the Texas DMV help consumers with repeated vehicle repair problems?",
      ["lemon law", "manufacturer's original warranty"], "Paraphrase: lemon law description")

check("What is the annual operating cost of a basic dealer license?",
      ["700", "$700"], "Paraphrase: GDN cost")

check("Can I run a dealership from my home in Texas?",
      ["apartment", "house", "office"], "Paraphrase: home-based dealership")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 19 — Cross-Topic / Multi-Fact  (5 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("19. Cross-Topic & Multi-Fact Questions", 5)

check("What are the differences between a lessor, lease facilitator, and franchised dealer in Texas leasing?",
      ["lessor", "facilitator", "franchised dealer"], "Cross: leasing roles comparison")

check("What licenses are needed to sell both new and used vehicles at a Texas dealership?",
      ["franchise", "gdn"], "Cross: franchise + GDN for new+used")

check("What makes a converted vehicle different from a manufactured vehicle?",
      ["converter", "manufacturer", "chassis"], "Cross: converter vs manufacturer")

check("How do the penalties for tag fraud compare between first offense and multiple tags?",
      ["class a misdemeanor", "felony"], "Cross: tag fraud penalty comparison")

check("What is the difference between a salvage and a nonrepairable motor vehicle?",
      ["salvage", "nonrepairable", "rebuilt"], "Cross: salvage vs nonrepairable")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 20 — Dealer Plate Fees & Specifics  (3 questions)
# ═══════════════════════════════════════════════════════════════════════════════
section("20. Dealer Plate Fees & Details", 3)

check("What is the fee for each metal dealer license plate?",
      ["90", "$90"], "Dealer plate $90 fee")

check("What payment methods are accepted in eLICENSING?",
      ["credit card", "echeck"], "eLICENSING payment methods")

check("When can a dealer place a new plate order in the IMS?",
      ["50%", "quarterly"], "IMS reorder at 50% threshold")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total  = results["pass"] + results["fail"] + results["warn"]
found  = results["pass"] + results["warn"]   # top-5 recall

print(f"\n{'='*72}")
print(f"{BOLD}  RETRIEVAL ACCURACY — FULL REPORT  ({total} questions){RESET}")
print(f"{'='*72}")
print(f"  {GREEN}Precision@1  (rank #1){RESET}       : {results['pass']:3d} / {total}  =  {100*results['pass']//total if total else 0}%")
print(f"  {YELLOW}Top-5 Recall  (in top 5){RESET}     : {found:3d} / {total}  =  {100*found//total if total else 0}%")
print(f"  {RED}Not Found     (complete miss){RESET} : {results['fail']:3d} / {total}  =  {100*results['fail']//total if total else 0}%")
print(f"{'='*72}")

# ── Per-section breakdown ─────────────────────────────────────────────────────
SECTION_SIZES = [10, 8, 8, 6, 7, 9, 7, 8, 5, 6, 5, 4, 3, 10, 5, 5, 4, 8, 5, 3]
SECTION_NAMES = [
    "GDN Basics", "Surety Bond", "Premises & Signage", "Hours/Blue Law",
    "Franchise", "HB718 Metal Plates", "Salvage Dealer", "Enforcement",
    "Leasing", "Converter", "Manufacturer", "Distributor", "In-Transit",
    "Lemon Law", "eLICENSING", "IMMV Dealer", "Wholesale", "Paraphrase",
    "Cross-Topic", "Plate Fees",
]

print(f"\n  {'Section':<30} {'Pass':>5} {'Warn':>5} {'Fail':>5} {'Recall%':>8}")
print(f"  {'─'*58}")
idx = 0
for i, (name, size) in enumerate(zip(SECTION_NAMES, SECTION_SIZES)):
    sec = results["details"][idx:idx+size]
    p = sum(1 for d in sec if d["status"] == "pass")
    w = sum(1 for d in sec if d["status"] == "warn")
    f = sum(1 for d in sec if d["status"] == "fail")
    recall = 100*(p+w)//size if size else 0
    bar = f"{GREEN}{'█'*p}{RESET}{YELLOW}{'░'*w}{RESET}{RED}{'·'*f}{RESET}"
    print(f"  {name:<30} {p:5d} {w:5d} {f:5d}  {recall:5d}%  {bar}")
    idx += size

print(f"\n  {'─'*58}")
print(f"  {'TOTAL':<30} {results['pass']:5d} {results['warn']:5d} {results['fail']:5d}  {100*found//total if total else 0:5d}%")

# Save JSON
with open("retrieval_report.json", "w") as f:
    json.dump({**results, "total": total, "recall": found, "precision1": results["pass"]}, f, indent=2)
print(f"\n  Report saved → retrieval_report.json")

if results["fail"] > 0:
    print(f"\n{RED}  {results['fail']} tests failed — see details above.{RESET}\n")
    sys.exit(1)
else:
    print(f"\n{GREEN}  All questions answered by the retriever!{RESET}\n")
