"""
Document Categories & Query Classifier
========================================
Maps filenames to categories and classifies incoming queries
to enable metadata-filtered retrieval in ChromaDB.
"""

# ============================================================================
# FILENAME → CATEGORY MAPPING
# Based on your actual document corpus
# ============================================================================

CATEGORY_MAP = {

    # ── Licensing & eLICENSING Guides ──────────────────────────────────────
    "licensing_elicensing_guides": [
        "elicensing-userguide_independent-gdn-licensees",
        "elicensing_user_guide_for_new_independent_gdn_licenses",
        "elicensing_user_guide_for_independent_gdn_amendment_licenses",
        "elicensing_userguide_franchise_dealer_licenses",
        "elicensing-user_guide_lessor_licenses",
        "elicensing-user-guide-for-salvage-dealer-licenses",
        "elicensing-userguide_converterlicense",
        "elicensing-userguide_lease_facilitator",
        "elicensing-userguide_mfg_mfrep_licenses",
        "elicensing_user_guide_intransit_licenses",
        "_dealers_licensing",
        "_dealers_licensing_independent-gdn",
        "_dealers_licensing_franchise",
        "_dealers_licensing_manufacturer",
        "_dealers_licensing_distributor",
        "_dealers_licensing_converter",
        "_dealers_licensing_leasing",
        "_dealers_licensing_in-transit",
        "_dealers_licensing_salvage-dealer",
    ],

    # ── eLICENSING Quick Start Guides ───────────────────────────────────────
    "elicensing_quickstart": [
        "elicensing-quickstart_newlicense",
        "elicensing-quickstart_admins",
        "elicensing-quickstart_addinganattorney",
        "elicensing-quickstart_licenseeaccessrequest",
        "elicensing-quick-start-guide-protest-dealer",
        "elicensing-quick-start-guide-protest-mfr-dist",
        "elicensing-quick-start-guide-protest-termination",
        "_dealers_licensing_elicensing-resources",
        "_dealers_licensing_elicensing-resources 2",
    ],

    # ── HB 718 Metal Plates & webDEALER ────────────────────────────────────
    "hb718_metal_plates": [
        "_dealers_hb718",
        "dealer_reference_brochure",
        "dealer-plate-faqs_hb718",
        "dealer_plate_issuance_guidelines",
        "elicensing_plate_app_guide-may_2025",
        "elicensing_plate_app_guide-may_2025 2",
        "hb_718_support-resources-for-dealers_06272025",
        "hb_718_support-resources-for-counties_06262025",
        "hb-718_lobby_slides",
        "enf%20hb%20718%20faq%20dealer%20handout%20032125",
        "internet_down_receipt_instructions_before_1_july_2025",
        "inventory_management_system_access_via_webdealer",
        "rts_25.3_webdealer_addendum",
        "texas_license_plates_law_enforcement_guide",
        "webdealer_4.1.1_dealer_user_guide_0",
        "what%20dealers%20need%20to%20know%20about%20house%20bill%20718",
        "2025-06-10_texas_license_plate_changes_press_release",
        "memo_exempt-plates-process_07092025",
        "urgent-action_required_for_motor_vehicle_dealers",
        "urgent_action_required_motor_vehicle_dealers",
    ],

    # ── Enforcement, Compliance & Penalties ─────────────────────────────────
    "enforcement_compliance_penalties": [
        "texas_tag_penalties",
        "enf-salvage_notice",
        "enf-non-repairable_notice",
        "enf-sal-221",
        "enf-sal-221-nr",
        "sampleletter",
        "_motorists_consumer-protection_lemon-law",
    ],

    # ── Forms & Applications ────────────────────────────────────────────────
    "forms_applications": [
        "dmv_lf131ab",
        "dmv_lf621",
        "lf610",
        "lf706",
        "ld001",
        "li001",
        "mvd_lf629",
        "mvd_lf630",
        "mvd_lf631",
        "dealership-premises-checklist-mvd_lf628",
        "title_application_processing_guidelines",
    ],

    # ── Spanish Language Documents ───────────────────────────────────────────
    "spanish_language": [
        "acci%c3%93n_urgente_requerida_concesionarios_de_veh%c3%adculos_motorizados",
        "urgente-acci%c3%b3n_requerida_para_los_concesionarios_de_veh%c3%adculos_motorizados",
        "directrices_para_el_procesamiento_de_solicitudes_de_t%c3%adtulo",
        "directrices_para_la_emisi%c3%b3n_de_placas_de_concesionario",
        "referencia_del_concecionardio",
        "accion_urgente_requerida",
        "urgente-accion_requerida",
        "directrices_para",
    ],

    # ── General Reference & Summaries ───────────────────────────────────────
    "general_reference": [
        "summary",
        "summary 2",
        "summary 3",
        "summary 4",
        "txdmv_compact_with_texans",
        "txdmv_compact_with_texans 2",
        "txdmv_compact_with_texans 3",
        "gemini_upload_manifest",
    ],
}

# ── Reverse map: stem → category (built at import time) ──────────────────────
STEM_TO_CATEGORY = {}
for category, stems in CATEGORY_MAP.items():
    for stem in stems:
        STEM_TO_CATEGORY[stem.lower()] = category


def get_category(filename: str) -> str:
    """
    Given a filename, return its category.
    Falls back to 'general_reference' if not found.
    """
    import re
    # Normalize: lowercase, remove extension, strip spaces
    stem = filename.lower()
    stem = re.sub(r'\.(pdf|txt|html|json)$', '', stem)
    stem = stem.strip()

    # Exact match first
    if stem in STEM_TO_CATEGORY:
        return STEM_TO_CATEGORY[stem]

    # Partial match — check if any known stem is contained in filename
    for known_stem, category in STEM_TO_CATEGORY.items():
        if known_stem in stem or stem in known_stem:
            return category

    return "general_reference"


# ============================================================================
# QUERY CLASSIFIER
# Maps user queries to the most relevant category for metadata filtering
# ============================================================================

# Keywords that strongly indicate a specific category
CATEGORY_KEYWORDS = {
    "licensing_elicensing_guides": [
        "gdn", "general distinguishing number", "what is a gdn",
        "gdn definition", "who needs a gdn", "apply for license",
        "license application", "independent dealer license", "franchise license",
        "surety bond", "dealer license", "new license", "renew license",
        "renewal", "elicensing account", "ownership", "background check",
        "criminal history", "signage", "business location", "display area",
        "gathering information", "amendment", "in-transit license",
        "converter license", "lessor license", "salvage dealer license",
        "lease facilitator", "manufacturer license",
        "cost to open", "opening a dealership", "startup costs",
        "total cost", "all costs", "fees involved", "how much does it cost",
        "cost of license", "application fee", "license fee",
        "premises", "dealership location", "office space",
        "sign requirements", "permanent sign",
        "independent dealership", "new dealership", "open a dealership",
        "costs involved", "opening costs", "what does it cost",
        "how much to open", "how much to start",
    ],
    "elicensing_quickstart": [
        "how to register", "create account", "login", "add user",
        "administrator", "protest", "quick start", "access request",
        "attorney access", "admin", "account setup",
    ],
    "hb718_metal_plates": [
        "hb 718", "hb718", "house bill 718", "metal plate", "metal tag",
        "temporary plate", "buyer plate", "dealer plate", "paper tag",
        "temporary tag", "webdealer", "plate application", "plate order",
        "internet down", "internet-down", "inventory management",
        "out-of-state buyer plate", "provisional plate", "plate storage",
        "plate security", "plate delivery", "july 2025", "july 1",
        "general issue plate", "standard dealer plate",
    ],
    "enforcement_compliance_penalties": [
        "penalty", "penalties", "fine", "misdemeanor", "felony",
        "violation", "enforcement", "fraud", "fraudulent", "suspension",
        "revocation", "compliance", "lemon law", "salvage notice",
        "non-repairable", "criminal offense", "administrative penalty",
        "tag fraud", "report violation",
    ],
    "forms_applications": [
        "form", "application form", "lf610", "lf706", "mvd form",
        "title application", "premises checklist", "fill out",
        "download form", "submit form",
    ],
    "spanish_language": [
        "spanish", "español", "en español", "concesionario",
        "placas", "urgente",
    ],
}


def classify_query(query: str) -> str:
    """
    Classify a query into one of the document categories.
    Returns the best matching category, or None for no filter (search all).

    Strategy:
    - Count keyword matches per category
    - Return the winner if it has a clear lead
    - Return None (no filter) if ambiguous or no strong match
    """
    query_lower = query.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                # Longer keywords = more specific = higher weight
                scores[category] += len(keyword.split())

    # Find the best scoring category
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Only filter if there's a clear winner (score >= 2)
    if best_score < 2:
        return None  # No filter — search everything

    # Check if second-best is close (ambiguous query)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[1] >= best_score * 0.8:
        return None  # Too close to call — search everything

    return best_cat


def get_filter_categories(query: str) -> list:
    """
    Returns a list of categories to filter on.
    For most queries: one category.
    For broad queries: None (no filter).
    """
    primary = classify_query(query)
    if primary is None:
        return None  # No filter

    # Always include general_reference as a fallback alongside primary
    categories = [primary]
    if primary != "general_reference":
        categories.append("general_reference")

    return categories


# ============================================================================
# SMART top_k SELECTOR
# Returns the right number of chunks to retrieve based on query type.
#
# Simple question  → top_k = 5  (fast, focused)
# Aggregation      → top_k = 10 (broader, catches scattered info)
# Comparison       → top_k = 12 (needs chunks from multiple docs)
# ============================================================================

# Words that signal the query needs MORE chunks
AGGREGATION_SIGNALS = [
    # "all" type questions
    "all costs", "all fees", "all requirements", "all documents",
    "all changes", "all types", "all steps", "all violations",
    "everything needed", "complete list", "full list",

    # "what changed" type
    "what changed", "what changes", "what is new", "new requirements",
    "after july", "after 2025", "effective date",

    # Comparison type
    "difference between", "compare", "vs", "versus",
    "what is the difference",

    # Cost aggregation
    "total cost", "all costs", "cost to open", "opening costs",
    "startup costs", "fees involved", "how much does it cost to",

    # Multi-step processes
    "step by step", "all steps", "entire process", "full process",
    "from start to finish",
]


def get_smart_top_k(query: str, default: int = 5) -> int:
    """
    Returns the appropriate top_k for a query.

    Logic:
    - Aggregation/comparison queries → 10-12 chunks needed
    - Simple direct questions        → 5 chunks is enough

    Example:
        "What is the surety bond amount?"      → 5  (simple lookup)
        "What are ALL the costs to open?"      → 10 (aggregation)
        "Difference between franchise vs GDN?" → 12 (comparison)
    """
    query_lower = query.lower()

    # Check comparison signals first (highest top_k)
    comparison_words = ["difference between", "compare", " vs ", "versus",
                        "what is the difference"]
    for word in comparison_words:
        if word in query_lower:
            print(f"   📊 Smart top_k: 12 (comparison query detected: '{word}')")
            return 12

    # Check aggregation signals (medium-high top_k)
    for signal in AGGREGATION_SIGNALS:
        if signal in query_lower:
            print(f"   📊 Smart top_k: 10 (aggregation query detected: '{signal}')")
            return 10

    # Check for "all" or "every" which often signals aggregation
    if query_lower.startswith("what are all") or \
       query_lower.startswith("list all") or \
       "every type" in query_lower or \
       "every kind" in query_lower:
        print(f"   📊 Smart top_k: 10 (list/all query detected)")
        return 10

    # Default for simple direct questions
    print(f"   📊 Smart top_k: {default} (direct question)")
    return default
