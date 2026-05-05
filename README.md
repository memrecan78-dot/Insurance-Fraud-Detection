# Insurance-Fraud-Detection
PoC for Fraud Detection

================================================================================
INSURANCE CLAIM FRAUD DETECTION SYSTEM
Multi-Agent RAG Architecture — Design Notes
================================================================================

================================================================================
Author: EMRE CAN
================================================================================


================================================================================
1. SYSTEM OVERVIEW
================================================================================

This system extends a pre-built insurance claims management platform with a new
fraud detection capability using Retrieval-Augmented Generation (RAG). The core
innovation is the FraudPatternDetector — a vector-based similarity engine that
compares incoming claims against a knowledge base of known fraud patterns using
TF-IDF vectorization and cosine similarity.

The system inherits a fully-functioning 4-agent complaint resolution platform
from demo/demo.py and adds one new specialist agent (FraudDetectionAgent) plus
an enhanced orchestrator (EnhancedOrchestrator) that integrates fraud checking
into every claim processing workflow.


================================================================================
2. AGENT ARCHITECTURE
================================================================================

The system uses 5 agents total — 3 inherited from demo, 2 new in this lesson.

┌─────────────────────────────────────────────────────┐
│              EnhancedOrchestrator (NEW)             │
│  Inherits ComplaintResolutionOrchestrator           │
│  Adds: handle_claim_with_fraud_check nested @tool   │
└────┬────────────┬────────────┬───────────┬──────────┘
     │            │            │           │
┌────▼───┐  ┌────▼───┐  ┌────▼───┐  ┌────▼────────────┐
│ Claim  │  │Customer│  │Medical │  │ Fraud           │
│Processor│  │Service │  │Review  │  │ investigator    │
│(demo)  │  │(demo)  │  │(demo)  │  │ (NEW)           │
└────────┘  └────────┘  └────────┘  └─────────────────┘


ENHANCEDORCHESTRATOR (new — inherits from ComplaintResolutionOrchestrator)
  Role: Central coordinator. Inherits all existing claim and complaint handling
  from the demo orchestrator, then adds fraud detection to every new claim
  submission via a nested @tool called handle_claim_with_fraud_check.

  New tool:
  - handle_claim_with_fraud_check: Nested @tool defined inside __init__.
    Calls ClaimProcessorAgent to process the claim, then immediately calls
    FraudDetectionAgent to assess fraud risk. Returns combined result including
    claim status, fraud risk level, and action recommendation.

  Access control:
  - Uses AccessControl.can_access() with PrivacyLevel (PUBLIC / AGENT / ADMIN)
    to gate which fraud pattern details each agent can see.


CLAIMPROCESSINGAGENT (inherited from demo)
  Role: Receives new claim submissions, validates them, and records approved or
  denied decisions in the in-memory database.

  Tools: process_new_claim


CUSTOMERSERVICEAGENT (inherited from demo)
  Role: Handles patient complaints about denied or disputed claims.

  Tools: submit_complaint, respond_to_complaint, get_complaint_history


MEDICALREVIEWAGENT (inherited from demo)
  Role: Reviews claims requiring clinical judgment, checks procedure codes
  against patient history.

  Tools: get_claim_details, find_similar_claims, search_knowledge_base


FRAUDDETECTIONAGENT (new — ToolCallingAgent)
  Role: The specialist fraud investigator. Retrieves claim details and patient
  history, then runs check_claim_for_fraud which feeds the claim through the
  FraudPatternDetector pipeline.

  Access level: PrivacyLevel.AGENT (can see all AGENT-level fraud patterns)

  Tools:
  - check_claim_for_fraud: Main fraud detection tool. Calls
    FraudPatternDetector.detect_fraud_indicators() which runs TF-IDF + cosine
    similarity against the 7 known fraud patterns, plus rule-based checks.
  - get_claim_details: Retrieves full claim record from database.
  - get_patient_info: Retrieves patient profile and claim history.
  - retrieve_claim_history: Gets recent claims for the same patient to
    provide context for pattern matching (e.g. detecting rapid resubmission).
  - search_knowledge_base: Searches the VectorKnowledgeBase for relevant
    medical and policy information.


================================================================================
3. THE RAG FRAUD DETECTION ENGINE
================================================================================

The core new concept in Lesson 7 is Retrieval-Augmented Generation applied to
fraud detection. Here is how it works step by step:

STEP 1 — Knowledge base creation
  Seven fraud patterns are defined as dictionaries with: pattern_id,
  pattern_name, description, indicators, severity, and privacy_level.
  Patterns range from FP-001 (Rapid Claim Submission) to FP-007 (Provider
  Shopping), covering medium, high, and critical severity levels.

STEP 2 — Vectorization (FraudPatternDetector.update_patterns)
  Each fraud pattern's name + description + indicators text is concatenated
  into a single string. TfidfVectorizer from scikit-learn converts all 7
  pattern strings into numerical vectors — a matrix where each row is a
  pattern and each column represents a word's weighted importance.

  TF-IDF (Term Frequency-Inverse Document Frequency) weights words by how
  often they appear in one document vs. how rare they are across all documents.
  "duplicate" appearing only in the duplicate billing pattern gets high weight;
  "claim" appearing in all patterns gets low weight.

STEP 3 — Claim vectorization and similarity scoring
  When a new claim arrives, its procedure code, amount, patient ID, status,
  and recent procedure history are combined into a text string. The same
  TfidfVectorizer transforms this into a vector in the same space as the
  patterns. cosine_similarity() measures the angle between the claim vector
  and each of the 7 pattern vectors, returning a similarity score 0.0–1.0.

STEP 4 — Pattern matching
  Any pattern with similarity > 0.1 (the threshold) is flagged as a match.
  Matches are sorted by similarity score descending. Access control is checked:
  only patterns at or below the requester's privacy level are shown.

STEP 5 — Rule-based checks
  In addition to vector similarity, hard-coded rules also run:
  - high_amount: flags claims over $400 (confidence 0.6)
  - repeat_procedure: flags same procedure code appearing 2+ times in recent
    patient history (confidence 0.7)

STEP 6 — Risk scoring
  Pattern matches are weighted by severity multiplier (low=0.3, medium=0.6,
  high=0.8, critical=1.0) × similarity score, then summed. Rule indicator
  confidence scores are added. Final thresholds:
    total < 0.3  → 'low'     → Proceed with normal processing
    total < 0.7  → 'medium'  → Flag for manual review
    total < 1.2  → 'high'    → Requires investigator review
    total >= 1.2 → 'critical'→ Suspend and initiate investigation


================================================================================
4. THE 7 FRAUD PATTERNS
================================================================================

ID      Name                      Severity   Description
------  ------------------------  ---------  ----------------------------------
FP-001  Rapid claim submission    medium     Multiple claims, same patient,
                                             short timeframe
FP-002  Procedure upcoding        high       Billing for more complex procedures
                                             than performed
FP-003  Duplicate billing         high       Same service billed multiple times
                                             on the same date
FP-004  Phantom billing           critical   Services never actually provided
FP-005  Unusual procedure freq.   medium     Statistically anomalous repetition
                                             of the same procedure
FP-006  Service date manipulation high       Dates altered to maximise
                                             reimbursement or avoid limits
FP-007  Provider shopping         medium     Multiple providers, same condition,
                                             short timeframe


================================================================================
5. THE IN-MEMORY DATABASE
================================================================================

Unlike production systems, Lesson 7 uses Python dictionaries in RAM:

  database.patients    = {patient_id: PatientRecord}
  database.claims      = {claim_id:   Claim}
  database.complaints  = {complaint_id: ComplaintRecord}

DataGenerator.populate_database(num_patients=20, num_claims=50, num_complaints=10)
generates synthetic data on every run. All data resets when the script ends —
there are no files written to disk. This is intentional for the learning
exercise; production systems would use PostgreSQL + Redis.

Privacy levels control data access:
  PUBLIC  → basic claim status only
  AGENT   → full claim details + fraud pattern matches
  ADMIN   → all data including internal decision reasons


================================================================================
6. KEY PYTHON AND LIBRARY CONCEPTS INTRODUCED
================================================================================

TfidfVectorizer (scikit-learn)
  Converts text strings into numerical vectors. fit_transform() builds the
  vocabulary from training documents and transforms them simultaneously.
  transform() converts new unseen documents using the existing vocabulary.
  Important: transform() must be called after fit_transform() — the vectorizer
  must have seen the training data before it can process new inputs.

cosine_similarity (scikit-learn)
  Measures similarity between two vectors as the cosine of the angle between
  them. Returns 1.0 for identical vectors, 0.0 for completely unrelated.
  Unlike Euclidean distance, cosine similarity is length-agnostic — a short
  claim description and a long one can still score high if they share the
  same important words.

super().__init__() with nested @tool
  EnhancedOrchestrator defines handle_claim_with_fraud_check as a nested
  function decorated with @tool inside __init__, before calling
  ComplaintResolutionOrchestrator.__init__(). This allows the nested tool
  to close over self.claim_processor and self.fraud_detector via Python's
  closure mechanism, giving it access to the specialist agents without
  making them global variables.

PrivacyLevel and AccessControl
  Enum-like class where each level has a numeric rank. AccessControl.can_access()
  returns True only if the requester's level is >= the resource's required level.
  This prevents AGENT-level callers from seeing ADMIN-only data.


================================================================================
7. DESIGN DECISIONS
================================================================================

Why inherit from ComplaintResolutionOrchestrator?
  The demo already has a working 4-agent system handling claims, complaints,
  and medical reviews. Inheritance lets the lesson add fraud detection without
  rewriting existing functionality. EnhancedOrchestrator gets all parent
  capabilities for free and adds exactly one new capability: fraud checking.

Why TF-IDF + cosine similarity instead of LLM embeddings?
  TF-IDF is fast, deterministic, and requires no API calls. For a keyword-heavy
  domain like insurance fraud patterns, TF-IDF works well because fraud
  indicators are highly specific vocabulary ("duplicate billing", "phantom",
  "upcoding"). In production, dense embeddings (OpenAI, Sentence-BERT) would
  capture semantic similarity better, but TF-IDF serves the learning objective.

Why a threshold of 0.1 for pattern matching?
  Low threshold — catches more patterns, risks more false positives. The risk
  scoring system then weights matches by severity, so a low-similarity match
  to a critical pattern still contributes meaningfully to the final score.
  Raising the threshold to 0.25+ would reduce noise but might miss subtle
  fraud signals.

Why nested @tool rather than module-level @tool for fraud check?
  handle_claim_with_fraud_check needs access to self.claim_processor and
  self.fraud_detector. Module-level functions can't access instance variables.
  The nested function pattern (used in Lessons 5 and 6 as well) is the
  smolagents-idiomatic way to give tools access to agent instances via closure.


================================================================================
8. KNOWN ISSUES AND LIMITATIONS
================================================================================

Claim ID lookup bug
  The code that identifies the newly-created claim after ClaimProcessorAgent
  runs searches for any claim belonging to the patient rather than the specific
  newly-created one. In practice this means the fraud check sometimes runs on
  an older existing claim rather than the newly submitted one. This affects
  demo accuracy but not the fraud detection logic itself.

access_level=None default
  The LLM occasionally passes access_level=None to tools that have a default
  value of PrivacyLevel.AGENT. When None is passed explicitly it overrides the
  default and causes AccessControl.can_access() to reject the request. Fixed by
  adding "Always use access_level='agent' — never pass None" to agent prompts.

Both claims scored 'medium' in demo run
  Because the demo uses randomly generated synthetic data and TF-IDF is
  vocabulary-sensitive, many claims produce similar cosine similarity scores
  against the pattern vectors. The risk tiers would differentiate better with
  a richer claim text representation including diagnosis codes and provider IDs.


================================================================================
9. SUGGESTIONS FOR FURTHER IMPROVEMENT
================================================================================

1. DENSE EMBEDDINGS FOR BETTER SEMANTIC MATCHING
   Replace TfidfVectorizer with OpenAI text-embedding-3-small or Sentence-BERT.
   Dense embeddings capture semantic meaning ("overcharging" would match
   "upcoding" even without exact word overlap), significantly improving pattern
   matching recall for novel fraud language.

2. PERSISTENT VECTOR STORE
   Replace the in-memory pattern_vectors numpy array with ChromaDB or Pinecone.
   This allows fraud patterns to be added, updated, or retired without
   restarting the system, and supports millions of historical claims as
   training examples for the vectorizer.

3. TEMPORAL FRAUD RULES
   The current rule-based checks only look at procedure code repetition. Adding
   time-window analysis (e.g. 3 claims within 14 days = high suspicion) would
   catch Rapid Claim Submission (FP-001) and Provider Shopping (FP-007) far more
   reliably than TF-IDF text similarity alone.

4. FEEDBACK LOOP FROM INVESTIGATORS
   Add a tool that lets human investigators mark claims as confirmed fraud or
   false positive. These labels could periodically retrain the TF-IDF vectorizer
   on real confirmed fraud cases, continuously improving detection accuracy.

5. AUDIT TRAIL
   Every fraud check result should be logged with timestamp, claim ID, patterns
   matched, score, and recommendation. Currently results are returned but not
   persisted. An audit log is a legal requirement in production insurance systems.


================================================================================
END OF DESIGN NOTES
================================================================================
