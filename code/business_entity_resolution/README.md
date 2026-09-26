# Business Entity Resolution Pipeline

High-Performance, Precision-Weighted Entity Resolution Pipeline for the ML Challenge 2026.

## Architecture Overview
The pipeline resolves business identities across 3 independent, noisy sources (`Source 1`, `Source 2`, and `Source 3`) referring to real-world business entities.

The solution operates as a high-precision multi-stage system:
1. **Partitioning & Candidate Generation (Dual IDF Multi-Pass Blocking)**:
   - Records are partitioned strictly by `country` (US, India, France), eliminating cross-country comparisons with zero recall loss.
   - For India, a specialized Brahmi Unicode phonetic transliteration engine converts Indic scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam, Gujarati, Bengali, Odia) into Latin phonetics, extracting vowel-free consonant skeletons (`NSK`) to bridge cross-script spelling variants.
   - For France, specialized French corporate entity legal suffixes (`sarl`, `sas`, `sasu`, `sa`, `sci`, `eurl`, `snc`, `gie`, `scp`, `sel`, `sem`) and address abbreviations (`r`, `av`, `bd`, `pl`, `all`, `imp`, `crs`, `rte`, `ch`, `bis`, `ter`) are normalized.
   - Inverted indexing with Inverse Document Frequency (IDF) candidate scoring:
     - Full normalized names (`NF:...`)
     - Distinctive name tokens (`NW:...`) and prefixes (`NP:...`)
     - Phonetic consonant skeletons (`NSK:...`)
     - Address numbers + significant address words (`ANW:...`)
     - Address word pairs (`AAW:...`)
     - Rare address words ($\ge 6$ chars) (`RAW:...`)
     - Name word + address word pairs (`NAW:...`)
   - Distractor keys with posting list length $> 1,500$ are pruned (`max_key_len=1500`).
   - Candidates are ranked by cumulative key IDF (`top_name=16`, `top_addr=8`), reaching 98.32% candidate recall ceiling while averaging 20.83 candidates per entity (strictly within the competition limit of 10–50).
   - Produces `output/candidate_pairs.tsv`.

2. **Precision-Weighted Pairwise Matching Model & Country Calibration**:
   - 14-dimensional feature vector combining Levenshtein distance, token sort/set ratio, partial ratio, token Jaccard similarity, address null indicator, address token set/sort ratio, address number equality, and interaction features.
   - HistGradientBoostingClassifier trained with address-colocated hard-negative mining to defeat multi-tenant office building false merges.
   - Country and address-calibrated dual decision boundaries:
     - US: $\tau_{\text{addr}} = 0.80$, $\tau_{\text{no\_addr}} = 0.84$
     - India: $\tau_{\text{addr}} = 0.82$, $\tau_{\text{no\_addr}} = 0.78$
     - France: $\tau_{\text{addr}} = 0.80$, $\tau_{\text{no\_addr}} = 0.84$

3. **Competitive Bipartite Target Disambiguation (Winner-Takes-All)**:
   - Enforces the ground-truth exclusivity constraint that target records map to at most one Source 1 entity.
   - Eliminates 651,952 false cross-merges, restoring singleton integrity (42,083 singletons protected).
   - Produces `output/matching_results.tsv` (7,060,284 final matches).

## Setup & Environment
Ensure Python 3.9+ is installed. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline End-to-End
To regenerate both output files (`output/candidate_pairs.tsv` and `output/matching_results.tsv`) from the root directory:

```bash
python3 code/business_entity_resolution/src/run_pipeline.py
```

Or run individual steps:
1. **Train Model** (generates `src/matching_model.joblib`):
   ```bash
   python3 code/business_entity_resolution/src/train.py
   ```
2. **Run Inference on Test Set**:
   ```bash
   python3 code/business_entity_resolution/src/infer.py
   ```

## Validating Outputs
To validate the outputs against official competition constraints:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
