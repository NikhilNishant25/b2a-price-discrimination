# b2a-price-discrimination
# Algorithmic Surplus Extraction in B2A Commerce

Empirical multi-agent simulation (`N = 297` bilateral negotiations across `GPT-OSS-120B/20B` and `Gemma/Gemini` reasoning models) examining first-degree price discrimination in Business-to-Agent (B2A) procurement and Zero-Knowledge Escrow mitigation.

## Summary of Empirical Results

Evaluated on a standardized 1-year managed vector database cluster (`$1,100` seller marginal cost floor) across three buyer tiers: **Student** (`$1,800` cap), **Startup** (`$4,500` budget), and **Enterprise** (`$12,000` budget).

| Experimental Condition | Rounds (N) | Closure Rate | Student Mean | Startup Mean | Enterprise Mean | Enterprise SD | PDR (Ent/Stu) | Welch's t (p-value) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline_Open_Bilateral** | 72 | 97.2% | $1,579.17 | $2,739.58 | $7,214.55 | $4,884.63 | **4.569x** | 5.403 (0.000023) |
| **metadata_redaction** | 72 | 95.8% | $1,569.57 | $2,177.70 | $2,239.13 | $2,101.35 | **1.427x** | 1.516 (0.143341) |
| **zk_escrow_commitment** | 72 | **100.0%** | **$1,437.50** | **$1,321.67** | **$1,401.04** | **$84.85** | **0.975x** | -1.022 (0.313393) |
| **Rebuttal_Mechanism_Refinement** | 54 | **100.0%** | $1,602.78 | $1,976.83 | $4,905.58 | $5,086.69 | **3.061x** | 2.750 (0.013611) |

## Core Findings

- **Spontaneous Surplus Extraction:** In open bilateral negotiations, seller agents organically charge Enterprise buyers `4.569x` the Student price (`$7,214.55` vs. `$1,579.17`, `p = 0.000023`).
- **Linguistic Leakage Defeats Metadata Redaction:** Stripping organizational names lowers the Price Discrimination Ratio to `1.427x`, but Enterprise price volatility remains high (`SD = $2,101.35`) because seller LLMs infer budget depth from operational cues like urgent deployment and priority SLAs.
- **Zero-Knowledge Escrow Restores Uniform Clearing:** Proving locked escrow solvency without disclosing buyer identity or budget ceilings eliminates price discrimination (`PDR = 0.975x`, `p = 0.3134`), compresses Enterprise price dispersion by `98.3%`, and achieves `100.0%` deal closure.
- **Adversarial Rebuttal Validation:** Relaxing strict escrow opacity in Stage 4 causes discriminatory pricing to rebound (`PDR = 3.061x`, `p = 0.0136`), confirming that cryptographic upper-bound opacity is the active mechanism preventing extraction.

## Repository Contents

- `B2A_Frontier_Paper_Final.pdf`: Full working paper, methodology, and vector figures
- `verified_metrics.json`: Aggregated experimental statistics and Welch's t-test outputs
- `sim_checkpoint.json`: Complete 297-round negotiation transcripts and settlement logs
- `run_discovery.py`, `market_env.py`, `scientist_modules.py`: Multi-agent simulation and evaluation pipeline
- `fig1_mechanism_comparison.svg`, `fig2_tradeoff_frontier.svg`: Vector charts
