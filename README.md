# BRSM Project: Sentence Memorability Study

A research data analysis pipeline for a psycholinguistic study examining how sentence structure (active vs. passive voice) and noun imageability affect human memory performance. Participants were exposed to sentences and later tested on recognition. The study investigates the effects of:

- **Sentence voice**: Active vs. Passive
- **Noun imageability combinations**: HH, HL, LH, LL (H = High imageability, L = Low imageability)
- **Practice effects**: Changes in memorability across experimental blocks

## Participant Summary

- **Total participants processed**: 114
- **Valid blocks retained**: 339 / 342 (99.1%)
- **Complete participants (all 3 blocks)**: 112
- **Exclusion rate**: 0.9%

---

## Required: Raw Data Folder

The `NewLogsAnonymized/` folder containing raw participant `.log` files **must be present in the root project directory** before running any code.

```
BRSM_project/
├── NewLogsAnonymized/     
│   ├── 232.log
│   ├── 235.log
│   └── ... 
├── brsm-project.ipynb
└── ...
```

---

## Project Structure

```
BRSM_project/
├── NewLogsAnonymized/                    
├── brsm-project.ipynb                   
├── preprocess_data.py                   
├── processed_memorability_data.csv      
├── exclusion_log.txt                    
└── output/                              
    └── descriptive_analysis_outputs/
                                    ├── table2_overall_stats.csv
                                    ├── table3_per_condition.csv
                                    ├── table4_wr_breakdown.csv
                                    ├── per_condition_per_participant.csv
                                    ├── per_participant_overall.csv
                                    ├── figure1_corrected_score.png      
                                    ├── figure2_reaction_time.png        
                                    └── figure3_wr_proportion_yes.png    
    ├── clean_trials.csv                 
    ├── memorability_scores.csv          
    ├── descriptives_table.csv        
    ├── hypothesis_results.csv         
    ├── results_report.txt               
    │
    ├── glm1_ols_summary.txt             
    ├── glm1_ols_hc3_summary.txt        
    ├── glm1_coefficients.csv            
    ├── glm1_vif.csv                    
    ├── glm1_assumption_checks.txt       
    ├── glm1_figure_coef_plot.png       
    │
    ├── glm2_logistic_summary.txt       
    ├── glm2_coefficients.csv            
    ├── glm2_vif.csv                    
    ├── glm2_wr_descriptives.csv         
    ├── glm2_figure_wr_by_condition.png  
    │
    ├── advanced_test1_interaction_plot.png  
    └── advanced_test2_rt_diagnostics.png    
```

---

## Prerequisites

**Install dependencies:**
```bash
pip install pandas numpy scipy matplotlib statsmodels scikit-learn
```

---

## Usage

All analysis is contained in a single Jupyter notebook: **`brsm-project.ipynb`**

Run cells sequentially from top to bottom. Each cell section builds on outputs from the previous one.

### Cell 1 — Data Preprocessing

Reads raw `.log` files from `NewLogsAnonymized/` and outputs `processed_memorability_data.csv` and `exclusion_log.txt`.

- Validates each block using the criterion: `Correct Validations > (Wrong / 2) + Missed`
- Parses stimulus codes to extract voice (Active/Passive) and noun type (HH/HL/LH/LL)
- Computes corrected memorability score: `Hits − False Alarms`
- Divides the experiment into 3 blocks (16 target sentences per block)
- Reports per-participant block counts and exclusions

### Cell 2 — Descriptive Analysis and Visualisation

Processes trial-level data from `NewLogsAnonymized/`, generates all figures, and saves aggregated CSV tables to `output/`.

- Filters practice trials and assigns blocks via "Rest Phase" timestamps
- Computes hit rates, false alarm rates, corrected scores, and reaction times
- Saves `clean_trials.csv` and `memorability_scores.csv` (required by all later cells)
- Generates Figures 1–3 (corrected score, reaction time, WR proportion)

### Cell 3 — Hypothesis Testing (Report 1 + Corrections)

Runs all primary hypothesis tests and applies Bonferroni correction. Outputs to `output/`.

| Hypothesis | Test | Description |
|---|---|---|
| H1 | Wilcoxon Signed-Rank | Active vs. Passive memorability (within-subjects) |
| H1a | Friedman Test | HH vs. HL vs. LH vs. LL memorability (repeated-measures) |
| H1b | Paired *t*-test | HL-Active vs. LH-Active (subject saliency) |
| H2 | Paired *t*-test | Active vs. Passive reaction time (within-subjects) |
| H3 | Paired *t*-test | WR accuracy: Active-repeat vs. Passive-repeat |
| Practice | Kruskal-Wallis | Memorability across Blocks 1, 2, and 3 |

### Cell 4 — OLS GLM and Logistic GLM (Report 2)

Fits two generalised linear models with full assumption checking.

**GLM 1 — OLS on Corrected Memorability Score** (`output/glm1_*`):
- Predictors: Voice (dummy), Condition dummies (HL/LH/LL vs HH), z-scored FA rate and median RT
- Heteroscedasticity detected (Breusch-Pagan *p* = .042) → HC3 robust standard errors applied
- *N* = 912 observations (114 participants × 8 condition cells)

**GLM 2 — Logistic GLM on WR Accuracy** (`output/glm2_*`):
- Outcome: binary WR correct/incorrect; binomial family, logit link
- Clustered standard errors on participant ID to correct for within-participant correlation
- Predictors: Voice, Subject Imageability, Object Imageability, Voice × Noun interactions
- *N* = 906 condition-level WR observations

### Cell 5 — RM-ANOVA and Linear Mixed-Effects Model (Report 2)

Fits two advanced models. Outputs figures to `output/`.

**RM-ANOVA — 2×2×2 Repeated-Measures ANOVA** (`output/advanced_test1_*`):
- Factors: Subject Noun imageability × Object Noun imageability × Voice (all within-subjects)
- Block practice effect absorbed via residualisation before analysis
- Sphericity trivially satisfied (all factors have 2 levels); no Mauchly correction needed
- Key finding: significant Subject Noun × Object Noun interaction (*p* = .013)

**LMM — Linear Mixed-Effects Model on RT** (`output/advanced_test2_*`):
- Outcome: log-transformed trial-level RT (*N* = 2,902 valid hits)
- Fixed effects: Voice, Subject Noun, Object Noun, Voice×Noun interactions
- Random intercept per participant (REML estimation)
- Key finding: Passive voice significantly increases RT by ~5% (β = +0.049, *p* = .010)

### Cell 6 — OLS Multiple Regression on RT

Supplementary regression bridging Report 1's paired *t*-test and the full LMM. Fits an OLS model on aggregated RT with Voice, Subject Imageability, Object Imageability, and Block as predictors (*N* = 2,448 observations). Produces assumption diagnostic plots (residuals vs fitted, Q-Q plot of log-RT residuals, Cook's Distance, interaction plot).

---

## Hypotheses Tested

| Hypothesis | Description |
|---|---|
| H1 | Sentence memorability is higher for Active than Passive voice |
| H1a | Memorability follows HH ≈ HL ≈ LH > LL regardless of voice |
| H1b | HL-Active sentences are more memorable than LH-Active (subject saliency) |
| H2 | Passive voice produces longer Initial Recognition reaction times |
| H3 | WR accuracy is lower for voice-transformed (Passive-repeat) than exact-repeat (Active-repeat) sentences |

---

## Data

### Input

Raw participant log files from the experiment software, containing fields:
`participant_ID`, `Timestamp`, `Event`, `Stimulus`, `isTarget`, `isRepeat`, `isValidation`, `Accuracy_IR`, `Reaction_time_IR`, `Button`

### Key Output Files

| File | Description |
|---|---|
| `processed_memorability_data.csv` | Block-level results (339 blocks, 114 participants) |
| `exclusion_log.txt` | Per-participant block exclusion record |
| `clean_trials.csv` | Trial-level data (3,536 rows) |
| `memorability_scores.csv` | Per-participant × condition aggregates |
| `hypothesis_results.csv` | Structured hypothesis test output with Bonferroni flags |
| `results_report.txt` | Human-readable statistical report |
| `glm1_coefficients.csv` | OLS GLM coefficients with HC3 SEs and 95% CIs |
| `glm2_coefficients.csv` | Logistic GLM coefficients with odds ratios and CIs |
| `figure1_corrected_score.png` | Corrected memorability score by condition × voice |
| `figure2_reaction_time.png` | Mean IR reaction time by condition × voice |
| `figure3_wr_proportion_yes.png` | WR proportion "Yes" by condition |
| `glm1_figure_coef_plot.png` | OLS GLM coefficient forest plot |
| `glm2_figure_wr_by_condition.png` | WR accuracy by condition and voice |
| `advanced_test1_interaction_plot.png` | RM-ANOVA Subject × Object Noun interaction |
| `advanced_test2_rt_diagnostics.png` | LMM RT distribution diagnostics |

---
