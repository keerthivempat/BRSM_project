# BRSM Project: Sentence Memorability Study

A research data analysis pipeline for a psycholinguistic study examining how sentence structure (active vs. passive voice) and noun frequency affect human memory performance.This project analyzes data from a sentence memorability recognition task in which participants were exposed to sentences and later tested on whether they remembered them. The study investigates the effects of:

- **Sentence voice**: Active vs. Passive
- **Noun frequency combinations**: HH, HL, LH, LL (H = High frequency, L = Low frequency)
- **Practice effects**: Changes in memorability across experimental blocks

## Participant Summary

- **Total participants processed**: 114  
- **Valid blocks retained**: 339 / 342 (99.1%)  
- **Complete participants (all 3 blocks)**: 112  
- **Exclusion rate**: 0.9%

## Project Structure

```
BRSM_project/
├── preprocess_data.py                    # Data preprocessing pipeline
├── descriptive_analysis.py              # Descriptive analysis and visualization
├── methods.py                           # Statistical hypothesis testing
├── processed_memorability_data.csv      # Preprocessed block-level data
├── descriptive_analysis_outputs/        # Generated figures and tables
│   ├── figure1_corrected_score.png      # Corrected memorability scores by condition
│   ├── figure2_reaction_time.png        # Mean reaction times by condition
│   ├── figure3_wr_proportion_yes.png    # Word recognition proportions by condition
│   ├── clean_trials.csv                 # Trial-level cleaned data
│   ├── memorability_scores.csv          # Per-participant per-condition aggregates
│   ├── per_condition_per_participant.csv
│   ├── per_participant_overall.csv
│   ├── table2_overall_stats.csv
│   ├── table3_per_condition.csv
│   └── table4_wr_breakdown.csv
└── output/
    ├── results_report.txt               # Human-readable statistical results
    ├── hypothesis_results.csv           # Hypothesis test results
    └── descriptives_table.csv          # Descriptive statistics table
```


## Pre-requisites to run the codebase

**Install dependencies:**
   ```bash
   pip install pandas numpy scipy matplotlib
   ```

## Usage

The pipeline runs in three sequential steps:

### Step 1: Preprocess Raw Data

```bash
python preprocess_data.py
```

Reads raw participant log files from the `NewLogsAnonymized/` folder and outputs `processed_memorability_data.csv`.

- Validates participant blocks using the criterion: `Correct > (Wrong / 2) + Missed`
- Parses stimulus codes to extract voice (Active/Passive) and noun type (HH/HL/LH/LL)
- Calculates corrected memorability scores: `Hits − False Alarms`
- Divides the experiment into 3 blocks (16 target sentences per block)

### Step 2: Descriptive Analysis and Visualization

```bash
python descriptive_analysis.py --logs_dir ./logs --out_dir ./descriptive_analysis_outputs
```

Processes raw log files in detail, generates figures, and produces CSV tables in the `descriptive_analysis_outputs/` folder.

- Parses participant event logs and filters out practice trials
- Assigns blocks based on "Rest Phase" timestamps
- Computes hit rates, false alarm rates, corrected scores, and reaction times
- Generates publication-ready bar charts (Figures 1–3)

### Step 3: Statistical Analysis

```bash
python methods.py
```

Runs hypothesis tests and writes results to the `output/` folder.

**Hypotheses tested:**

| Hypothesis | Test             | Description                              |
|------------|------------------|------------------------------------------|
| H1         | Independent t-test | Active vs. Passive memorability        |
| H1a        | Kruskal-Wallis   | HH vs. HL vs. LH vs. LL memorability   |
| H1b        | Paired t-test    | HL-Active vs. LH-Active                 |
| H2         | Independent t-test | Active vs. Passive reaction time       |
| H3         | Paired t-test    | WR accuracy: Active vs. Passive repeat  |
| Practice   | Kruskal-Wallis   | Memorability across blocks 1, 2, and 3  |

**Statistical approach:**
- Normality checked with Shapiro-Wilk test
- Non-parametric tests (Kruskal-Wallis) used where data is non-normal
- Bonferroni correction applied for multiple comparisons (α = 0.05 / 4 = 0.0125)
- Cohen's d calculated for effect sizes

## Data

### Input

Raw participant log files from the experiment software, containing fields such as:
`participant_ID`, `Timestamp`, `Event`, `Stimulus`, `isTarget`, `isRepeat`, `isValidation`, `Accuracy_IR`, `Reaction_time_IR`, `Button`

### Key Output Files

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `processed_memorability_data.csv` | Block-level results (339 blocks, 114 participants) |
| `clean_trials.csv`            | Trial-level data (3,537 rows)                    |
| `memorability_scores.csv`     | Per-participant per-condition aggregates          |
| `results_report.txt`          | Human-readable statistical report                |
| `hypothesis_results.csv`      | Structured hypothesis test output                |
| `figure1_corrected_score.png` | Corrected memorability score by condition        |
| `figure2_reaction_time.png`   | Mean IR reaction time by condition               |
| `figure3_wr_proportion_yes.png` | WR proportion "Yes" by condition               |


## Pipeline Diagram

```
Raw Log Files (Experiment Software)
           │
           ▼
  preprocess_data.py
           │
           ▼
processed_memorability_data.csv
           │
           ▼
  descriptive_analysis.py
           │
           ▼
clean_trials.csv + memorability_scores.csv + Figures (PNG)
           │
           ▼
      methods.py
           │
           ▼
results_report.txt + hypothesis_results.csv + descriptives_table.csv
```

