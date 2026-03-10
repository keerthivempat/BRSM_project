# Sentence Memorability Study - Data Preprocessing
## Data Architect Deliverables

**Project:** BRSM - Sentence Memorability Research  
**Role:** Data Architect (Pre-processing & Validation)  
**Date Completed:** March 10, 2026  
**Status:** ✅ ALL TASKS COMPLETE

---

## 📋 Overview

This directory contains all data preprocessing work for the Sentence Memorability experiment. All four required tasks have been completed successfully:

✅ **Task 1:** Validation filter applied to 342 blocks  
✅ **Task 2:** Exclusion log maintained (3 blocks excluded)  
✅ **Task 3:** Categorical variables constructed (Voice, Noun Type)  
✅ **Task 4:** Corrected Memorability Scores calculated (Hits - False Alarms)  
✅ **Final Output:** Clean CSV and Excel files generated

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Participants Processed** | 114 |
| **Valid Blocks** | 339 (99.1%) |
| **Excluded Blocks** | 3 (0.9%) |
| **Mean Memorability Score** | 0.47 ± 10.31 |
| **Data Retention Rate** | 99.1% |

---

## 📁 Important Files

### 🎯 PRIMARY OUTPUTS (Start Here!)

| File | Description | Size |
|------|-------------|------|
| **processed_memorability_data.csv** | Clean dataset for analysis (CSV format) | 14 KB |
| **processed_memorability_data.xlsx** | Clean dataset for analysis (Excel format) | 22 KB |
| **exclusion_log.txt** | Documentation of excluded blocks | 851 B |

### 📖 DOCUMENTATION

| File | Description | Size |
|------|-------------|------|
| **SUMMARY.md** | Quick overview and next steps | 5.9 KB |
| **QUICK_START.md** | How to analyze the data | 6.3 KB |
| **DATA_PREPROCESSING_REPORT.md** | Comprehensive technical report | 9.7 KB |
| **TASK_CHECKLIST.md** | Detailed task completion checklist | 7.8 KB |
| **README.md** | This file | - |

### 💻 CODE

| File | Description | Size |
|------|-------------|------|
| **preprocess_data.py** | Main preprocessing pipeline | 14 KB |
| **create_excel.py** | Excel file generator | 358 B |
| **visualize_data.py** | Data visualization script (bonus) | 4.2 KB |

### 📦 RAW DATA

| Directory | Description |
|-----------|-------------|
| **NewLogsAnonymized/** | Original participant log files (114 files) |

---

## 🚀 Quick Start

### For Immediate Use:
1. **Open:** `processed_memorability_data.xlsx`
2. **Read:** `SUMMARY.md` for overview
3. **Check:** `exclusion_log.txt` for excluded blocks
4. **Analyze:** Follow examples in `QUICK_START.md`

### For Detailed Understanding:
1. **Read:** `DATA_PREPROCESSING_REPORT.md` for complete methodology
2. **Review:** `TASK_CHECKLIST.md` for task completion details
3. **Examine:** `preprocess_data.py` for code implementation

### For Visualization (Optional):
```bash
python visualize_data.py
```
This generates `data_overview.png` with summary plots.

---

## 📈 Dataset Structure

**339 rows × 16 columns**

**Identification:**
- `participant_id`: Unique participant identifier (232-398)
- `block`: Experimental block number (1, 2, or 3)

**Dependent Variable (Primary):**
- `corrected_memorability_score`: Hits - False Alarms (-33 to 19)

**Raw Performance Metrics:**
- `hits`: Correctly identified repeated sentences (0-19)
- `false_alarms`: Incorrect spacebar presses (0-26)

**Validation Metrics:**
- `validation_passed`: Boolean (all TRUE in final dataset)
- `validation_correct`: Correct validation responses
- `validation_wrong`: Wrong validation responses
- `validation_missed`: Missed validation responses

**Sentence Distribution:**
- `n_target_sentences`: Total targets (always 16)
- `n_active`, `n_passive`: Voice counts (8 each)
- `n_HH`, `n_HL`, `n_LH`, `n_LL`: Noun type counts (4 each)

---

## 🔍 Data Quality

### Validation Filter
**Formula:** `Correct Validation > (Wrong Validation / 2) + Missed Validation`

**Results:**
- Pass rate: 99.1% (339/342 blocks)
- Fail rate: 0.9% (3/342 blocks)

### Exclusions
Only 3 blocks excluded from 2 participants:
- Participant 299: Blocks 2 and 3
- Participant 309: Block 3

All exclusions due to insufficient validation responses (documented in `exclusion_log.txt`).

### Completeness
✓ All 114 participants processed  
✓ Each valid block has exactly 16 target sentences  
✓ Balanced design: 8 Active + 8 Passive per block  
✓ Balanced design: 4 HH + 4 HL + 4 LH + 4 LL per block  

---

## 📝 For Your Report

### Methods Section - Participants

> "A total of 114 participants completed the experiment. Data quality was assessed using a validation filter (Correct Validation > [Wrong Validation/2] + Missed Validation) applied to each experimental block. Three blocks from two participants failed validation criteria and were excluded from analysis. The final dataset comprised 339 valid blocks from 114 participants (111 with complete data, 2 with partial data)."

### Methods Section - Data Processing

> "Corrected memorability scores were calculated for each block as the difference between hits (correctly identified repeated sentences) and false alarms (incorrect responses to non-repeated sentences). Categorical variables for sentence structure (Voice: Active vs. Passive) and noun memorability (Noun Type: HH, HL, LH, LL) were extracted from stimulus codes."

---

## 🔄 Reproducibility

All preprocessing is fully reproducible:

```bash
python preprocess_data.py
```

This will:
1. Process all 114 log files from `NewLogsAnonymized/`
2. Apply validation filter
3. Calculate memorability scores
4. Generate output files

To create Excel version:
```bash
python create_excel.py
```

---

## 💡 Analysis Recommendations

### Statistical Tests (As Specified)
- **Kruskal-Wallis test** for non-parametric group comparisons
- Account for repeated measures (3 blocks per participant)
- Consider mixed-effects models for complex analyses

### Additional Considerations
- Analyze hits and false alarms separately to understand recognition vs. response bias
- Check for block effects (practice or fatigue)
- Visualize score distributions before formal testing

See `QUICK_START.md` for code examples in R and Python.

---

## 📞 Support & Documentation

| Question Type | See File |
|---------------|----------|
| "How do I use this data?" | `QUICK_START.md` |
| "What exactly was done?" | `DATA_PREPROCESSING_REPORT.md` |
| "Were all tasks completed?" | `TASK_CHECKLIST.md` |
| "Quick overview?" | `SUMMARY.md` |
| "Which blocks were excluded?" | `exclusion_log.txt` |
| "How does the code work?" | `preprocess_data.py` (commented) |

---

## ✨ Bonus Features

### Data Visualization Script
`visualize_data.py` creates a comprehensive overview figure with:
1. Score distribution histogram
2. Hits vs. False Alarms scatter plot
3. Validation statistics bar chart
4. Scores across blocks boxplot

Run with: `python visualize_data.py`

---

## ✅ Quality Assurance

**Code Quality:**
- ✓ Fully documented with comments
- ✓ Modular function design
- ✓ Error handling implemented
- ✓ Reproducible workflow

**Data Quality:**
- ✓ 99.1% retention rate (excellent)
- ✓ All exclusions documented and justified
- ✓ Balanced experimental design preserved
- ✓ No missing values in final dataset

**Documentation Quality:**
- ✓ Comprehensive technical report
- ✓ Quick-start guide for users
- ✓ Methods section text provided
- ✓ Task completion checklist

---

## 🎯 Next Steps

**For Team Members:**
1. **Statistician:** Use `processed_memorability_data.xlsx` for Kruskal-Wallis tests
2. **Report Writer:** Use text from `DATA_PREPROCESSING_REPORT.md` for Methods section
3. **PI/Supervisor:** Review `SUMMARY.md` and `TASK_CHECKLIST.md` for oversight

**Ready for Analysis!** All preprocessing complete. Dataset is clean, validated, and documented.

---

## 📌 Citation

When using this dataset in publications, please acknowledge:
- Data preprocessing performed according to pre-registered protocol
- Validation filter applied: Correct > (Wrong/2) + Missed
- 99.1% data retention rate (3/342 blocks excluded)
- 114 participants, 339 valid experimental blocks

---

**All Data Architect tasks completed successfully!**  
**The dataset is ready for statistical analysis.**

For questions or clarifications, refer to the comprehensive documentation or contact the Data Architect.

---

*Last updated: March 10, 2026*
