"""
Data Pre-processing and Validation for Sentence Memorability Experiment

This script processes participant log files to:
1. Apply validation filter
2. Track exclusions
3. Construct categorical variables (Voice, Noun Type)
4. Calculate Corrected Memorability Scores
5. Generate clean output CSV
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np

# Constants
DATA_DIR = Path("NewLogsAnonymized")
OUTPUT_FILE = "processed_memorability_data.csv"
EXCLUSION_LOG_FILE = "exclusion_log.txt"

def extract_voice_and_noun_type(stimulus):
    """
    Extract Voice (Active/Passive) and Noun Type (HH, HL, LH, LL) from stimulus code.
    
    Examples:
    - HH_112_A -> Voice: Active, Noun Type: HH
    - HL_16_P -> Voice: Passive, Noun Type: HL
    - LVL_148_P -> Voice: Passive, Noun Type: LL (L-Verb-L)
    - HVL_121_P -> Voice: Passive, Noun Type: HL (H-Verb-L)
    """
    if pd.isna(stimulus) or stimulus == "N/A":
        return None, None
    
    parts = stimulus.split("_")
    if len(parts) < 3:
        return None, None
    
    # Extract noun type
    noun_code = parts[0]
    # Map complex codes to simple ones
    if noun_code == "HVH" or noun_code == "HH":
        noun_type = "HH"
    elif noun_code == "HVL" or noun_code == "HL":
        noun_type = "HL"
    elif noun_code == "LVH" or noun_code == "LH":
        noun_type = "LH"
    elif noun_code == "LVL" or noun_code == "LL":
        noun_type = "LL"
    else:
        noun_type = None
    
    # Extract voice
    voice_code = parts[2]
    if voice_code == "A":
        voice = "Active"
    elif voice_code == "P":
        voice = "Passive"
    else:
        voice = None
    
    return voice, noun_type


def validate_block(block_df):
    """
    Apply validation filter: Correct Validation > (Wrong Validation/2) + Missed Validation
    
    Returns:
    - is_valid (bool): Whether the block passes validation
    - validation_stats (dict): Statistics for reporting
    """
    # Filter for validation events
    validation_events = block_df[block_df['isValidation'] == True].copy()
    
    if len(validation_events) == 0:
        return False, {"correct": 0, "wrong": 0, "missed": 0, "reason": "No validation events"}
    
    # Count correct validation IRs (when Validation IR pressed with accuracy = 1)
    correct_validation = len(validation_events[
        (validation_events['Event'].str.contains('Validation IR pressed', na=False)) &
        (validation_events['Accuracy IR'] == 1)
    ])
    
    # Count wrong validation IRs (when Validation Wrong IR pressed)
    wrong_validation = len(validation_events[
        validation_events['Event'].str.contains('Validation Wrong IR pressed', na=False)
    ])
    
    # Count missed validation (when validation target shown but no IR recorded)
    # For each validation target, check if there's a corresponding IR
    validation_targets = validation_events[validation_events['Event'].str.contains('Sentence shown', na=False)]
    validation_irs = validation_events[validation_events['Event'].str.contains('Validation IR pressed', na=False)]
    
    # Count missed: targets without corresponding IR presses
    missed_validation = len(validation_targets) - (correct_validation + wrong_validation)
    missed_validation = max(0, missed_validation)  # Ensure non-negative
    
    # Apply validation formula
    threshold = (wrong_validation / 2) + missed_validation
    is_valid = correct_validation > threshold
    
    validation_stats = {
        "correct": correct_validation,
        "wrong": wrong_validation,
        "missed": missed_validation,
        "threshold": threshold,
        "passed": is_valid
    }
    
    return is_valid, validation_stats


def calculate_memorability_score(block_df):
    """
    Calculate Corrected Memorability Score: Hits - False Alarms
    
    Hits: Correctly identifying repeated sentences (target + isRepeat, IR pressed with accuracy=1)
    False Alarms: Incorrectly pressing IR on non-repeated sentences
    """
    # Exclude practice trials
    experimental_trials = block_df[~block_df['Event'].str.contains('Practice', na=False)].copy()
    
    # Hits: Target sentences that are repeats with correct IR response
    hits = len(experimental_trials[
        (experimental_trials['isTarget'] == True) &
        (experimental_trials['isRepeat'] == True) &
        (experimental_trials['Event'].str.contains('IR pressed', na=False)) &
        (experimental_trials['Accuracy IR'] == 1)
    ])
    
    # False Alarms: Non-target sentences where participant incorrectly pressed IR
    # OR first-time target sentences (not repeats) where IR was pressed
    false_alarms = len(experimental_trials[
        (
            ((experimental_trials['isTarget'] == False) | (experimental_trials['isRepeat'] != True)) &
            (experimental_trials['Event'].str.contains('IR pressed', na=False)) &
            (experimental_trials['Accuracy IR'] == 0)
        )
    ])
    
    # Corrected Memorability Score
    corrected_score = hits - false_alarms
    
    return corrected_score, hits, false_alarms


def identify_block(row_index, df, target_indices):
    """
    Identify which experimental block a row belongs to (1, 2, or 3).
    Blocks are defined by target sentence count: 16 targets per block.
    """
    # Count how many targets come before this row
    targets_before = sum(idx < row_index for idx in target_indices)
    
    # Assign to block based on target count (16 targets per block)
    if targets_before < 16:
        return 1
    elif targets_before < 32:
        return 2
    elif targets_before < 48:
        return 3
    else:
        return 0  # After all experimental blocks


def process_participant_file(file_path):
    """
    Process a single participant log file.
    
    Returns:
    - results: List of dictionaries with block-level results
    - exclusions: List of exclusion information
    """
    # Read the log file
    df = pd.read_csv(file_path)
    participant_id = df['participant_ID'].iloc[0]
    
    results = []
    exclusions = []
    
    # Filter to experimental trials (exclude practice)
    experimental_df = df[~df['Event'].str.contains('Practice', na=False)].copy()
    
    # Find all target sentence presentations (first time shown)
    target_indices = experimental_df[
        (experimental_df['isTarget'] == True) &
        (experimental_df['Event'].str.contains('Sentence shown', na=False)) &
        (experimental_df['isRepeat'] != True)
    ].index.tolist()
    
    # Assign blocks based on target count
    experimental_df['Block'] = experimental_df.index.map(
        lambda idx: identify_block(idx, experimental_df, target_indices)
    )
    
    # Process each experimental block (1, 2, 3)
    for block_num in [1, 2, 3]:
        block_df = experimental_df[experimental_df['Block'] == block_num].copy()
        
        if len(block_df) == 0:
            exclusions.append({
                'participant_id': participant_id,
                'block': block_num,
                'reason': 'Block not found in data'
            })
            continue
        
        # Validate block
        is_valid, validation_stats = validate_block(block_df)
        
        if not is_valid:
            exclusions.append({
                'participant_id': participant_id,
                'block': block_num,
                'reason': f"Failed validation: Correct={validation_stats['correct']}, "
                          f"Threshold={validation_stats['threshold']:.1f}",
                'validation_stats': validation_stats
            })
            continue
        
        # Calculate memorability score for valid block
        corrected_score, hits, false_alarms = calculate_memorability_score(block_df)
        
        # Extract target sentences with their properties
        target_sentences = block_df[
            (block_df['isTarget'] == True) & 
            (block_df['Event'].str.contains('Sentence shown', na=False)) &
            (block_df['isRepeat'] != True)  # Only first presentation
        ].copy()
        
        # Add voice and noun type
        target_sentences[['Voice', 'NounType']] = target_sentences['Stimulus'].apply(
            lambda x: pd.Series(extract_voice_and_noun_type(x))
        )
        
        # Create result record
        result = {
            'participant_id': participant_id,
            'block': block_num,
            'corrected_memorability_score': corrected_score,
            'hits': hits,
            'false_alarms': false_alarms,
            'validation_passed': True,
            'validation_correct': validation_stats['correct'],
            'validation_wrong': validation_stats['wrong'],
            'validation_missed': validation_stats['missed'],
            'n_target_sentences': len(target_sentences)
        }
        
        # Add voice and noun type distributions
        if len(target_sentences) > 0:
            voice_counts = target_sentences['Voice'].value_counts()
            noun_type_counts = target_sentences['NounType'].value_counts()
            
            result['n_active'] = voice_counts.get('Active', 0)
            result['n_passive'] = voice_counts.get('Passive', 0)
            result['n_HH'] = noun_type_counts.get('HH', 0)
            result['n_HL'] = noun_type_counts.get('HL', 0)
            result['n_LH'] = noun_type_counts.get('LH', 0)
            result['n_LL'] = noun_type_counts.get('LL', 0)
        else:
            result.update({
                'n_active': 0, 'n_passive': 0,
                'n_HH': 0, 'n_HL': 0, 'n_LH': 0, 'n_LL': 0
            })
        
        results.append(result)
    
    return results, exclusions


def main():
    """
    Main processing function.
    """
    print("=" * 70)
    print("Sentence Memorability Data Pre-processing")
    print("=" * 70)
    print()
    
    # Get all log files
    log_files = sorted(DATA_DIR.glob("*.log"))
    print(f"Found {len(log_files)} participant log files")
    print()
    
    all_results = []
    all_exclusions = []
    
    # Process each file
    for i, log_file in enumerate(log_files, 1):
        print(f"Processing {i}/{len(log_files)}: {log_file.name}...", end=" ")
        
        try:
            results, exclusions = process_participant_file(log_file)
            all_results.extend(results)
            all_exclusions.extend(exclusions)
            print(f"✓ ({len(results)} valid blocks, {len(exclusions)} excluded)")
        except Exception as e:
            print(f"✗ Error: {e}")
            all_exclusions.append({
                'participant_id': log_file.stem,
                'block': 'ALL',
                'reason': f'File processing error: {str(e)}'
            })
    
    print()
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    exclusions_df = pd.DataFrame(all_exclusions)
    
    # Statistics
    total_participants = len(log_files)
    total_possible_blocks = total_participants * 3
    total_valid_blocks = len(results_df)
    total_excluded_blocks = len(exclusions_df)
    
    print(f"Total participants processed: {total_participants}")
    print(f"Total possible blocks: {total_possible_blocks}")
    print(f"Valid blocks: {total_valid_blocks}")
    print(f"Excluded blocks: {total_excluded_blocks}")
    print(f"Exclusion rate: {(total_excluded_blocks/total_possible_blocks)*100:.1f}%")
    print()
    
    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Clean data saved to: {OUTPUT_FILE}")
    
    # Save exclusion log
    with open(EXCLUSION_LOG_FILE, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXCLUSION LOG - Sentence Memorability Study\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total participants: {total_participants}\n")
        f.write(f"Total possible blocks: {total_possible_blocks}\n")
        f.write(f"Valid blocks: {total_valid_blocks}\n")
        f.write(f"Excluded blocks: {total_excluded_blocks}\n")
        f.write(f"Exclusion rate: {(total_excluded_blocks/total_possible_blocks)*100:.1f}%\n\n")
        f.write("=" * 70 + "\n")
        f.write("Detailed Exclusions:\n")
        f.write("=" * 70 + "\n\n")
        
        for exc in all_exclusions:
            f.write(f"Participant: {exc['participant_id']}, Block: {exc['block']}\n")
            f.write(f"Reason: {exc['reason']}\n")
            if 'validation_stats' in exc:
                stats = exc['validation_stats']
                f.write(f"  Correct: {stats['correct']}, Wrong: {stats['wrong']}, "
                       f"Missed: {stats['missed']}, Threshold: {stats['threshold']:.1f}\n")
            f.write("\n")
    
    print(f"✓ Exclusion log saved to: {EXCLUSION_LOG_FILE}")
    print()
    
    # Display sample of results
    print("=" * 70)
    print("Sample of Processed Data (first 5 rows):")
    print("=" * 70)
    if len(results_df) > 0:
        print(results_df.head())
        print()
        
        # Summary statistics
        print("=" * 70)
        print("Corrected Memorability Score Statistics:")
        print("=" * 70)
        print(results_df['corrected_memorability_score'].describe())
        print()
    else:
        print("No valid blocks found!")
        print()
    
    print("Processing complete! ✓")


if __name__ == "__main__":
    main()
