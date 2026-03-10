"""
STEP 2 — COMBINED HYPOTHESIS TESTING & DESCRIPTIVES
=====================================================
Uses BOTH datasets:

  teammate_data  = processed_memorability_data.csv  (her preprocessing)
  our_data       = memorability_scores.csv          (our preprocessing)

What her data adds that ours did not:
  1. True 3-block structure (she uses target-count to split blocks;
     we used 'Rest Phase started' markers which only found 2)
  2. Raw hit and false alarm COUNTS stored per block
  3. Explicit per-block balance verification (4 of each condition per block)
  4. Shapiro-Wilk normality check on scores → confirms non-parametric tests
  5. Block-order / practice-effect analysis (score trend across blocks 1→2→3)

What her data CANNOT do that ours can:
  - Per-condition×voice corrected memorability scores (needed for H1, H1a, H1b, H2)
  - RT analysis (needed for H2)
  - WR accuracy comparison (needed for H3)

Strategy:
  - Use HER data for: normality justification, block-effect analysis, exclusion reporting
  - Use OUR data for: all hypothesis tests requiring condition/voice breakdown
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

TEAMMATE_FILE = "./processed_memorability_data.csv"
OUR_MEM_FILE  = "./output/memorability_scores.csv"
OUR_TRIAL_FILE= "./output/clean_trials.csv"
OUTPUT_FOLDER = "./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def cohens_d_ind(a, b):
    pooled = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff,ddof=1) > 0 else np.nan

def fmt_p(p):
    if p < 0.001: return "< .001"
    return f"= {p:.3f}"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
tm   = pd.read_csv(TEAMMATE_FILE)       # her block-level data
mem  = pd.read_csv(OUR_MEM_FILE)        # our condition×voice scores
tri  = pd.read_csv(OUR_TRIAL_FILE)      # our trial-level data

print(f"Teammate data  : {len(tm)} blocks, {tm['participant_id'].nunique()} participants")
print(f"Our mem data   : {len(mem)} rows, {mem['participant_ID'].nunique()} participants")
print(f"Our trial data : {len(tri)} trials")
print()

results   = []   # t-test results for Bonferroni
report    = []   # human-readable lines

def sec(title):
    line = "-" * 55
    print(f"\n{line}\n{title}\n{line}")
    report.extend([line, title, line])

def out(line=""):
    print(line)
    report.append(line)

# ═════════════════════════════════════════════
# SECTION 0 — EXCLUSION SUMMARY (from her data)
# ═════════════════════════════════════════════
sec("EXCLUSION SUMMARY (from teammate preprocessing)")

n_files     = 114
total_blocks= n_files * 3
valid_blocks= len(tm)
excl_blocks = total_blocks - valid_blocks
n_complete  = (tm.groupby('participant_id')['block'].count() == 3).sum()
n_partial   = (tm.groupby('participant_id')['block'].count() < 3).sum()

out(f"  Log files processed          : {n_files}")
out(f"  Total possible blocks        : {total_blocks}")
out(f"  Valid blocks retained        : {valid_blocks} ({100*valid_blocks/total_blocks:.1f}%)")
out(f"  Blocks excluded              : {excl_blocks} ({100*excl_blocks/total_blocks:.1f}%)")
out(f"  Complete participants (3 blk): {n_complete}")
out(f"  Partial participants (<3 blk): {n_partial}")
out(f"  Final N for analysis         : {tm['participant_id'].nunique()}")

# ═════════════════════════════════════════════
# SECTION 1 — NORMALITY CHECK (from her data — justifies non-parametric)
# ═════════════════════════════════════════════
sec("NORMALITY CHECK — Shapiro-Wilk (from teammate data)")

sw_stat, sw_p = stats.shapiro(tm['corrected_memorability_score'].dropna())
out(f"  Shapiro-Wilk: W = {sw_stat:.4f}, p {fmt_p(sw_p)}")
out(f"  Distribution: {'NON-NORMAL' if sw_p < 0.05 else 'normal'}")
out(f"  Conclusion: {'Non-parametric tests (Kruskal-Wallis) are justified' if sw_p < 0.05 else 'Parametric tests appropriate'}")

# ═════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE STATISTICS (per condition×voice, from our data)
# ═════════════════════════════════════════════
sec("TASK 5 — DESCRIPTIVE STATISTICS (per condition x voice)")

desc = (
    mem.groupby(["condition","voice"])
    .agg(
        N             = ("participant_ID",      "count"),
        Mean_MemScore = ("corrected_mem_score", "mean"),
        SD_MemScore   = ("corrected_mem_score", "std"),
        SE_MemScore   = ("corrected_mem_score", lambda x: x.std()/np.sqrt(len(x))),
        Mean_HitRate  = ("hit_rate",            "mean"),
    ).reset_index().round(4)
)
rt_desc = (
    tri[tri["RT_IR_ms"].notna()]
    .groupby(["condition","voice"])
    .agg(Mean_RT=("RT_IR_ms","mean"), SD_RT=("RT_IR_ms","std"), Median_RT=("RT_IR_ms","median"))
    .reset_index().round(1)
)
desc = desc.merge(rt_desc, on=["condition","voice"], how="left")
desc["condition_voice"] = desc["condition"] + "_" + desc["voice"]

out()
out(desc[["condition_voice","N","Mean_MemScore","SD_MemScore","Mean_HitRate","Mean_RT","SD_RT"]].to_string(index=False))
desc.to_csv(os.path.join(OUTPUT_FOLDER,"descriptives_table.csv"), index=False)

# Block-level descriptives from her data
out()
out("  Block-level corrected scores (from teammate data, raw Hits-FA counts):")
for b in [1,2,3]:
    bdf = tm[tm['block']==b]['corrected_memorability_score']
    out(f"    Block {b}: M={bdf.mean():.2f}, SD={bdf.std():.2f}, "
        f"Median={bdf.median():.1f}, N={len(bdf)}")

# ═════════════════════════════════════════════
# SECTION 3 — BLOCK PRACTICE EFFECT (new — from her data only)
# Tests whether memorability score changes across blocks 1→2→3
# A significant result means participants got better/worse over time
# ═════════════════════════════════════════════
sec("BLOCK PRACTICE EFFECT — Kruskal-Wallis across Blocks 1/2/3")

b1 = tm[tm['block']==1]['corrected_memorability_score'].values
b2 = tm[tm['block']==2]['corrected_memorability_score'].values
b3 = tm[tm['block']==3]['corrected_memorability_score'].values

kw_b, p_b = stats.kruskal(b1, b2, b3)
out(f"\n  H(2) = {kw_b:.3f}, p {fmt_p(p_b)}")
out(f"  Block 1 M={np.mean(b1):.2f}  Block 2 M={np.mean(b2):.2f}  Block 3 M={np.mean(b3):.2f}")
out(f"  Interpretation: {'Significant practice/order effect across blocks' if p_b < 0.05 else 'No significant block-order effect'}")

# Post-hoc pairwise if significant
if p_b < 0.05:
    out("\n  Pairwise Mann-Whitney post-hoc (Bonferroni a=0.05/3=0.0167):")
    pairs = [("1 vs 2", b1, b2), ("1 vs 3", b1, b3), ("2 vs 3", b2, b3)]
    for label, x, y in pairs:
        u, pu = stats.mannwhitneyu(x, y, alternative='two-sided')
        sig = "significant" if pu < 0.0167 else "not significant"
        out(f"    Block {label}: U={u:.1f}, p {fmt_p(pu)} [{sig}]")

# ═════════════════════════════════════════════
# SECTION 4 — TASK 6a: H1 — Active vs Passive Memorability
# ═════════════════════════════════════════════
sec("TASK 6a — H1: Active vs Passive Memorability (independent t-test)")

voice_mem = mem.groupby(["participant_ID","voice"])["corrected_mem_score"].mean().reset_index()
act_m  = voice_mem[voice_mem["voice"]=="Active"]["corrected_mem_score"].values
pas_m  = voice_mem[voice_mem["voice"]=="Passive"]["corrected_mem_score"].values

t6a, p6a = stats.ttest_ind(act_m, pas_m)
d6a = cohens_d_ind(act_m, pas_m)

out(f"\n  Active  : M={np.mean(act_m):.4f}, SD={np.std(act_m,ddof=1):.4f}, N={len(act_m)}")
out(f"  Passive : M={np.mean(pas_m):.4f}, SD={np.std(pas_m,ddof=1):.4f}, N={len(pas_m)}")
out(f"  t({len(act_m)+len(pas_m)-2}) = {t6a:.3f}, p {fmt_p(p6a)}, d = {d6a:.3f}")

results.append({"test":"H1 - Memorability Active vs Passive","hypothesis":"H1",
    "M1":round(np.mean(act_m),4),"M2":round(np.mean(pas_m),4),
    "statistic":round(t6a,4),"df":len(act_m)+len(pas_m)-2,
    "p_value":round(p6a,4),"cohens_d":round(d6a,4)})

# ═════════════════════════════════════════════
# SECTION 5 — TASK 6b: H2 — Active vs Passive RT
# ═════════════════════════════════════════════
sec("TASK 6b — H2: Active vs Passive RT (independent t-test)")

voice_rt = mem.groupby(["participant_ID","voice"])["mean_RT_IR"].mean().reset_index()
act_rt  = voice_rt[voice_rt["voice"]=="Active"]["mean_RT_IR"].dropna().values
pas_rt  = voice_rt[voice_rt["voice"]=="Passive"]["mean_RT_IR"].dropna().values

t6b, p6b = stats.ttest_ind(act_rt, pas_rt)
d6b = cohens_d_ind(act_rt, pas_rt)

out(f"\n  Active  : M={np.mean(act_rt):.1f}ms, SD={np.std(act_rt,ddof=1):.1f}, N={len(act_rt)}")
out(f"  Passive : M={np.mean(pas_rt):.1f}ms, SD={np.std(pas_rt,ddof=1):.1f}, N={len(pas_rt)}")
out(f"  t({len(act_rt)+len(pas_rt)-2}) = {t6b:.3f}, p {fmt_p(p6b)}, d = {d6b:.3f}")

results.append({"test":"H2 - RT Active vs Passive","hypothesis":"H2",
    "M1":round(np.mean(act_rt),2),"M2":round(np.mean(pas_rt),2),
    "statistic":round(t6b,4),"df":len(act_rt)+len(pas_rt)-2,
    "p_value":round(p6b,4),"cohens_d":round(d6b,4)})

# ═════════════════════════════════════════════
# SECTION 6 — TASK 7: H1b — HL-Active vs LH-Active (paired t-test)
# ═════════════════════════════════════════════
sec("TASK 7 — H1b: HL-Active vs LH-Active (paired t-test)")

hl = mem[(mem["condition"]=="HL")&(mem["voice"]=="Active")][["participant_ID","corrected_mem_score"]].rename(columns={"corrected_mem_score":"HL"})
lh = mem[(mem["condition"]=="LH")&(mem["voice"]=="Active")][["participant_ID","corrected_mem_score"]].rename(columns={"corrected_mem_score":"LH"})
paired7 = hl.merge(lh, on="participant_ID")

t7, p7 = stats.ttest_rel(paired7["HL"], paired7["LH"])
d7 = cohens_d_paired(paired7["HL"].values, paired7["LH"].values)

out(f"\n  HL-Active : M={paired7['HL'].mean():.4f}, SD={paired7['HL'].std(ddof=1):.4f}")
out(f"  LH-Active : M={paired7['LH'].mean():.4f}, SD={paired7['LH'].std(ddof=1):.4f}")
out(f"  N (paired)= {len(paired7)}")
out(f"  t({len(paired7)-1}) = {t7:.3f}, p {fmt_p(p7)}, d = {d7:.3f}")

results.append({"test":"H1b - HL-Active vs LH-Active","hypothesis":"H1b",
    "M1":round(paired7["HL"].mean(),4),"M2":round(paired7["LH"].mean(),4),
    "statistic":round(t7,4),"df":len(paired7)-1,
    "p_value":round(p7,4),"cohens_d":round(d7,4)})

# ═════════════════════════════════════════════
# SECTION 7 — KRUSKAL-WALLIS: H1a — All 4 noun conditions
# ═════════════════════════════════════════════
sec("KRUSKAL-WALLIS — H1a: HH vs HL vs LH vs LL")

cond_m = mem.groupby(["participant_ID","condition"])["corrected_mem_score"].mean().reset_index()
groups = [cond_m[cond_m["condition"]==c]["corrected_mem_score"].values for c in ["HH","HL","LH","LL"]]
kw_c, p_c = stats.kruskal(*groups)

out(f"\n  H(3) = {kw_c:.3f}, p {fmt_p(p_c)}")
for c, g in zip(["HH","HL","LH","LL"], groups):
    out(f"    {c}: M={np.mean(g):.4f}, SD={np.std(g,ddof=1):.4f}, N={len(g)}")

if p_c < 0.05:
    out("\n  Pairwise Mann-Whitney post-hoc (Bonferroni a=0.05/6=0.0083):")
    conds = ["HH","HL","LH","LL"]
    for i in range(len(conds)):
        for j in range(i+1, len(conds)):
            u, pu = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            sig = "significant" if pu < 0.0083 else "not significant"
            out(f"    {conds[i]} vs {conds[j]}: U={u:.1f}, p {fmt_p(pu)} [{sig}]")

# ═════════════════════════════════════════════
# SECTION 8 — H3: WR accuracy Active vs Passive repeat
# ═════════════════════════════════════════════
sec("H3 — WR Accuracy: Active-repeat vs Passive-repeat (paired t-test)")

wr = tri[tri["accuracy_WR"].notna()].copy()
wr_v = wr.groupby(["participant_ID","voice"])["accuracy_WR"].mean().reset_index()
wr_a = wr_v[wr_v["voice"]=="Active"][["participant_ID","accuracy_WR"]].rename(columns={"accuracy_WR":"WR_Active"})
wr_p = wr_v[wr_v["voice"]=="Passive"][["participant_ID","accuracy_WR"]].rename(columns={"accuracy_WR":"WR_Passive"})
paired3 = wr_a.merge(wr_p, on="participant_ID").dropna()

t3, p3 = stats.ttest_rel(paired3["WR_Active"], paired3["WR_Passive"])
d3 = cohens_d_paired(paired3["WR_Active"].values, paired3["WR_Passive"].values)

out(f"\n  Active-repeat  WR: M={paired3['WR_Active'].mean():.4f}, SD={paired3['WR_Active'].std(ddof=1):.4f}")
out(f"  Passive-repeat WR: M={paired3['WR_Passive'].mean():.4f}, SD={paired3['WR_Passive'].std(ddof=1):.4f}")
out(f"  N (paired) = {len(paired3)}")
out(f"  t({len(paired3)-1}) = {t3:.3f}, p {fmt_p(p3)}, d = {d3:.3f}")

results.append({"test":"H3 - WR accuracy Active vs Passive","hypothesis":"H3",
    "M1":round(paired3["WR_Active"].mean(),4),"M2":round(paired3["WR_Passive"].mean(),4),
    "statistic":round(t3,4),"df":len(paired3)-1,
    "p_value":round(p3,4),"cohens_d":round(d3,4)})

# ═════════════════════════════════════════════
# SECTION 9 — BONFERRONI CORRECTION (t-tests only)
# ═════════════════════════════════════════════
sec("TASK 8 — BONFERRONI CORRECTION")

res_df = pd.DataFrame(results)
n_tests = len(res_df)
bonf_a  = round(0.05 / n_tests, 4)
res_df["bonferroni_alpha"]      = bonf_a
res_df["significant_corrected"] = res_df["p_value"] < bonf_a

out(f"\n  t-tests run      : {n_tests}")
out(f"  Bonferroni alpha : 0.05 / {n_tests} = {bonf_a}")
out()
out(f"  {'Test':<45} {'p-value':<10} {'Result'}")
out(f"  {'-'*45} {'-'*9} {'-'*20}")
for _, row in res_df.iterrows():
    label = "SIGNIFICANT" if row["significant_corrected"] else "not significant"
    out(f"  {row['test']:<45} {str(row['p_value']):<10} {label}")

# Also note KW block result
out()
out(f"  [Non-parametric, no Bonferroni applied]")
out(f"  KW blocks H1a    p {fmt_p(p_c)}  {'significant at p<.05' if p_c < 0.05 else 'not significant'}")
out(f"  KW practice-eff  p {fmt_p(p_b)}  {'significant at p<.05 -- IMPORTANT NOTE' if p_b < 0.05 else 'not significant'}")

# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
# Add KW results to full table
kw_rows = [
    {"test":"KW H1a - HH/HL/LH/LL","hypothesis":"H1a","M1":"see desc","M2":"see desc",
     "statistic":round(kw_c,4),"df":3,"p_value":round(p_c,4),"cohens_d":"N/A",
     "bonferroni_alpha":"N/A","significant_corrected": p_c < 0.05},
    {"test":"KW Block effect","hypothesis":"Practice","M1":"see desc","M2":"see desc",
     "statistic":round(kw_b,4),"df":2,"p_value":round(p_b,4),"cohens_d":"N/A",
     "bonferroni_alpha":"N/A","significant_corrected": p_b < 0.05},
]
all_results = pd.concat([res_df, pd.DataFrame(kw_rows)], ignore_index=True)
all_results.to_csv(os.path.join(OUTPUT_FOLDER,"hypothesis_results.csv"), index=False)

with open(os.path.join(OUTPUT_FOLDER,"results_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n" + "="*55)
print("  ANALYSIS COMPLETE")
print("="*55)
print(f"  -> {OUTPUT_FOLDER}/descriptives_table.csv")
print(f"  -> {OUTPUT_FOLDER}/hypothesis_results.csv")
print(f"  -> {OUTPUT_FOLDER}/results_report.txt")
print("="*55)