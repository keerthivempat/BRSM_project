"""
Sentence Memorability — Results Analysis
=========================================
Reproduces exactly the tables, figures, and statistics in Report 1 Section 3:
  - Table 2  : Overall recognition performance summary
  - Table 3  : Per-condition performance (corrected score, RT)
  - Table 4  : WR verbatim recognition breakdown
  - Figure 1 : Corrected memorability score by condition × voice
  - Figure 2 : Mean IR reaction time by condition × voice
  - Figure 3 : WR proportion "Yes" by condition × voice
  - All inferential stats (t-tests, ANOVA, binomial, chi-squared)

KEY FIX vs previous version
----------------------------
"Rest Phase started" rows have Stimulus=N/A and were being filtered out before
block assignment — so all rows ended up in block 1 and per-block validation
never fired correctly.  Fix: parse the RAW file (all rows including N/A) first
to get Rest Phase timestamps, then use those timestamps to assign block IDs to
the filtered main rows.

Usage
-----
    python results_analysis.py
    python results_analysis.py --logs_dir ./logs --out_dir ./output

Requirements
------------
    pip install pandas matplotlib scipy numpy
"""

import os, csv, io, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", default="./logs")
parser.add_argument("--out_dir",  default="./descriptive_analysis_outputs")
args = parser.parse_args()

LOGS_DIR = args.logs_dir
OUT_DIR  = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_TO_COND = {"HH": "HH", "HVL": "HL", "LVH": "LH", "LVL": "LL"}
COND_ORDER  = ["HH", "HL", "LH", "LL"]
COND_LABELS = ["HH\n(High–High)", "HL\n(High–Low)", "LH\n(Low–High)", "LL\n(Low–Low)"]
VOICE_ORDER = ["Active", "Passive"]
COLORS      = {"Active": "#3B74C2", "Passive": "#E07B2A"}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PARSE LOG FILE
#     Returns:
#       all_rows  – every non-Practice row (including Rest Phase / Stimulus=N/A)
#       main_rows – rows that carry a real stimulus value
# ═════════════════════════════════════════════════════════════════════════════
def parse_log(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read().replace("\r\n", "\n")
    reader   = csv.DictReader(io.StringIO(content))
    all_rows  = [r for r in reader if "Practice" not in r["Event"]]
    main_rows = [r for r in all_rows  if r["Stimulus"] != "N/A"]
    return all_rows, main_rows


# ═════════════════════════════════════════════════════════════════════════════
# 2.  ASSIGN BLOCK IDs VIA TIMESTAMPS
#     "Rest Phase started" rows have Stimulus=N/A so they only exist in
#     all_rows.  We extract their timestamps and use them as block boundaries
#     when tagging each main_row.
#
#     With 2 rest phases:
#       block 1 : ts  <  rest[0]
#       block 2 : rest[0] <= ts  <  rest[1]
#       block 3 : ts >= rest[1]
# ═════════════════════════════════════════════════════════════════════════════
def assign_blocks(all_rows, main_rows):
    rest_ts = sorted(
        int(r["Timestamp"])
        for r in all_rows
        if "Rest Phase" in r["Event"]
    )

    def get_block(ts):
        ts  = int(ts)
        bid = 1
        for boundary in rest_ts:
            if ts > boundary:
                bid += 1
        return bid

    for r in main_rows:
        r["_block"] = get_block(r["Timestamp"])


# ═════════════════════════════════════════════════════════════════════════════
# 3.  BLOCK VALIDATION
#     Criterion: Correct > (Wrong / 2) + Missed
# ═════════════════════════════════════════════════════════════════════════════
def block_passes(block_rows):
    correct = sum(
        1 for r in block_rows
        if r["isValidation"] == "true"
        and r["Event"]       == "Validation IR pressed"
        and r["Accuracy IR"] == "1"
    )
    wrong = sum(
        1 for r in block_rows
        if r["isValidation"] == "true"
        and r["Event"]       == "Validation Wrong IR pressed"
    )
    val_shown = {
        r["Stimulus"] for r in block_rows
        if r["isValidation"] == "true"
        and r["isRepeat"]    == "true"
        and r["Event"]       == "Sentence shown"
    }
    val_responded = {
        r["Stimulus"] for r in block_rows
        if r["isValidation"] == "true"
        and r["isRepeat"]    == "true"
        and r["Event"] in ("Validation IR pressed",
                           "Validation Wrong IR pressed")
    }
    missed = len(val_shown - val_responded)
    return correct > (wrong / 2) + missed


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PROCESS ONE PARTICIPANT
# ═════════════════════════════════════════════════════════════════════════════
def process_participant(all_rows, main_rows, pid):
    assign_blocks(all_rows, main_rows)

    all_blocks   = sorted({r["_block"] for r in main_rows})
    valid_blocks = {
        b for b in all_blocks
        if block_passes([r for r in main_rows if r["_block"] == b])
    }

    if not valid_blocks:
        return None, None

    vrows = [r for r in main_rows if r["_block"] in valid_blocks]

    # ── false alarm rate ──────────────────────────────────────────────────────
    filler_shown   = [r for r in vrows
                      if r["isTarget"]      != "true"
                      and r["isRepeat"]     != "true"
                      and r["isValidation"] != "true"
                      and r["Event"]        == "Sentence shown"]
    filler_pressed = [r for r in vrows
                      if r["isTarget"]      != "true"
                      and r["isRepeat"]     != "true"
                      and r["isValidation"] != "true"
                      and r["Event"]        == "IR pressed"]
    n_filler = len(filler_shown)
    n_fa     = len(filler_pressed)
    fa_rate  = n_fa / n_filler if n_filler else 0.0

    # ── per-condition cells ───────────────────────────────────────────────────
    cell_records = []
    for raw_cond, cond in RAW_TO_COND.items():
        for voice_code, voice_label in [("A", "Active"), ("P", "Passive")]:

            targets = [r for r in vrows
                       if r["isTarget"]      == "true"
                       and r["isRepeat"]     == "true"
                       and r["isValidation"] != "true"
                       and r["Event"]        == "Sentence shown"
                       and r["Stimulus"].startswith(raw_cond + "_")
                       and r["Stimulus"].endswith("_" + voice_code)]
            n_total = len(targets)
            if n_total == 0:
                continue

            hits = [r for r in vrows
                    if r["isTarget"]      == "true"
                    and r["isRepeat"]     == "true"
                    and r["isValidation"] != "true"
                    and r["Event"]        == "IR pressed"
                    and r["Accuracy IR"]  == "1"
                    and r["Stimulus"].startswith(raw_cond + "_")
                    and r["Stimulus"].endswith("_" + voice_code)]
            n_hits   = len(hits)
            hit_rate = n_hits / n_total
            corr     = hit_rate - fa_rate

            rts = [int(r["Reaction_time_IR"]) for r in hits
                   if r["Reaction_time_IR"] not in ("N/A", "")
                   and 100 <= int(r["Reaction_time_IR"]) <= 8000]
            mean_rt = float(np.mean(rts)) if rts else np.nan

            wr_yes = sum(1 for r in vrows
                         if r["isRepeat"]  == "true"
                         and r["isTarget"] == "true"
                         and r["Event"]    == "WR pressed"
                         and r["Button"]   == "Yes"
                         and r["Stimulus"].startswith(raw_cond + "_")
                         and r["Stimulus"].endswith("_" + voice_code))
            wr_no  = sum(1 for r in vrows
                         if r["isRepeat"]  == "true"
                         and r["isTarget"] == "true"
                         and r["Event"]    == "WR pressed"
                         and r["Button"]   == "No"
                         and r["Stimulus"].startswith(raw_cond + "_")
                         and r["Stimulus"].endswith("_" + voice_code))

            cell_records.append({
                "participant_ID": pid,
                "condition":      cond,
                "encoding_voice": voice_label,
                "n_total":        n_total,
                "n_hits":         n_hits,
                "hit_rate":       hit_rate,
                "fa_rate":        fa_rate,
                "corr_score":     corr,
                "mean_rt":        mean_rt,
                "wr_yes":         wr_yes,
                "wr_no":          wr_no,
            })

    # ── overall stats ─────────────────────────────────────────────────────────
    all_target_shown = [r for r in vrows
                        if r["isTarget"]      == "true"
                        and r["isRepeat"]     == "true"
                        and r["isValidation"] != "true"
                        and r["Event"]        == "Sentence shown"]
    all_hits = [r for r in vrows
                if r["isTarget"]      == "true"
                and r["isRepeat"]     == "true"
                and r["isValidation"] != "true"
                and r["Event"]        == "IR pressed"
                and r["Accuracy IR"]  == "1"]
    hit_rts  = [int(r["Reaction_time_IR"]) for r in all_hits
                if r["Reaction_time_IR"] not in ("N/A", "")
                and 100 <= int(r["Reaction_time_IR"]) <= 8000]

    n_targets  = len(all_target_shown)
    n_hits_tot = len(all_hits)
    overall_hr = n_hits_tot / n_targets if n_targets else 0.0

    overall = {
        "participant_ID":  pid,
        "n_targets":       n_targets,
        "n_hits":          n_hits_tot,
        "n_missed":        n_targets - n_hits_tot,
        "hit_rate":        overall_hr,
        "n_filler":        n_filler,
        "n_fa":            n_fa,
        "fa_rate":         fa_rate,
        "corr_score":      overall_hr - fa_rate,
        "mean_rt":         float(np.mean(hit_rts))   if hit_rts else np.nan,
        "median_rt":       float(np.median(hit_rts)) if hit_rts else np.nan,
        "sd_rt":           float(np.std(hit_rts))    if hit_rts else np.nan,
        "min_rt":          min(hit_rts) if hit_rts else np.nan,
        "max_rt":          max(hit_rts) if hit_rts else np.nan,
        "n_valid_blocks":  len(valid_blocks),
    }
    return cell_records, overall


# ═════════════════════════════════════════════════════════════════════════════
# 5.  LOAD ALL LOG FILES
# ═════════════════════════════════════════════════════════════════════════════
print(f"Loading logs from : {LOGS_DIR}")
log_files = sorted(f for f in os.listdir(LOGS_DIR)
                   if f.endswith(".log") or f.endswith(".csv"))
print(f"  {len(log_files)} files found\n")

all_cells   = []
all_overall = []
excluded    = []

for fname in log_files:
    try:
        all_rows, main_rows = parse_log(os.path.join(LOGS_DIR, fname))
        if not main_rows:
            continue
        pid = main_rows[0]["participant_ID"]
        cells, overall = process_participant(all_rows, main_rows, pid)
        if cells is None:
            excluded.append(pid)
        else:
            all_cells.extend(cells)
            all_overall.append(overall)
    except Exception as e:
        print(f"  ERROR {fname}: {e}")

df      = pd.DataFrame(all_cells)
df_over = pd.DataFrame(all_overall)
N       = df["participant_ID"].nunique()

print(f"Participants included : {N}")
print(f"Participants excluded : {len(excluded)}"
      + (f"  {excluded}" if excluded else ""))
print(f"Condition-level rows  : {len(df)}\n")

df.to_csv(     os.path.join(OUT_DIR, "per_condition_per_participant.csv"), index=False)
df_over.to_csv(os.path.join(OUT_DIR, "per_participant_overall.csv"),       index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  AGGREGATE  (mean ± SEM per condition × voice cell)
# ═════════════════════════════════════════════════════════════════════════════
agg = (df.groupby(["condition", "encoding_voice"])
         .agg(
             mean_corr = ("corr_score", "mean"),
             sem_corr  = ("corr_score", lambda x: x.sem()),
             mean_rt   = ("mean_rt",    "mean"),
             sem_rt    = ("mean_rt",    lambda x: x.sem()),
             n         = ("participant_ID", "nunique"),
         )
         .reset_index())


# ═════════════════════════════════════════════════════════════════════════════
# 7.  TABLE 2  —  Overall recognition performance
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("TABLE 2 — Overall Recognition Performance")
print("=" * 65)

table2 = pd.DataFrame({
    "Metric": [
        "Participants included (N)",
        "Total recognition trials",
        "Mean correct hits per participant",
        "Mean hit rate",
        "Mean missed targets per participant",
        "Mean false alarms per participant",
        "Mean FA rate",
        "Mean corrected memorability score",
        "Mean IR reaction time — hits",
        "Mean median RT — hits",
        "Mean valid blocks per participant",
    ],
    "Value": [
        str(N),
        str(int(df_over["n_targets"].sum())),
        f"{df_over['n_hits'].mean():.1f}  (total = {int(df_over['n_hits'].sum())})",
        f"{df_over['hit_rate'].mean():.3f}",
        f"{df_over['n_missed'].mean():.2f}  (total = {int(df_over['n_missed'].sum())})",
        f"{df_over['n_fa'].mean():.1f}  (total = {int(df_over['n_fa'].sum())})",
        f"{df_over['fa_rate'].mean():.3f}",
        f"{df_over['corr_score'].mean():.3f}",
        f"{df_over['mean_rt'].mean():.1f} ms  (SD = {df_over['sd_rt'].mean():.1f})",
        f"{df_over['median_rt'].mean():.1f} ms",
        f"{df_over['n_valid_blocks'].mean():.2f}",
    ]
})
print(table2.to_string(index=False))
table2.to_csv(os.path.join(OUT_DIR, "table2_overall_stats.csv"), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  TABLE 3  —  Per-condition performance
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TABLE 3 — Per-Condition Performance (mean across participants)")
print("=" * 65)

t3_rows = []
for cond in COND_ORDER:
    for voice in VOICE_ORDER:
        raw = df[(df["condition"] == cond) & (df["encoding_voice"] == voice)]
        if raw.empty:
            continue
        t3_rows.append({
            "Condition":             cond,
            "Voice":                 voice,
            "Mean Hits / Total":     f"{raw['n_hits'].mean():.1f} / {raw['n_total'].mean():.1f}",
            "Hit Rate (M)":          f"{raw['hit_rate'].mean():.3f}",
            "FA Rate (M)":           f"{raw['fa_rate'].mean():.3f}",
            "Corr. Score (M ± SEM)": f"{raw['corr_score'].mean():.3f} ± {raw['corr_score'].sem():.3f}",
            "Mean RT ms (M ± SEM)":  f"{raw['mean_rt'].mean():.1f} ± {raw['mean_rt'].sem():.1f}",
        })

table3 = pd.DataFrame(t3_rows)
print(table3.to_string(index=False))
table3.to_csv(os.path.join(OUT_DIR, "table3_per_condition.csv"), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  TABLE 4  —  WR verbatim recognition
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TABLE 4 — WR Verbatim Recognition (summed across all participants)")
print("=" * 65)

t4_rows = []
grand = {"ay": 0, "an": 0, "py": 0, "pn": 0}

for cond in COND_ORDER:
    act = df[(df["condition"] == cond) & (df["encoding_voice"] == "Active")]
    pas = df[(df["condition"] == cond) & (df["encoding_voice"] == "Passive")]
    ay, an = int(act["wr_yes"].sum()), int(act["wr_no"].sum())
    py, pn = int(pas["wr_yes"].sum()), int(pas["wr_no"].sum())
    at, pt = ay + an, py + pn
    grand["ay"] += ay; grand["an"] += an
    grand["py"] += py; grand["pn"] += pn
    t4_rows.append({
        "Condition":     cond,
        "Active – Yes":  f"{ay}  ({100*ay/at:.0f}%)" if at else "—",
        "Active – No":   f"{an}  ({100*an/at:.0f}%)" if at else "—",
        "Passive – Yes": f"{py}  ({100*py/pt:.0f}%)" if pt else "—",
        "Passive – No":  f"{pn}  ({100*pn/pt:.0f}%)" if pt else "—",
    })

at = grand["ay"] + grand["an"]
pt = grand["py"] + grand["pn"]
t4_rows.append({
    "Condition":     "Total",
    "Active – Yes":  f"{grand['ay']}  ({100*grand['ay']/at:.0f}%)",
    "Active – No":   f"{grand['an']}  ({100*grand['an']/at:.0f}%)",
    "Passive – Yes": f"{grand['py']}  ({100*grand['py']/pt:.0f}%)",
    "Passive – No":  f"{grand['pn']}  ({100*grand['pn']/pt:.0f}%)",
})

table4 = pd.DataFrame(t4_rows)
print(table4.to_string(index=False))
table4.to_csv(os.path.join(OUT_DIR, "table4_wr_breakdown.csv"), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 10.  INFERENTIAL STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("INFERENTIAL STATISTICS")
print("=" * 65)

# [i] Binomial: hit rate vs 0.5
tot_hits = int(df_over["n_hits"].sum())
tot_targ = int(df_over["n_targets"].sum())
b_hit    = stats.binomtest(tot_hits, tot_targ, 0.5, alternative="greater")
print(f"\n[i]   Binomial — Hit rate vs chance (0.50)")
print(f"      Hits = {tot_hits} / {tot_targ}   HR = {tot_hits/tot_targ:.3f}")
print(f"      p = {b_hit.pvalue:.2e}")

# [ii] Binomial: FA rate vs 0.5
tot_fa   = int(df_over["n_fa"].sum())
tot_fill = int(df_over["n_filler"].sum())
b_fa     = stats.binomtest(tot_fa, tot_fill, 0.5, alternative="less")
print(f"\n[ii]  Binomial — FA rate vs chance (0.50)")
print(f"      FAs = {tot_fa} / {tot_fill}   FAR = {tot_fa/tot_fill:.3f}")
print(f"      p = {b_fa.pvalue:.2e}")

# [iii] Independent t-test: Active vs Passive corrected score
act_cs = df[df["encoding_voice"] == "Active"]["corr_score"].dropna()
pas_cs = df[df["encoding_voice"] == "Passive"]["corr_score"].dropna()
t3v, p3v = stats.ttest_ind(act_cs, pas_cs)
print(f"\n[iii] Independent t-test — Corrected Score: Active vs Passive")
print(f"      Active  M={act_cs.mean():.4f}  SD={act_cs.std():.4f}  n={len(act_cs)}")
print(f"      Passive M={pas_cs.mean():.4f}  SD={pas_cs.std():.4f}  n={len(pas_cs)}")
print(f"      t({len(act_cs)+len(pas_cs)-2}) = {t3v:.3f},  p = {p3v:.4f}")

# [iv] Independent t-test: Active vs Passive RT
act_rt = df[df["encoding_voice"] == "Active"]["mean_rt"].dropna()
pas_rt = df[df["encoding_voice"] == "Passive"]["mean_rt"].dropna()
t4v, p4v = stats.ttest_ind(act_rt, pas_rt)
print(f"\n[iv]  Independent t-test — RT: Active vs Passive")
print(f"      Active  M={act_rt.mean():.1f} ms  SD={act_rt.std():.1f}  n={len(act_rt)}")
print(f"      Passive M={pas_rt.mean():.1f} ms  SD={pas_rt.std():.1f}  n={len(pas_rt)}")
print(f"      t({len(act_rt)+len(pas_rt)-2}) = {t4v:.3f},  p = {p4v:.4f}")

# [v] One-way ANOVA: corrected score across 4 conditions
grps_cs = [df[df["condition"] == c]["corr_score"].dropna().values for c in COND_ORDER]
fv, pv  = stats.f_oneway(*grps_cs)
df_b    = len(COND_ORDER) - 1
df_w    = sum(len(g) for g in grps_cs) - len(COND_ORDER)
print(f"\n[v]   One-Way ANOVA — Corrected Score across 4 conditions")
for c, g in zip(COND_ORDER, grps_cs):
    print(f"      {c}: M={g.mean():.4f}  SD={g.std():.4f}  n={len(g)}")
print(f"      F({df_b}, {df_w}) = {fv:.3f},  p = {pv:.4f}")

# [vi] One-way ANOVA: RT across 4 conditions
grps_rt  = [df[df["condition"] == c]["mean_rt"].dropna().values for c in COND_ORDER]
frv, prv = stats.f_oneway(*grps_rt)
df_br    = len(COND_ORDER) - 1
df_wr    = sum(len(g) for g in grps_rt) - len(COND_ORDER)
print(f"\n[vi]  One-Way ANOVA — RT across 4 conditions")
for c, g in zip(COND_ORDER, grps_rt):
    print(f"      {c}: M={g.mean():.1f} ms  SD={g.std():.1f}  n={len(g)}")
print(f"      F({df_br}, {df_wr}) = {frv:.3f},  p = {prv:.4f}")

# [vii] Chi-squared: WR Yes/No × voice
aw  = df[df["encoding_voice"] == "Active"][["wr_yes",  "wr_no"]].sum()
pw  = df[df["encoding_voice"] == "Passive"][["wr_yes", "wr_no"]].sum()
ct  = np.array([[aw["wr_yes"], aw["wr_no"]], [pw["wr_yes"], pw["wr_no"]]])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct, correction=False)
print(f"\n[vii] Chi-squared — WR Yes/No × encoding voice")
print(f"      Active  Yes={int(aw['wr_yes'])}  No={int(aw['wr_no'])}"
      f"  prop_yes={aw['wr_yes']/(aw['wr_yes']+aw['wr_no']):.2f}")
print(f"      Passive Yes={int(pw['wr_yes'])}  No={int(pw['wr_no'])}"
      f"  prop_yes={pw['wr_yes']/(pw['wr_yes']+pw['wr_no']):.2f}")
print(f"      chi2({dof}) = {chi2:.3f},  p = {p_chi:.4f}")

# [viii] Paired t-test: HH vs LL
hh_p = df[df["condition"] == "HH"].groupby("participant_ID")["corr_score"].mean()
ll_p = df[df["condition"] == "LL"].groupby("participant_ID")["corr_score"].mean()
prd  = pd.DataFrame({"HH": hh_p, "LL": ll_p}).dropna()
if len(prd) > 1:
    t8, p8 = stats.ttest_rel(prd["HH"], prd["LL"])
    print(f"\n[viii] Paired t-test — HH vs LL corrected score")
    print(f"      HH M={prd['HH'].mean():.4f}  LL M={prd['LL'].mean():.4f}  n={len(prd)}")
    print(f"      t({len(prd)-1}) = {t8:.3f},  p = {p8:.4f}")

# [ix] Paired t-test: HL vs LH (position effect)
hl_p  = df[df["condition"] == "HL"].groupby("participant_ID")["corr_score"].mean()
lh_p  = df[df["condition"] == "LH"].groupby("participant_ID")["corr_score"].mean()
pos_d = pd.DataFrame({"HL": hl_p, "LH": lh_p}).dropna()
if len(pos_d) > 1:
    t9, p9 = stats.ttest_rel(pos_d["HL"], pos_d["LH"])
    print(f"\n[ix]  Paired t-test — HL vs LH (position effect)")
    print(f"      HL M={pos_d['HL'].mean():.4f}  LH M={pos_d['LH'].mean():.4f}  n={len(pos_d)}")
    print(f"      t({len(pos_d)-1}) = {t9:.3f},  p = {p9:.4f}")

print(f"\nBonferroni-corrected alpha (0.05 / 9 tests) = {0.05/9:.5f}")


# ═════════════════════════════════════════════════════════════════════════════
# 11.  PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title, xlabel, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title(title,   fontsize=10, fontweight="bold", pad=9)
    ax.set_xlabel(xlabel, fontsize=9,  labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9,  labelpad=6)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

x = np.arange(len(COND_ORDER))
w = 0.35


# ═════════════════════════════════════════════════════════════════════════════
# 12.  FIGURE 1  —  Corrected Memorability Score by Condition × Voice
# ═════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

fig, ax = plt.subplots(figsize=(8, 4.8))
for i, voice in enumerate(VOICE_ORDER):
    sub   = agg[agg["encoding_voice"] == voice].set_index("condition").reindex(COND_ORDER)
    means = sub["mean_corr"].values.astype(float)
    sems  = np.nan_to_num(sub["sem_corr"].values.astype(float), nan=0.0)
    offset = (i - 0.5) * w

    bars = ax.bar(x + offset, means, w,
                  color=COLORS[voice], label=f"{voice} encoding",
                  edgecolor="white", linewidth=0.8, alpha=0.92,
                  yerr=sems, capsize=4,
                  error_kw=dict(elinewidth=1.2, capthick=1.2))

    for bar, m, se in zip(bars, means, sems):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + se + 0.006,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, fontsize=9)
ax.set_ylim(0, 1.05)
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(fontsize=9, framealpha=0.9)
style_ax(ax,
         title="Figure 1. Corrected Memorability Score by Condition and Encoding Voice",
         xlabel="Sentence Condition  (Subject Noun – Object Noun Memorability)",
         ylabel="Mean Corrected Score  (Hit Rate – FA Rate)  ±  1 SEM")
ax.text(0.01, 0.01,
        f"N = {N} participants  ·  Error bars = ±1 SEM  ·  Only validated blocks",
        transform=ax.transAxes, fontsize=7.5, color="gray", style="italic")
fig.tight_layout()
save(fig, "figure1_corrected_score.png")


# ═════════════════════════════════════════════════════════════════════════════
# 13.  FIGURE 2  —  Mean IR Reaction Time by Condition × Voice
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, voice in enumerate(VOICE_ORDER):
    sub   = agg[agg["encoding_voice"] == voice].set_index("condition").reindex(COND_ORDER)
    means = sub["mean_rt"].values.astype(float)
    sems  = np.nan_to_num(sub["sem_rt"].values.astype(float), nan=0.0)
    offset = (i - 0.5) * w

    bars = ax.bar(x + offset, means, w,
                  color=COLORS[voice], label=f"{voice} encoding",
                  edgecolor="white", linewidth=0.8, alpha=0.92,
                  yerr=sems, capsize=4,
                  error_kw=dict(elinewidth=1.2, capthick=1.2))

    for bar, m, se in zip(bars, means, sems):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + se + 8,
                    f"{int(round(m))}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, fontsize=9)
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
ax.legend(fontsize=9, framealpha=0.9)
style_ax(ax,
         title="Figure 2. Mean IR Reaction Time by Condition and Encoding Voice",
         xlabel="Sentence Condition  (Subject Noun – Object Noun Memorability)",
         ylabel="Mean IR Reaction Time (ms)  ±  1 SEM")
ax.text(0.01, 0.01,
        "Correct hits only  ·  RT range 100–8000 ms  ·  Only validated blocks",
        transform=ax.transAxes, fontsize=7.5, color="gray", style="italic")
fig.tight_layout()
save(fig, "figure2_reaction_time.png")


# ═════════════════════════════════════════════════════════════════════════════
# 14.  FIGURE 3  —  WR Proportion "Yes" by Condition × Voice
# ═════════════════════════════════════════════════════════════════════════════
wr_agg = (df[df["condition"].isin(COND_ORDER)]
          .groupby(["condition", "encoding_voice"])
          .agg(yes=("wr_yes", "sum"), no=("wr_no", "sum"))
          .reset_index())
wr_agg["total"]    = wr_agg["yes"] + wr_agg["no"]
wr_agg["prop_yes"] = wr_agg["yes"] / wr_agg["total"]

fig, ax = plt.subplots(figsize=(8, 4.8))
for i, voice in enumerate(VOICE_ORDER):
    sub    = (wr_agg[wr_agg["encoding_voice"] == voice]
              .set_index("condition").reindex(COND_ORDER))
    props  = sub["prop_yes"].values.astype(float)
    offset = (i - 0.5) * w

    bars = ax.bar(x + offset, props, w,
                  color=COLORS[voice], label=f"{voice} repeat",
                  edgecolor="white", linewidth=0.8, alpha=0.92)

    for bar, p in zip(bars, props):
        if not np.isnan(p):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{p:.2f}", ha="center", va="bottom", fontsize=8)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.3, label="Chance (0.50)")
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, framealpha=0.9)
style_ax(ax,
         title='Figure 3. Verbatim Recognition (WR) Proportion "Yes" by Condition and Repeat Voice',
         xlabel="Sentence Condition",
         ylabel='Proportion "Yes"  (Verbatim Match)')
ax.text(0.01, 0.01,
        "Active repeat: 'Yes' = correct  ·  Passive repeat: 'No' = correct",
        transform=ax.transAxes, fontsize=7.5, color="gray", style="italic")
fig.tight_layout()
save(fig, "figure3_wr_proportion_yes.png")


# ═════════════════════════════════════════════════════════════════════════════
# 15.  DONE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nAll done. Outputs saved to: {OUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")