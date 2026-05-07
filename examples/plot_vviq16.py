"""Standalone VVIQ-16 figure generator.

Reads vviq16_singleturn.csv and vviq16_convo.csv from the same directory,
extracts Vividness ratings from the pred_resp_raw column, and saves a
two-panel scatter figure (one panel per memory condition).

Run:
    python plot_vviq16.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent

# ── load ──────────────────────────────────────────────────────────────────────
st_path = HERE / "vviq16_singleturn.csv"
cv_path = HERE / "vviq16_convo.csv"

for p in (st_path, cv_path):
    if not p.exists():
        sys.exit(f"Missing file: {p}")

df_st = pd.read_csv(st_path)
df_cv = pd.read_csv(cv_path)


# ── parse Vividness from pred_resp_raw ────────────────────────────────────────
_VIV_RE = re.compile(r"['\"]?Vividness['\"]?\s*:\s*(\d+)")

def extract_vividness(raw: pd.Series) -> pd.Series:
    def _parse(s):
        if not isinstance(s, str):
            return float("nan")
        m = _VIV_RE.search(s)
        return int(m.group(1)) if m else float("nan")
    return raw.apply(_parse)


df_st["Vividness"] = extract_vividness(df_st["pred_resp_raw"])
df_cv["Vividness"] = extract_vividness(df_cv["pred_resp_raw"])

n_missing_st = df_st["Vividness"].isna().sum()
n_missing_cv = df_cv["Vividness"].isna().sum()
if n_missing_st or n_missing_cv:
    print(f"Warning: could not parse Vividness for "
          f"{n_missing_st} SingleTurn / {n_missing_cv} Convo rows")

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_st, ax_cv) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

JITTER = 0.10  # small horizontal jitter to reveal overlapping points
import numpy as np
rng = np.random.default_rng(42)


def _jitter(x, scale=JITTER):
    return x + rng.uniform(-scale, scale, size=len(x))


# ── Left: No Memory ───────────────────────────────────────────────────────────
x_st = (df_st["trial_idx"] + 1).values.astype(float)
y_st = df_st["Vividness"].values.astype(float)
ax_st.scatter(_jitter(x_st), y_st,
              color="steelblue", s=55, alpha=0.80, linewidths=0.3,
              edgecolors="white", zorder=3)
ax_st.set_title("No Memory", fontsize=14, fontweight="bold")
ax_st.set_xlabel("Trial Number", fontsize=12)
ax_st.set_ylabel("Rating response (1 to 5)", fontsize=12)
ax_st.set_xlim(0.5, 16.5)
ax_st.set_xticks(range(1, 17))
ax_st.set_xticklabels(range(1, 17), fontsize=8)
ax_st.set_ylim(0.5, 5.5)
ax_st.set_yticks([1, 2, 3, 4, 5])
ax_st.grid(axis="y", linestyle="--", alpha=0.4)
ax_st.spines[["top", "right"]].set_visible(False)

# ── Right: Conversation Memory ────────────────────────────────────────────────
x_cv = (df_cv["trial_idx"] + 1).values.astype(float)
y_cv = df_cv["Vividness"].values.astype(float)
ax_cv.scatter(_jitter(x_cv), y_cv,
              color="gray", s=55, alpha=0.80, linewidths=0.3,
              edgecolors="white", zorder=3)
ax_cv.set_title("Conversation Memory", fontsize=14, fontweight="bold")
ax_cv.set_xlabel("Trial Number", fontsize=12)
ax_cv.set_xlim(0.5, 16.5)
ax_cv.set_xticks(range(1, 17))
ax_cv.set_xticklabels(range(1, 17), fontsize=8)
ax_cv.set_ylim(0.5, 5.5)
ax_cv.set_yticks([1, 2, 3, 4, 5])
ax_cv.grid(axis="y", linestyle="--", alpha=0.4)
ax_cv.spines[["top", "right"]].set_visible(False)

# model name from first row
model_name = df_st["model"].iloc[0] if "model" in df_st.columns else "LLM"
nsim = df_st["trace_id"].nunique() if "trace_id" in df_st.columns else "?"

fig.suptitle(
    f"VVIQ-16 (Imagination Questionnaire) response distribution\n"
    f"Model: {model_name}   ·   {nsim} simulations × 16 trials",
    fontsize=13, y=1.03,
)
plt.tight_layout()

out = HERE / "vviq16_figure.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out}")
plt.show()

# ── quick stats ───────────────────────────────────────────────────────────────
print("\nMean Vividness by condition:")
print(f"  SingleTurn : {df_st['Vividness'].mean():.2f}  (SD={df_st['Vividness'].std():.2f})")
print(f"  Convo      : {df_cv['Vividness'].mean():.2f}  (SD={df_cv['Vividness'].std():.2f})")
