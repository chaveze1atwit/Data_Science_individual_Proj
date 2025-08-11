import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIOE_PATH = "data/AIOE_DataAppendix(Appendix A).csv"
OEWS_2013_PATH = "data/national_M2013_dl(national_dl).csv"
OEWS_2023_PATH = "data/national_M2023_dl(national_M2023_dl).csv"
OUT_FIG = "pictures/q1_employment_growth_by_aioe_decile.png"

def clean_soc(s):
    if pd.isna(s): return None
    s = str(s).strip().replace("–","-").replace("—","-")
    if "." in s: s = s.split(".")[0]
    return s

aioe = pd.read_csv(AIOE_PATH)
aioe.columns = [c.strip() for c in aioe.columns]
if "SOC Code" in aioe.columns:
    aioe = aioe.rename(columns={"SOC Code":"SOC"})
else:
    for c in aioe.columns:
        if "soc" in c.lower() and "code" in c.lower():
            aioe = aioe.rename(columns={c:"SOC"}); break
aioe["SOC"] = aioe["SOC"].map(clean_soc)
aioe = aioe[["SOC","AIOE"]].dropna()

o13 = pd.read_csv(OEWS_2013_PATH, low_memory=False)
o13.columns = [c.lower() for c in o13.columns]
o13 = o13.rename(columns={"occ_code":"SOC","tot_emp":"Employment_2013"})
o13["SOC"] = o13["SOC"].map(clean_soc)
o13 = o13[o13["SOC"].astype(str).str.len()==7]
o13["Employment_2013"] = pd.to_numeric(o13["Employment_2013"], errors="coerce")

o23 = pd.read_csv(OEWS_2023_PATH, low_memory=False)
o23.columns = [c.lower() for c in o23.columns]
o23 = o23.rename(columns={"occ_code":"SOC","tot_emp":"Employment_2023"})
o23["SOC"] = o23["SOC"].map(clean_soc)
o23 = o23[o23["SOC"].astype(str).str.len()==7]
o23["Employment_2023"] = pd.to_numeric(o23["Employment_2023"], errors="coerce")

df = aioe.merge(o13[["SOC","Employment_2013"]], on="SOC", how="inner") \
         .merge(o23[["SOC","Employment_2023"]], on="SOC", how="inner")

# If very few rows, continue without filtering to avoid empty bins
df = df.dropna(subset=["Employment_2013","Employment_2023"])

# Guard: need at least ~10 rows for deciles; if fewer, use quantiles by available bins
n = len(df)
bins = min(10, max(2, n))
df["aioe_bin"] = pd.qcut(df["AIOE"].rank(method="first"), bins, labels=False) + 1

def wavg(x, w):
    import numpy as _np
    return _np.average(x, weights=w) if len(x)>0 else _np.nan

df["pct_change"] = 100.0*(df["Employment_2023"] - df["Employment_2013"])/df["Employment_2013"]
grp = df.groupby("aioe_bin").apply(
    lambda g: wavg(g["pct_change"], g["Employment_2013"].fillna(1.0))
).reset_index(name="weighted_pct_change")

plt.figure(figsize=(8,5))
plt.bar(grp["aioe_bin"], grp["weighted_pct_change"])
plt.xlabel("AI exposure bins (low → high)")
plt.ylabel("Employment growth 2013–2023 (%)")
plt.title("Employment growth by AI exposure (bins adapted to data size)")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print("Saved:", OUT_FIG)
