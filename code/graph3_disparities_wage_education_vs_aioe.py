
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIOE_PATH = "data/AIOE_DataAppendix(Appendix A).csv"
OEWS_2023_PATH = "data/national_M2023_dl(national_M2023_dl).csv"
EDU_PATH = "data/education(Table 5.csv"
OUT_FIG1 = "pictures/q3_median_wage_by_aioe_decile.png"
OUT_FIG2 = "pictures/q3_entry_level_education_share_top_vs_bottom.png"

def clean_soc(s):
    if pd.isna(s): return None
    s = str(s).strip().replace("–","-").replace("—","-")
    if "." in s: s = s.split(".")[0]
    return s

aioe = pd.read_csv(AIOE_PATH).rename(columns={"SOC Code":"SOC"})
aioe["SOC"] = aioe["SOC"].map(clean_soc)

o23 = pd.read_csv(OEWS_2023_PATH, low_memory=False)
o23.columns = [c.lower() for c in o23.columns]
o23 = o23.rename(columns={"occ_code":"SOC", "tot_emp":"Employment"})
o23["SOC"] = o23["SOC"].map(clean_soc)
o23 = o23[o23["SOC"].astype(str).str.len()==7]
o23["Employment"] = pd.to_numeric(o23["Employment"], errors="coerce")

# Wage: try annual median/mean; strip commas first; if still NaN, convert hourly median to annual (x2080)
wage_col = "a_median" if "a_median" in o23.columns else ("a_mean" if "a_mean" in o23.columns else None)
annual_wage = None
if wage_col is not None:
    s = o23[wage_col].astype(str).str.replace(",","", regex=False)
    annual_wage = pd.to_numeric(s, errors="coerce")

if annual_wage is None or annual_wage.isna().all():
    if "h_median" in o23.columns:
        hm = pd.to_numeric(o23["h_median"].astype(str).str.replace(",","", regex=False), errors="coerce")
        annual_wage = hm * 2080  # hours/year
    elif "h_mean" in o23.columns:
        hm = pd.to_numeric(o23["h_mean"].astype(str).str.replace(",","", regex=False), errors="coerce")
        annual_wage = hm * 2080

o23["Annual_Wage"] = annual_wage

df = aioe.merge(o23[["SOC","Employment","Annual_Wage"]], on="SOC", how="inner").dropna(subset=["Annual_Wage"])

# If very few rows, use #bins=min(10, max(2, len(df)))
bins = min(10, max(2, len(df)))
df["aioe_decile"] = pd.qcut(df["AIOE"].rank(method="first"), bins, labels=False) + 1

def wavg(x, w):
    import numpy as _np
    return _np.average(x, weights=w) if len(x)>0 else _np.nan

grp = df.groupby("aioe_decile").apply(
    lambda g: wavg(g["Annual_Wage"], g["Employment"].fillna(1.0))
).reset_index(name="employment_weighted_median_annual_wage")

plt.figure(figsize=(8,5))
plt.bar(grp["aioe_decile"], grp["employment_weighted_median_annual_wage"])
plt.xlabel("AI exposure bins (low → high)")
plt.ylabel("Median annual wage (employment-weighted)")
plt.title("Median annual wage by AI exposure (2023)")
plt.tight_layout()
plt.savefig(OUT_FIG1, dpi=300)
print("Saved:", OUT_FIG1)

# Education
edu = pd.read_csv(EDU_PATH, encoding="latin-1", skiprows=1)
soc_col = None
for c in edu.columns:
    cl = str(c).lower()
    if "matrix code" in cl or ("soc" in cl and "code" in cl):
        soc_col = c; break
if soc_col is None:
    soc_col = edu.columns[1]

edu = edu.rename(columns={soc_col:"SOC"})
edu["SOC"] = edu["SOC"].map(clean_soc)

bachelor_cols = [c for c in edu.columns if "Bachelor" in str(c)]
master_cols   = [c for c in edu.columns if "Master" in str(c)]
doctoral_cols = [c for c in edu.columns if ("Doctoral" in str(c)) or ("professional" in str(c).lower())]

edu_simple = edu[["SOC"] + bachelor_cols + master_cols + doctoral_cols].copy()
for c in edu_simple.columns:
    if c!="SOC":
        edu_simple[c] = pd.to_numeric(edu_simple[c], errors="coerce")
edu_simple["Bachelors_or_higher"] = edu_simple[bachelor_cols + master_cols + doctoral_cols].sum(axis=1)

merged = aioe.merge(o23[["SOC","Employment"]], on="SOC", how="inner") \
             .merge(edu_simple[["SOC","Bachelors_or_higher"]], on="SOC", how="inner") \
             .dropna(subset=["Bachelors_or_higher"])

q_low = merged["AIOE"].quantile(0.2)
q_high = merged["AIOE"].quantile(0.8)
low = merged[merged["AIOE"]<=q_low]
high = merged[merged["AIOE"]>=q_high]

def wavg2(x, w):
    import numpy as _np
    return _np.average(x, weights=w) if len(x)>0 else _np.nan

low_share = wavg2(low["Bachelors_or_higher"], low["Employment"])
high_share = wavg2(high["Bachelors_or_higher"], high["Employment"])

plt.figure(figsize=(6,5))
plt.bar(["Low AIOE (bottom 20%)","High AIOE (top 20%)"], [low_share, high_share])
plt.ylabel("Share with Bachelor's degree or higher (%)")
plt.title("Education disparity by AI exposure (employment-weighted)")
plt.tight_layout()
plt.savefig(OUT_FIG2, dpi=300)
print("Saved:", OUT_FIG2)
