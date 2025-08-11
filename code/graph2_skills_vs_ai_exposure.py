
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIOE_PATH = "data/AIOE_DataAppendix(Appendix A).csv"
BLS_SKILLS_PATH = "data/skills(Table 6.csv"
OUT_FIG = "pictures/q2_skills_correlation_with_aioe.png"
OUT_TABLE = "data/q2_top_skills_correlations.csv"

def clean_soc(s):
    if pd.isna(s): return None
    s = str(s).strip().replace("–","-").replace("—","-")
    if "." in s: s = s.split(".")[0]
    return s

aioe = pd.read_csv(AIOE_PATH).rename(columns={"SOC Code":"SOC"})
aioe["SOC"] = aioe["SOC"].map(clean_soc)

skills = pd.read_csv(BLS_SKILLS_PATH, encoding="latin-1", skiprows=1)
# SOC column name in this file:
soc_col = None
for c in skills.columns:
    if "matrix code" in str(c).lower() or ("soc" in str(c).lower() and "code" in str(c).lower()):
        soc_col = c; break
if soc_col is None:
    soc_col = skills.columns[1]
skills = skills.rename(columns={soc_col:"SOC"})
skills["SOC"] = skills["SOC"].map(clean_soc)

# Keep numeric columns that are clearly skill categories (exclude employment and wage columns)
exclude_keywords = ["employment", "wage", "title", "education", "code", "percent", "change"]
score_cols = []
for c in skills.columns:
    if c=="SOC": continue
    if not pd.api.types.is_numeric_dtype(skills[c]): continue
    cl = str(c).lower()
    if any(k in cl for k in exclude_keywords):
        continue
    score_cols.append(c)

df = aioe.merge(skills[["SOC"]+score_cols], on="SOC", how="inner")

cors = []
for col in score_cols:
    cors.append((col, df[col].corr(df["AIOE"])))
cor_df = pd.DataFrame(cors, columns=["Skill_Category","Correlation"]).sort_values("Correlation", ascending=False)
cor_df.to_csv(OUT_TABLE, index=False)

top = cor_df.head(10)
bottom = cor_df.tail(10)
plot_df = pd.concat([top, bottom])

plt.figure(figsize=(9,6))
y = np.arange(len(plot_df))
plt.barh(y, plot_df["Correlation"])
plt.yticks(y, plot_df["Skill_Category"])
plt.axvline(0, linewidth=1)
plt.xlabel("Correlation with AI exposure (AIOE)")
plt.title("Skill categories most associated with AI exposure")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print("Saved:", OUT_FIG)
print("Saved table:", OUT_TABLE)
