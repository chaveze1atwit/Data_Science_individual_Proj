
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/share-artificial-intelligence-job-postings.csv"
OUT_FIG = "pictures/ctx_ai_job_postings_share_us.png"

df = pd.read_csv(CSV_PATH)
# Value column may be named 'Share of artificial intelligence jobs among all job postings'
value_cols = [c for c in df.columns if c not in ["Entity","Code","Year"]]
value_col = value_cols[-1]
us = df[df["Code"]=="USA"].sort_values("Year")

plt.figure(figsize=(8,5))
plt.plot(us["Year"], us[value_col])
plt.xlabel("Year")
plt.ylabel("Share of job postings mentioning AI")
plt.title("AI-related job postings share in the United States")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print("Saved:", OUT_FIG)
