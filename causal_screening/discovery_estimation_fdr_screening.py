import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

np.random.seed(42)
df = pd.read_csv('FISAD_data.csv')  # run from the directory containing FISAD_data.csv, or edit this path

TRAITS = {'a':'Agreeableness','e':'Extraversion','o':'Openness','c':'Conscientiousness','n':'Neuroticism'}

# Build composite features using the NOW-COMPLETE Tops Color / Bottom Color
df['Tops_Bottom'] = df['Tops'].astype(str) + '_' + df['Bottom'].astype(str)
df['Tops_Color'] = df['Tops'].astype(str) + '_' + df['Tops Color'].astype(str)
df['Color_Sleeve'] = df['Tops Color'].astype(str) + '_' + df['Sleeve'].astype(str)
df['Tops_Sleeve'] = df['Tops'].astype(str) + '_' + df['Sleeve'].astype(str)
df['Tops_Color_Sleeve'] = df['Tops'].astype(str) + '_' + df['Tops Color'].astype(str) + '_' + df['Sleeve'].astype(str)

SINGLE_FEATURES = ['Tops','Bottom','Tops Color','Bottom Color','Sleeve','Hair Style','Hair Color']
COMPOSITE_FEATURES = ['Tops_Bottom','Tops_Color','Color_Sleeve','Tops_Sleeve','Tops_Color_Sleeve']
ALL_FEATURES = SINGLE_FEATURES + COMPOSITE_FEATURES

# Discovery/estimation split
shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n_half = len(shuffled) // 2
discovery = shuffled.iloc[:n_half].copy()

print(f"Discovery sample: {len(discovery)}\n")

results = []
for trait, trait_name in TRAITS.items():
    pvals = []
    for feat in ALL_FEATURES:
        groups = [g[trait].values for _, g in discovery.groupby(feat) if len(g) >= 5]
        if len(groups) < 2:
            pvals.append(1.0); continue
        _, pval = stats.f_oneway(*groups)
        pvals.append(pval)
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for feat, p_raw, p_adj, sig in zip(ALL_FEATURES, pvals, pvals_adj, reject):
        results.append({'Trait': trait_name, 'Feature': feat, 'p_raw': p_raw, 'p_fdr': p_adj, 'sig': sig})

results_df = pd.DataFrame(results)
results_df.to_csv('screening_results_complete.csv', index=False)

print("=== FDR-corrected screening results (complete Tops/Bottom Color data, 12 features tested) ===\n")
for t in TRAITS.values():
    sig = results_df[(results_df['Trait']==t) & (results_df['sig'])]
    print(f"{t}: {list(sig['Feature']) if len(sig) else 'NONE'}")
