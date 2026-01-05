import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 1. LOAD DATA
df = pd.read_csv('sweden_player_ratings.csv')
df.columns = df.columns.str.strip().str.lower()

# Auto-detect minutes column
minutes_candidates = [c for c in df.columns if 'minute' in c]
if len(minutes_candidates) == 0:
    raise ValueError("No column containing 'minute' found in CSV")
minutes_col = minutes_candidates[0]
df.rename(columns={minutes_col: 'minutes'}, inplace=True)
df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)

# Identify stat columns
exclude = ['player', 'minutes', 'unnamed: 0']
stats = [c for c in df.columns if c not in exclude]
df[stats] = df[stats].apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. PER-90 NORMALISATION
for s in [c for c in stats if '%' not in c]:
    df[f'{s}_p90'] = (df[s] / df['minutes'].replace(0, 1)) * 90

feature_cols = [c for c in df.columns if '_p90' in c or '%' in c]

#Only use players with enough minutes for stats
min_minutes_for_stats = 100
eligible_df = df[df['minutes'] >= min_minutes_for_stats].copy()

# 3. TACTICAL CLUSTERING (exclude minutes)
# Use only per-90 stats or % stats for clustering
clustering_features = feature_cols.copy()  # minutes are NOT included
scaler = StandardScaler()
scaled_features = scaler.fit_transform(eligible_df[clustering_features])

# Automatically choose best number of clusters via silhouette score
best_n = 3
best_sil = -1
for n in range(3, 8):
    gmm = GaussianMixture(n_components=n, random_state=42).fit(scaled_features)
    labels = gmm.predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    if score > best_sil:
        best_sil = score
        best_n = n

# Fit final model
model = GaussianMixture(n_components=best_n, random_state=42).fit(scaled_features)
eligible_df['cluster_id'] = model.predict(scaled_features)

# 4. ROLE IMPORTANCE
squad_mean = eligible_df[feature_cols].mean()
squad_std = eligible_df[feature_cols].std() + 1e-6
cluster_means = eligible_df.groupby('cluster_id')[feature_cols].mean()

# how much does this cluster deviate from squad avg
importance = (cluster_means - squad_mean).abs() / squad_std
# stat thhat makes cluster most unique gets 1, most average gets 0
weights = importance.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6), axis=1)
# ensures no stat is ever 0 and ensures all stats add up to 100%
weights = (weights + 0.05).div((weights + 0.05).sum(axis=1), axis=0)

# 5. CALCULATE WEIGHTED Z-SCORES
def calculate_weighted_z(row):
    role_weights = weights.loc[row['cluster_id']]
    z = (row[feature_cols] - squad_mean) / squad_std
    return (z * role_weights).sum()

eligible_df['raw_z'] = eligible_df.apply(calculate_weighted_z, axis=1)

# DIMINISHING RETURNS ON MINUTES
def effective_minutes(minutes, cap=1800):
    return cap * (1 - np.exp(-minutes / cap))

# 6. RELIABILITY & STARTER'S BONUS
# This avoids sudden jumps for moderate-minute players
# Apply diminishing returns to minutes first
eligible_df['eff_minutes'] = eligible_df['minutes'].apply(lambda x: effective_minutes(x))

def reliability_weight(eff_minutes, midpoint=800):
    return eff_minutes / (eff_minutes + midpoint)

eligible_df['credibility'] = eligible_df['eff_minutes'].apply(reliability_weight)

eligible_df['final_z'] = eligible_df['raw_z'] * eligible_df['credibility']

# 7. PRO-SCALING (The Established Anchor + Sample Cap)
if not eligible_df.empty:
    # Anchor the 10.0 to the best player with 900+ minutes
    established_players = eligible_df[eligible_df['minutes'] >= 900]

    if not established_players.empty:
        z_anchor = established_players['final_z'].max()
    else:
        z_anchor = eligible_df['final_z'].max()

    def pro_scale(z):
        # 5.0 is the Squad Mean. 10.0 is the Established Star.
        return 5.0 + (z / z_anchor) * 5.0

    eligible_df['final_rating'] = eligible_df['final_z'].apply(pro_scale)

    # --- SAMPLE SIZE CAP ---
    # max 8.5 for players <300 min, then scales up to 10 at 900+
    def sample_cap(row):
        if row['minutes'] < 300:
            return min(row['final_rating'], 8.0)
        elif row['minutes'] < 900:
            scale = (row['minutes'] - 300) / (900 - 300)  # 0 → 1
            cap = 8.0 + 0.5 * scale  # 8 → 8.5 max
            return min(row['final_rating'], cap)
        else:
            return row['final_rating']

    eligible_df['final_rating'] = eligible_df.apply(sample_cap, axis=1)

    # managerial judgement floor: encourage established players
    eligible_df.loc[eligible_df['minutes'] >= 1000, 'final_rating'] = eligible_df.loc[eligible_df['minutes'] >= 1000, 'final_rating'].clip(lower=2.5)

# Final Clamp
eligible_df['final_rating'] = eligible_df['final_rating'].clip(0.0, 10.0)

# 8. MERGE & OUTPUT
res = eligible_df[['player', 'final_rating', 'cluster_id']]
df = df.drop(columns=['final_rating', 'cluster_id'], errors='ignore').merge(res, on='player', how='left')
df['final_rating'] = df['final_rating'].fillna(0.0).round(1)
df['cluster_id'] = df['cluster_id'].fillna(-1).astype(int)

# print summary metrics
squad_avg = eligible_df['final_rating'].mean()
print(f"Algorithm identified {best_n} distinct tactical roles.\n")
print(f"Squad average rating (>=100 min): {squad_avg:.2f}\n")
print("--- RELATIVE SQUAD RATINGS (0-10) ---")
print(df[['player', 'minutes', 'cluster_id', 'final_rating']].sort_values('final_rating', ascending=False))
