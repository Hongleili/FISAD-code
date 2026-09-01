"""
FISAD CLIP Embedding Predictive Baseline
===========================================
Extracts CLIP (ViT-B/32) image embeddings directly from the FISAD images
and evaluates predictive power for each Big Five trait via 5-fold
cross-validation, using both a properly regularized linear model (RidgeCV)
and a nonlinear model (Gradient Boosting on PCA-reduced embeddings).

This provides a feature representation that does NOT depend on any
discretized apparel attribute labeling, serving as an independent check
on whether predictive power is capped by measurement reliability (as
opposed to attribute-extraction quality).

Usage:
    python3 clip_baseline.py --input FISAD_data.csv --embeddings-out embeddings.npz
    (first run downloads images and computes embeddings, checkpointed;
     re-run to resume or to skip straight to modelling if embeddings.npz exists)
"""
import argparse
import os
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import concurrent.futures

TRAITS = {'a': 'Agreeableness', 'e': 'Extraversion', 'o': 'Openness',
          'c': 'Conscientiousness', 'n': 'Neuroticism'}


def extract_embeddings(df, output_path, batch_size=64, max_workers=16, time_budget=100):
    """Extracts CLIP ViT-B/32 embeddings for all images in df, checkpointing
    progress to output_path so the process can be resumed if interrupted."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', quick_gelu=True
    )
    model.eval()

    existing_embeddings = {}
    if os.path.exists(output_path):
        data = np.load(output_path)
        for i, emb in zip(data['ids'], data['embeddings']):
            existing_embeddings[int(i)] = emb
        print(f"Resuming: {len(existing_embeddings)} embeddings already computed.")

    todo = df[~df['id'].isin(existing_embeddings.keys())].reset_index(drop=True)
    print(f"Remaining: {len(todo)} images.")

    def fetch_and_preprocess(row):
        try:
            resp = requests.get(row['path'], timeout=15)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            return row['id'], preprocess(img)
        except Exception:
            return row['id'], None

    start_time = time.time()
    new_results = {}
    i = 0
    while i < len(todo) and (time.time() - start_time) < time_budget:
        batch_rows = todo.iloc[i:i + batch_size].to_dict('records')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(fetch_and_preprocess, batch_rows))
        valid = [(img_id, tensor) for img_id, tensor in results if tensor is not None]
        if valid:
            ids_batch = [v[0] for v in valid]
            tensors_batch = torch.stack([v[1] for v in valid])
            with torch.no_grad():
                embs_batch = model.encode_image(tensors_batch).numpy()
            for img_id, emb in zip(ids_batch, embs_batch):
                new_results[int(img_id)] = emb
        i += batch_size

    all_embeddings = {**existing_embeddings, **new_results}
    ids_arr = np.array(list(all_embeddings.keys()))
    embs_arr = np.array(list(all_embeddings.values()))
    np.savez(output_path, ids=ids_arr, embeddings=embs_arr)
    print(f"Saved {len(all_embeddings)}/{len(df)} total embeddings.")
    return len(all_embeddings) == len(df)


def run_baseline(df, embeddings_path):
    """Runs RidgeCV (linear) and GBR-on-PCA (nonlinear) 5-fold CV baselines
    using the extracted CLIP embeddings."""
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    data = np.load(embeddings_path)
    emb_df = pd.DataFrame(data['embeddings'], index=data['ids']).reset_index()
    emb_df.columns = ['id'] + [f'e{i}' for i in range(data['embeddings'].shape[1])]
    merged = df.merge(emb_df, on='id')
    X = merged[[c for c in emb_df.columns if c != 'id']].values

    ridge_pipe = Pipeline([('scale', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-2, 4, 20)))])
    gbr_pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=50, random_state=42)),
                          ('gbr', GradientBoostingRegressor(random_state=42, n_estimators=100))])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{'Trait':20s} {'RidgeCV R2':>12s} {'GBR(PCA50) R2':>16s}")
    for code, name in TRAITS.items():
        y = merged[code].values
        r2_ridge = np.mean(cross_validate(ridge_pipe, X, y, cv=kf, scoring='r2')['test_score'])
        r2_gbr = np.mean(cross_validate(gbr_pipe, X, y, cv=kf, scoring='r2')['test_score'])
        print(f"{name:20s} {r2_ridge:12.4f} {r2_gbr:16.4f}")


def main():
    parser = argparse.ArgumentParser(description="FISAD CLIP embedding baseline")
    parser.add_argument("--input", default="FISAD_data.csv")
    parser.add_argument("--embeddings-out", default="clip_embeddings.npz")
    args = parser.parse_args()

    df = pd.read_csv(args.input)[['id', 'path', 'a', 'e', 'o', 'c', 'n']]
    complete = extract_embeddings(df, args.embeddings_out)
    if not complete:
        print("\nNot all embeddings extracted yet - re-run this script to resume.")
        return
    run_baseline(df, args.embeddings_out)


if __name__ == "__main__":
    main()
