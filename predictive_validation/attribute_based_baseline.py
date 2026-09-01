import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

np.random.seed(42)
df = pd.read_csv('FISAD_data.csv')  # run from the directory containing FISAD_data.csv, or edit this path

TRAITS = {'a':'Agreeableness','e':'Extraversion','o':'Openness','c':'Conscientiousness','n':'Neuroticism'}
FEATURE_COLS = ['Tops','Bottom','Tops Color','Bottom Color','Sleeve','Hair Style','Hair Color']

X_raw = df[FEATURE_COLS]
preprocessor = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore'), FEATURE_COLS)])

models = {
    'SVR': SVR(),
    'GBR': GradientBoostingRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'Ridge': Ridge(),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for trait_code, trait_name in TRAITS.items():
    y = df[trait_code].values
    for model_name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        scores = cross_validate(pipe, X_raw, y, cv=kf, scoring=('r2','neg_mean_squared_error'), n_jobs=1)
        r2 = np.mean(scores['test_r2'])
        nmse = np.mean(scores['test_neg_mean_squared_error'])
        results.append({'Trait': trait_name, 'Model': model_name, 'R2': r2, 'NMSE': nmse})
        print(f"{trait_name:20s} {model_name:8s}  R2={r2:.4f}  NMSE={nmse:.4f}")

pd.DataFrame(results).to_csv('baseline_complete_colors.csv', index=False)
