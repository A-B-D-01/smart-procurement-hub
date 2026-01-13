# models/ml_recommendation_engine.py
"""
MLRecommendationEngine + EmbeddingEngine combined.

Provides:
 - Trainer (train_all_models)
 - MLRecommendationEngine (load, predict_all, recommend_suppliers)
 - EmbeddingEngine (TF-IDF based search, fallback substring scoring)
 - load_embedding_engine() convenience function
 - train_models(...) helper + CLI entrypoint
"""
import os
import json
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib

# ML libs (sklearn)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# TF-IDF / semantic dependencies (optional — fallback available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------
# Utilities (same as earlier)
# -----------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

def map_single_risk_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Mid"
    try:
        n = float(v)
        if n <= 2:
            return "Low"
        elif n == 3:
            return "Mid"
        else:
            return "High"
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in ("low",):
        return "Low"
    if s in ("mid", "medium"):
        return "Mid"
    if s in ("high",):
        return "High"
    return "Mid"

def compute_overall_risk_level_from_row(row):
    score_map = {"Low": 1.0, "Mid": 2.0, "High": 3.0}
    vals = [
        score_map[map_single_risk_value(row.get("political_risk"))],
        score_map[map_single_risk_value(row.get("trade_risk"))],
        score_map[map_single_risk_value(row.get("natural_disaster_risk"))]
    ]
    avg = sum(vals) / len(vals)
    if avg < 1.5:
        return "Low"
    elif avg < 2.3:
        return "Mid"
    else:
        return "High"

def combine_risk_score_normalized(row):
    mapping = {'Low': 0.0, 'Mid': 0.5, 'High': 1.0}
    vals = []
    for col in ['political_risk', 'trade_risk', 'natural_disaster_risk']:
        v = row.get(col, None)
        mapped = mapping.get(map_single_risk_value(v).title(), 0.5)
        vals.append(mapped)
    credit = row.get('credit_rating', None)
    credit_score = 0.5
    if pd.notna(credit) and isinstance(credit, str):
        s = credit.strip().upper()
        if s.startswith('AAA'):
            credit_score = 0.0
        elif s.startswith('AA'):
            credit_score = 0.1
        elif s.startswith('A'):
            credit_score = 0.2
        elif s.startswith('BBB'):
            credit_score = 0.4
        else:
            credit_score = 0.7
    vals.append(credit_score)
    return float(np.clip(np.mean(vals), 0.0, 1.0))

# -----------------------
# Feature engineering (keeps your previous logic)
# -----------------------
def build_features_for_supplier_level(suppliers: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    suppliers = suppliers.copy() if suppliers is not None else pd.DataFrame()
    pos = pos.copy() if pos is not None else pd.DataFrame()

    if suppliers.empty:
        return pd.DataFrame()

    if 'supplier_id' not in suppliers.columns:
        raise ValueError("suppliers DataFrame must contain 'supplier_id'")

    suppliers['supplier_id'] = suppliers['supplier_id'].astype(str)

    if pos is None or pos.empty or 'supplier_id' not in pos.columns:
        sup = suppliers.set_index('supplier_id', drop=False)
        sup['total_historical_orders'] = 0
        sup['days_since_last_order'] = np.nan
        sup['total_order_value'] = 0.0
        sup['completion_rate'] = 0.0
        sup['on_time_rate'] = 0.0
        sup['lead_time_avg'] = np.nan
        sup['lead_time_std'] = np.nan
        sup['price_mean'] = np.nan
        sup['price_std'] = np.nan
        sup['overall_risk_score'] = sup.apply(combine_risk_score_normalized, axis=1)
        sup = sup.reset_index(drop=True)
        return sup

    pos['supplier_id'] = pos['supplier_id'].astype(str)
    if 'order_date' in pos.columns:
        pos['order_date'] = pd.to_datetime(pos['order_date'], errors='coerce')
    if 'expected_delivery_date' in pos.columns:
        pos['expected_delivery_date'] = pd.to_datetime(pos['expected_delivery_date'], errors='coerce')
    if 'actual_delivery_date' in pos.columns:
        pos['actual_delivery_date'] = pd.to_datetime(pos['actual_delivery_date'], errors='coerce')

    ensure_numeric(pos, ['total_amount', 'unit_price', 'quantity'])

    agg = pos.groupby('supplier_id').agg(
        total_historical_orders=('po_number', 'count'),
        days_since_last_order=('order_date', lambda s: (pd.Timestamp.now() - s.max()).days if not s.isna().all() else np.nan),
        total_order_value=('total_amount', lambda s: pd.to_numeric(s, errors='coerce').sum()),
        price_mean=('unit_price', lambda s: pd.to_numeric(s, errors='coerce').mean()),
        price_std=('unit_price', lambda s: pd.to_numeric(s, errors='coerce').std())
    ).fillna(0)

    def comp_stats(df_group):
        total = len(df_group)
        completed = df_group
        if 'status' in df_group.columns:
            completed = df_group[df_group['status'] == 'Completed']
        completion_rate = (len(completed) / total) * 100 if total > 0 else 0.0

        on_time_rate = 0.0
        lead_avg = np.nan
        lead_std = np.nan
        if 'actual_delivery_date' in df_group.columns and 'expected_delivery_date' in df_group.columns:
            compc = df_group[df_group['actual_delivery_date'].notna() & df_group['expected_delivery_date'].notna()]
            if not compc.empty:
                compc = compc.copy()
                compc['on_time'] = compc['actual_delivery_date'] <= compc['expected_delivery_date']
                on_time_rate = (compc['on_time'].sum() / max(len(compc), 1)) * 100
                if 'order_date' in compc.columns:
                    lt = (compc['actual_delivery_date'] - compc['order_date']).dt.days.dropna()
                    if len(lt) > 0:
                        lead_avg = float(lt.mean())
                        lead_std = float(lt.std())
        return pd.Series({
            'completion_rate': completion_rate,
            'on_time_rate': on_time_rate,
            'lead_time_avg': (lead_avg if not pd.isna(lead_avg) else 0.0),
            'lead_time_std': (lead_std if not pd.isna(lead_std) else 0.0)
        })

    try:
        comp = pos.groupby('supplier_id', group_keys=False).apply(comp_stats).fillna(0)
    except TypeError:
        comp = pos.groupby('supplier_id').apply(comp_stats).fillna(0)
    except Exception:
        comp = pos.groupby('supplier_id').apply(comp_stats).fillna(0)

    features = pd.concat([agg, comp], axis=1).fillna(0)

    suppliers_indexed = suppliers.set_index('supplier_id', drop=False)
    merged = suppliers_indexed.join(features, how='left').fillna(0)

    merged['avg_order_value'] = merged.apply(
        lambda r: (r.get('total_order_value', 0.0) / max(int(r.get('total_historical_orders', 1)) , 1)), axis=1
    )
    merged['price_stability'] = merged['price_std'].apply(lambda x: 1.0 / (1.0 + (x if not pd.isna(x) else 0.0)))
    merged['overall_risk_score'] = merged.apply(combine_risk_score_normalized, axis=1)

    for c in ['industry', 'country']:
        if c in merged.columns:
            try:
                le = LabelEncoder()
                merged[c + '_enc'] = le.fit_transform(merged[c].astype(str).fillna(''))
            except Exception:
                merged[c + '_enc'] = 0

    merged = merged.reset_index(drop=True)
    merged['supplier_id'] = merged['supplier_id'].astype(str)
    return merged

# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _save(self, obj, name: str):
        path = os.path.join(self.model_dir, name)
        if isinstance(obj, dict) and name.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
        else:
            joblib.dump(obj, path)

    def _load(self, name: str):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            if name.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            return joblib.load(path)
        return None

    def train_all_models(self, suppliers_csv: str, pos_csv: str, force_retrain: bool=False) -> Dict[str, Any]:
        suppliers = safe_read_csv(suppliers_csv)
        pos = safe_read_csv(pos_csv)

        if suppliers.empty:
            raise ValueError("Trainer: suppliers.csv empty")

        features = build_features_for_supplier_level(suppliers, pos)
        if features.empty:
            raise ValueError("Trainer: feature table empty")

        numeric = features.select_dtypes(include=[np.number]).copy()
        numeric = numeric.loc[:, numeric.columns != 'supplier_id']

        target = (
            0.30 * (features['completion_rate'] / 100.0) +
            0.30 * (features['on_time_rate'] / 100.0) +
            0.20 * (features.get('quality_score', 0) / 5.0) +
            0.20 * (1.0 - features['overall_risk_score'])
        ).clip(0, 1)

        X = numeric.fillna(0)
        y = target.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
        model.fit(X_train_scaled, y_train)

        self._save(scaler, 'scaler.joblib')
        self._save(model, 'final_regressor.joblib')

        metadata = {
            "feature_columns": list(X.columns),
            "model_type": "GradientBoostingRegressor",
        }
        self._save(metadata, 'metadata.json')

        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        return {"train_score": float(train_score), "test_score": float(test_score), "metadata": metadata}

# -----------------------
# MLRecommendationEngine
# -----------------------
class MLRecommendationEngine:
    def __init__(self, model_dir: str, suppliers_df: pd.DataFrame, pos_df: Optional[pd.DataFrame] = None):
        self.model_dir = model_dir
        self.suppliers_df = suppliers_df.copy() if suppliers_df is not None else pd.DataFrame()
        self.pos_df = pos_df.copy() if pos_df is not None else pd.DataFrame()

        self.scaler = None
        self.model = None
        self.metadata = None
        self.feature_columns = []

        self._load_artifacts()

    def _load_artifacts(self):
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        model_path = os.path.join(self.model_dir, 'final_regressor.joblib')
        meta_path = os.path.join(self.model_dir, 'metadata.json')

        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.scaler = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_columns = self.metadata.get('feature_columns', [])
            except Exception:
                self.metadata = None
                self.feature_columns = []

    @classmethod
    def load(cls, model_dir: str, suppliers_df: pd.DataFrame, pos_df: Optional[pd.DataFrame] = None):
        return cls(model_dir, suppliers_df, pos_df)

    def predict_all(self) -> pd.DataFrame:
        features = build_features_for_supplier_level(self.suppliers_df, self.pos_df)
        if features.empty:
            return pd.DataFrame()

        X = features.copy()
        if self.feature_columns:
            missing = [c for c in self.feature_columns if c not in X.columns]
            for c in missing:
                X[c] = 0.0
            X_model = X[self.feature_columns].fillna(0)
        else:
            X_model = X.select_dtypes(include=[np.number]).fillna(0)
            self.feature_columns = list(X_model.columns)

        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X_model)
            except Exception:
                X_scaled = X_model.values
        else:
            X_scaled = X_model.values

        if self.model is not None:
            try:
                preds = self.model.predict(X_scaled)
                preds = np.clip(preds, 0.0, 1.0)
            except Exception:
                preds = np.zeros(X_scaled.shape[0])
        else:
            preds = np.zeros(X_scaled.shape[0])

        df_out = features[['supplier_id']].copy()
        df_out['ml_score'] = preds
        df_out['risk_numeric'] = features['overall_risk_score'].fillna(0.5).astype(float)
        df_out['final_recommendation_score'] = df_out['ml_score'].astype(float)

        return df_out

    def recommend_suppliers(self, top_n: int = 20, semantic_scores: Optional[Dict[str, float]] = None,
                            alpha_ml: float = 0.7, alpha_semantic: float = 0.3, risk_penalty_weight: float = 0.25):
        preds = self.predict_all()
        if preds.empty:
            return []

        df = self.suppliers_df.copy().astype(object)
        df['supplier_id'] = df['supplier_id'].astype(str)
        df = df.merge(preds, on='supplier_id', how='left').fillna(0)

        df['semantic_score'] = 0.0
        if semantic_scores:
            df['semantic_score'] = df['supplier_id'].map(semantic_scores).fillna(0.0)

        df['final_score'] = (
            alpha_ml * df['ml_score'].astype(float) +
            alpha_semantic * df['semantic_score'].astype(float) -
            (risk_penalty_weight * df['risk_numeric'].astype(float))
        ).clip(0, 1)

        if (df['final_score'].max() - df['final_score'].min()) < 1e-8:
            df['final_score'] = df['ml_score']

        df_sorted = df.sort_values('final_score', ascending=False).head(top_n)

        out = []
        for _, r in df_sorted.iterrows():
            out.append({
                "supplier_id": str(r['supplier_id']),
                "name": r.get('name', ''),
                "industry": r.get('industry', ''),
                "city": r.get('city', ''),
                "country": r.get('country', ''),
                "ml_score": float(r.get('ml_score', 0.0)),
                "semantic_score": float(r.get('semantic_score', 0.0)),
                "risk_level": compute_overall_risk_level_from_row(r),
                "final_recommendation_score": float(r.get('final_score', 0.0)),
                "savings_potential": float((r.get('total_business_value') or 0) * 0.05)
            })
        return out

# -----------------------
# EmbeddingEngine (TF-IDF + fallback)
# -----------------------
def _safe_text(x):
    if x is None:
        return ""
    return str(x).strip()

class EmbeddingEngine:
    """
    Simple TF-IDF embedding/search engine over suppliers.
    Methods:
      - enabled() -> bool
      - ensure(df) -> builds internal corpus
      - search(query, top_k=10) -> list[(supplier_id, score)]
    """
    def __init__(self):
        self._ready = False
        self._supplier_ids: List[str] = []
        self._corpus: List[str] = []
        self._vectorizer = None
        self._tfidf_matrix = None

    def enabled(self) -> bool:
        # always available in fallback mode
        return True

    def ensure(self, df: pd.DataFrame):
        if df is None or df.empty:
            self._ready = False
            self._supplier_ids = []
            self._corpus = []
            self._vectorizer = None
            self._tfidf_matrix = None
            return

        rows = []
        ids = []
        for _, r in df.iterrows():
            sid = str(r.get('supplier_id', ''))
            name = _safe_text(r.get('name', ''))
            industry = _safe_text(r.get('industry', ''))
            city = _safe_text(r.get('city', ''))
            country = _safe_text(r.get('country', ''))
            extras = _safe_text(r.get('description', '') or r.get('notes', ''))
            text = " ".join([name, industry, city, country, extras])
            if text.strip() == "":
                text = name or industry or city or country or sid
            rows.append(text)
            ids.append(sid)

        self._supplier_ids = ids
        self._corpus = rows

        if SKLEARN_AVAILABLE:
            try:
                self._vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
                self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
                self._ready = True
            except Exception:
                self._vectorizer = None
                self._tfidf_matrix = None
                self._ready = True
        else:
            self._vectorizer = None
            self._tfidf_matrix = None
            self._ready = True

    def _score_substring(self, query: str, text: str) -> float:
        qtokens = [t for t in query.lower().split() if t.strip()]
        if not qtokens:
            return 0.0
        text_l = text.lower()
        hit = sum(1 for t in qtokens if t in text_l)
        return float(hit) / len(qtokens)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query = _safe_text(query)
        if not query:
            return []
        if not self._ready or not self._supplier_ids:
            return []
        results: List[Tuple[str, float]] = []
        if SKLEARN_AVAILABLE and self._tfidf_matrix is not None and self._vectorizer is not None:
            try:
                qvec = self._vectorizer.transform([query])
                sims = cosine_similarity(qvec, self._tfidf_matrix).flatten()
                for sid, s in zip(self._supplier_ids, sims.tolist()):
                    score = float(max(0.0, float(s)))
                    results.append((sid, score))
            except Exception:
                for sid, text in zip(self._supplier_ids, self._corpus):
                    results.append((sid, self._score_substring(query, text)))
        else:
            for sid, text in zip(self._supplier_ids, self._corpus):
                results.append((sid, self._score_substring(query, text)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:min(top_k, len(results))]

def load_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()

# -----------------------
# Training helper + CLI
# -----------------------
def train_models(suppliers_csv: str, pos_csv: str, model_dir: str, force_retrain: bool=False) -> Dict[str, Any]:
    """
    Convenience wrapper: train models and save artifacts to model_dir.
    Returns the same dict Trainer.train_all_models returns.
    """
    trainer = Trainer(model_dir)
    return trainer.train_all_models(suppliers_csv, pos_csv, force_retrain=force_retrain)

if __name__ == "__main__":
    import argparse
    import traceback
    parser = argparse.ArgumentParser(description="Train ML artifacts from models.ml_recommendation_engine")
    parser.add_argument("--suppliers", default="data/suppliers.csv", help="Path to suppliers CSV")
    parser.add_argument("--pos", default="data/purchase_orders.csv", help="Path to purchase orders CSV (pos)")
    parser.add_argument("--model-dir", default="models", help="Directory to write model artifacts")
    parser.add_argument("--force", action="store_true", help="Force retrain even if artifacts exist")
    args = parser.parse_args()

    suppliers_path = args.suppliers
    pos_path = args.pos
    model_dir = args.model_dir

    print("Training with suppliers:", suppliers_path)
    print("            pos file:", pos_path)
    print("          model dir:", model_dir)

    if not os.path.exists(suppliers_path):
        print(f"ERROR: suppliers file not found: {suppliers_path}")
        raise SystemExit(2)
    if not os.path.exists(pos_path):
        print(f"ERROR: pos file not found: {pos_path}")
        raise SystemExit(2)

    try:
        res = train_models(suppliers_path, pos_path, model_dir, force_retrain=args.force)
    except Exception:
        print("Training failed with exception:")
        traceback.print_exc()
        raise SystemExit(3)

    print("Training finished.")
    print("Train R^2:", res.get("train_score"))
    print("Test R^2:", res.get("test_score"))
    print("Saved metadata keys:", list(res.get("metadata", {}).keys()))
