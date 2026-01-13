# app.py
"""
Merged Flask app with improved recommendation scoring and several fixes:
 - ensures recommendations include overall_risk_level (templates expect it)
 - suppliers modal returns overall.match_score
 - purchase-orders view always supplies totals and statuses (no Jinja Undefined)
 - backward-compatible /api/update-po-status
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np

# -----------------------
# ML import (prefer models package)
# -----------------------
ML_MODULE_LOADED = False
try:
    from models.ml_recommendation_engine import (
        MLRecommendationEngine,
        safe_read_csv,
        load_embedding_engine,
        compute_overall_risk_level_from_row,
        combine_risk_score_normalized,
        EmbeddingEngine as ModelsEmbeddingEngine,
    )
    ML_MODULE_LOADED = True
except Exception:
    try:
        from ml_recommendation_engine import (
            MLRecommendationEngine,
            safe_read_csv,
            load_embedding_engine,
            compute_overall_risk_level_from_row,
            combine_risk_score_normalized,
            EmbeddingEngine as ModelsEmbeddingEngine,
        )
        ML_MODULE_LOADED = True
    except Exception:
        ML_MODULE_LOADED = False
        def safe_read_csv(path: str) -> pd.DataFrame:
            if path and os.path.exists(path):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()
        MLRecommendationEngine = None
        load_embedding_engine = None
        compute_overall_risk_level_from_row = None
        combine_risk_score_normalized = None
        ModelsEmbeddingEngine = None

# Also try to import a separate embedding engine if user created data/embedding_engine.py
DataEmbeddingEngine = None
try:
    from data.embedding_engine import EmbeddingEngine as DataEmbeddingEngine  # type: ignore
except Exception:
    DataEmbeddingEngine = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "change-me-in-prod")

# -----------------------
# Paths & config
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ml_models')
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)

SUPPLIERS_CSV = os.path.join(DATA_DIR, 'suppliers.csv')
PO_CSV = os.path.join(DATA_DIR, 'purchase_orders.csv')

# -----------------------
# Helpers: load/save CSVs
# -----------------------
def load_data():
    suppliers = safe_read_csv(SUPPLIERS_CSV)
    pos = safe_read_csv(PO_CSV)

    # normalize supplier_id
    if suppliers is not None and not suppliers.empty and 'supplier_id' in suppliers.columns:
        suppliers['supplier_id'] = suppliers['supplier_id'].astype(str)

    if pos is not None and not pos.empty and 'supplier_id' in pos.columns:
        pos['supplier_id'] = pos['supplier_id'].astype(str)

    # Cast numeric fields defensively
    sup_numeric_cols = ['total_orders', 'actual_on_time_rate', 'total_business_value', 'completed_orders',
                        'revenue_growth_yoy', 'net_profit_margin', 'response_time_hours', 'quality_score']
    for c in sup_numeric_cols:
        if suppliers is not None and c in suppliers.columns:
            suppliers[c] = pd.to_numeric(suppliers[c], errors='coerce').fillna(0)

    pos_numeric_cols = ['quantity', 'unit_price', 'subtotal', 'tax_amount', 'total_amount']
    for c in pos_numeric_cols:
        if pos is not None and c in pos.columns:
            pos[c] = pd.to_numeric(pos[c], errors='coerce').fillna(0)

    if suppliers is None:
        suppliers = pd.DataFrame()
    if pos is None:
        pos = pd.DataFrame()

    return suppliers, pos

suppliers_df, pos_df = load_data()

# -----------------------
# Embedding engine init
# -----------------------
embedding_engine = None
if ML_MODULE_LOADED and load_embedding_engine is not None:
    try:
        embedding_engine = load_embedding_engine()
        try:
            if suppliers_df is not None and not suppliers_df.empty:
                embedding_engine.ensure(suppliers_df)
        except Exception:
            pass
        print("Embedding engine loaded via models.ml_recommendation_engine.load_embedding_engine()")
    except Exception as e:
        print("load_embedding_engine() failure:", e)
        embedding_engine = None

if embedding_engine is None and DataEmbeddingEngine is not None:
    try:
        embedding_engine = DataEmbeddingEngine()
        if getattr(embedding_engine, 'enabled', lambda: True)() and suppliers_df is not None and not suppliers_df.empty:
            try:
                embedding_engine.ensure(suppliers_df)
            except Exception:
                pass
        print("Embedding engine loaded via data.embedding_engine.EmbeddingEngine")
    except Exception as e:
        print("data.embedding_engine instantiation error:", e)
        embedding_engine = None

if embedding_engine is None and ModelsEmbeddingEngine is not None:
    try:
        embedding_engine = ModelsEmbeddingEngine()
        if suppliers_df is not None and not suppliers_df.empty:
            try:
                embedding_engine.ensure(suppliers_df)
            except Exception:
                pass
        print("Embedding engine loaded via models.EmbeddingEngine")
    except Exception as e:
        print("ModelsEmbeddingEngine error:", e)
        embedding_engine = None

# -----------------------
# ML engine load
# -----------------------
ml_engine = None
if ML_MODULE_LOADED and MLRecommendationEngine is not None:
    try:
        if os.path.exists(MODEL_DIR):
            ml_engine = MLRecommendationEngine.load(MODEL_DIR, suppliers_df, pos_df)
            print("ML engine instantiated from", MODEL_DIR)
        else:
            ml_engine = MLRecommendationEngine(MODEL_DIR, suppliers_df, pos_df)
            print("ML engine instantiated (no artifacts present yet).")
    except Exception as e:
        print("ML engine init error:", e)
        ml_engine = None
else:
    print("ML module not available; ML features disabled.")

# -----------------------
# Utility functions
# -----------------------
def clean_record(d):
    out = {}
    for k, v in d.items():
        out[k] = None if pd.isna(v) else v
    return out

def generate_po_number(df: pd.DataFrame) -> str:
    if df is None or df.empty or 'po_number' not in df.columns:
        return "PO00001"
    nums = df['po_number'].astype(str).str.extract(r'PO(\d+)')[0].dropna()
    if nums.empty:
        return "PO00001"
    return f"PO{str(int(nums.astype(int).max()) + 1).zfill(5)}"

# -----------------------
# Risk helpers (kept from your code)
# -----------------------
def _norm(x, lo=0.0, hi=1.0):
    try:
        xv = float(x)
    except Exception:
        xv = 0.0
    return float(min(max((xv - lo) / (hi - lo + 1e-9), 0.0), 1.0))

def map_single_risk(v):
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
    if s in ("low", "l"):
        return "Low"
    if s in ("mid", "medium", "m"):
        return "Mid"
    if s in ("high", "h"):
        return "High"
    return "Mid"

def credit_rating_to_risk(credit):
    if credit is None or (isinstance(credit, float) and np.isnan(credit)):
        return 0.5
    try:
        s = str(credit).strip().upper()
    except Exception:
        return 0.5
    if s.startswith('AAA'):
        return 0.0
    if s.startswith('AA'):
        return 0.1
    if s.startswith('A'):
        return 0.2
    if s.startswith('BBB'):
        return 0.4
    try:
        n = float(s)
        return float(min(max(1.0 - (n / 850.0), 0.0), 1.0))
    except Exception:
        return 0.7

def risk_components(row):
    map_to_numeric = {'Low': 0.0, 'Mid': 0.5, 'High': 1.0}
    pr = map_single_risk(row.get('political_risk'))
    tr = map_single_risk(row.get('trade_risk'))
    nr = map_single_risk(row.get('natural_disaster_risk'))
    base_vals = [map_to_numeric.get(pr, 0.5), map_to_numeric.get(tr, 0.5), map_to_numeric.get(nr, 0.5)]
    base_risk = float(np.clip(np.mean(base_vals), 0.0, 1.0))

    credit_risk = credit_rating_to_risk(row.get('credit_rating'))

    try:
        completion = float(row.get('completion_rate') or row.get('actual_on_time_rate') or 0.0)
    except Exception:
        completion = 0.0
    try:
        on_time = float(row.get('on_time_rate') or row.get('actual_on_time_rate') or 0.0)
    except Exception:
        on_time = 0.0
    try:
        quality = float(row.get('quality_score') or 0.0)
    except Exception:
        quality = 0.0

    completion_n = _norm(completion, 0, 100)
    on_time_n = _norm(on_time, 0, 100)
    quality_n = _norm(quality, 0, 5)

    performance_score = (0.45 * completion_n) + (0.45 * on_time_n) + (0.10 * quality_n)
    operational_risk = float(np.clip(1.0 - performance_score, 0.0, 1.0))

    tbv = row.get('total_business_value') or 0.0
    try:
        tbv = float(tbv)
    except Exception:
        tbv = 0.0
    if tbv <= 0:
        size_factor = 0.15
    else:
        size_factor = float(max(0.0, 0.15 - min(0.14, np.log1p(tbv) / 50.0)))

    return {
        'base_risk': base_risk,
        'credit_risk': credit_risk,
        'operational_risk': operational_risk,
        'size_factor': size_factor
    }

def compute_risk_numeric(row):
    try:
        comps = risk_components(row)
    except Exception:
        mapped = map_single_risk(row.get('political_risk'))
        return float({'Low':0.0,'Mid':0.5,'High':1.0}.get(mapped,0.5))

    w_base = 0.50
    w_oper = 0.30
    w_credit = 0.15
    w_size = 0.05

    rn = (w_base * comps['base_risk']) + (w_oper * comps['operational_risk']) + (w_credit * comps['credit_risk']) + (w_size * comps['size_factor'])
    rn = float(np.clip(rn, 0.0, 1.0))
    return rn

def compute_overall_risk_level(row):
    try:
        rn = compute_risk_numeric(row)
    except Exception:
        mapped = map_single_risk(row.get("political_risk"))
        return mapped

    if rn < 0.33:
        return "Low"
    elif rn < 0.66:
        return "Mid"
    else:
        return "High"

# -----------------------
# Analytics data
# -----------------------
def analytics_data():
    """
    Analytics payload used by analytics.html.

    Returns a dict with top-level keys expected by templates:
      - stats: basic counters
      - risk_distribution: {low, medium, high}
      - monthly_pos: { labels, counts, amounts, data }
      - cycle_time: { labels, averages, median, data }
      - monthly_pos_totals: { total_count, total_amount }

    All values are native Python types (no numpy / pandas objects).
    """
    global pos_df, suppliers_df

    # defensive copies
    pos = pos_df.copy() if pos_df is not None else pd.DataFrame()
    suppliers = suppliers_df.copy() if suppliers_df is not None else pd.DataFrame()

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    stats = {}
    stats['total_pos'] = safe_int(len(pos))

    def safe_count(df_obj, col, value):
        try:
            return int((df_obj[col] == value).sum()) if (df_obj is not None and col in df_obj.columns) else 0
        except Exception:
            return 0

    stats['pending'] = safe_count(pos, 'status', 'Pending')
    stats['completed'] = safe_count(pos, 'status', 'Completed')
    stats['delivered'] = safe_count(pos, 'status', 'Delivered')
    stats['cancelled'] = safe_count(pos, 'status', 'Cancelled')

    # avg supplier rating
    try:
        if suppliers is not None and 'quality_score' in suppliers.columns and not suppliers['quality_score'].dropna().empty:
            avg_rating = float(pd.to_numeric(suppliers['quality_score'], errors='coerce').dropna().mean())
        else:
            avg_rating = 0.0
        stats['avg_rating'] = round(float(avg_rating or 0.0), 2)
    except Exception:
        stats['avg_rating'] = 0.0

    # cost savings
    try:
        if pos is not None and 'total_amount' in pos.columns and not pos['total_amount'].dropna().empty:
            total_amt = float(pd.to_numeric(pos['total_amount'], errors='coerce').fillna(0.0).sum())
            stats['cost_savings'] = round(total_amt * 0.05, 2)
        else:
            stats['cost_savings'] = 0.0
    except Exception:
        stats['cost_savings'] = 0.0

    # risk distribution (produce top-level value too)
    try:
        if suppliers is not None and not suppliers.empty and callable(compute_overall_risk_level):
            mapped = suppliers.apply(lambda row: compute_overall_risk_level(row), axis=1)
            risk_distribution = {
                'low': int((mapped == "Low").sum()),
                'medium': int((mapped == "Mid").sum()),
                'high': int((mapped == "High").sum())
            }
        else:
            risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
    except Exception:
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}

    stats['risk_distribution'] = risk_distribution  # keep inside stats for backward compatibility

    # --- monthly labels (last 12 months) ---
    try:
        now = pd.Timestamp.now()
    except Exception:
        from datetime import datetime
        now = pd.Timestamp(datetime.now())

    labels = []
    for i in range(11, -1, -1):
        try:
            per = (now - pd.DateOffset(months=i)).to_period('M')
            label = str(per)  # e.g. '2025-12'
        except Exception:
            label = str((now - pd.DateOffset(months=i)).strftime("%Y-%m"))
        labels.append(label)

    # prepare default lists
    counts = [0] * len(labels)
    amounts = [0.0] * len(labels)
    cycle_avg = [0.0] * len(labels)
    cycle_median = [0.0] * len(labels)

    if pos is not None and not pos.empty:
        pos_local = pos.copy()

        # parse dates defensively
        for col in ('order_date', 'expected_delivery_date', 'actual_delivery_date'):
            if col in pos_local.columns:
                try:
                    pos_local[f'__{col}'] = pd.to_datetime(pos_local[col], errors='coerce')
                except Exception:
                    pos_local[f'__{col}'] = pd.NaT
            else:
                pos_local[f'__{col}'] = pd.NaT

        try:
            pos_local['__period'] = pos_local['__order_date'].dt.to_period('M').astype(str)
        except Exception:
            pos_local['__period'] = None

        # counts and amounts
        try:
            grp = pos_local.groupby('__period', dropna=True)
            counts_map = {k: int(v) for k, v in grp.size().to_dict().items()}
            if 'total_amount' in pos_local.columns:
                pos_local['__amt'] = pd.to_numeric(pos_local['total_amount'], errors='coerce').fillna(0.0)
                amt_map = {k: float(v) for k, v in grp['__amt'].sum().to_dict().items()}
            else:
                amt_map = {}
        except Exception:
            counts_map = {}
            amt_map = {}

        # cycle times (expected -> actual)
        try:
            valid = pos_local.dropna(subset=['__expected_delivery_date', '__actual_delivery_date']).copy()
            if not valid.empty:
                valid['__cycle_days'] = (valid['__actual_delivery_date'] - valid['__expected_delivery_date']).dt.days.astype(float)
                cg = valid.groupby('__period')['__cycle_days']
                mean_map = {k: float(v.mean()) for k, v in cg}
                median_map = {k: float(v.median()) for k, v in cg}
            else:
                mean_map = {}
                median_map = {}
        except Exception:
            mean_map = {}
            median_map = {}

        # fill arrays based on labels
        for idx, lab in enumerate(labels):
            counts[idx] = int(counts_map.get(lab, 0))
            amounts[idx] = float(amt_map.get(lab, 0.0) or 0.0)
            cycle_avg[idx] = round(float(mean_map.get(lab, 0.0) or 0.0), 2)
            cycle_median[idx] = round(float(median_map.get(lab, 0.0) or 0.0), 2)

    monthly_pos = {
        'labels': labels,
        'counts': counts,
        'amounts': [round(a, 2) for a in amounts],
        'data': counts  # backward compatibility alias
    }

    cycle_time = {
        'labels': labels,
        'averages': cycle_avg,
        'median': cycle_median,
        'data': cycle_avg  # alias used by template
    }

    monthly_pos_totals = {
        'total_count': int(sum(counts)),
        'total_amount': round(sum(amounts), 2)
    }

    # top-level payload: include both stats and commonly-used top-level keys
    payload = {
        'stats': stats,
        'risk_distribution': risk_distribution,
        'monthly_pos': monthly_pos,
        'cycle_time': cycle_time,
        'monthly_pos_totals': monthly_pos_totals
    }

    return payload



# -----------------------
# Routes
# -----------------------
@app.route('/')
def index():
    data = analytics_data()
    top_suppliers = []
    try:
        if ml_engine:
            top_suppliers = ml_engine.recommend_suppliers(top_n=5, semantic_scores=None, alpha_ml=1.0, alpha_semantic=0.0)
    except Exception:
        top_suppliers = []
    return render_template('dashboard.html', stats=data['stats'], top_suppliers=top_suppliers)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', data=analytics_data())

# -----------------------
# Hybrid RECOMMEND (ML + embeddings)
# -----------------------
@app.route('/api/recommend')
def api_recommend():
    global suppliers_df, ml_engine, embedding_engine, pos_df

    query = request.args.get('query') or request.args.get('search') or ''
    industry = request.args.get('industry', '').strip()
    product = request.args.get('product', '').strip()
    try:
        top_n = int(request.args.get('top_n', 50))
    except Exception:
        top_n = 50

    df = suppliers_df.copy() if suppliers_df is not None else pd.DataFrame()
    if df is None or df.empty or 'supplier_id' not in df.columns:
        return jsonify({"success": True, "recommendations": []})

    df['supplier_id'] = df['supplier_id'].astype(str)
    qlower = (query or "").lower()

    electronics_keywords = [
        'monitor', 'display', 'screen', 'lcd', 'led', 'oled',
        'laptop', 'tablet', 'phone', 'camera', 'sensor', 'chip',
        'component', 'electronic', 'electronics'
    ]
    is_product_query = any(k in qlower for k in electronics_keywords)

    # ---- semantic (with fallback) ----
    semantic_map = {}
    try:
        if query.strip() and embedding_engine is not None and getattr(embedding_engine, "enabled", lambda: True)():
            try:
                embedding_engine.ensure(df)
            except Exception:
                pass
            try:
                sem = embedding_engine.search(query, top_k=len(df))
            except Exception as e:
                print("embedding_engine.search error:", e)
                sem = []
            try:
                semantic_map = {str(sid): float(score) for sid, score in sem}
            except Exception:
                semantic_map = {}
                for it in sem:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        semantic_map[str(it[0])] = float(it[1])

        use_fallback = False
        if not semantic_map:
            use_fallback = True
        else:
            try:
                vals = [float(v) for v in semantic_map.values()]
                if max(vals) == 0 or (max(vals) - min(vals) < 1e-12):
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            qtokens = [t for t in qlower.split() if t.strip()]
            fallback = {}
            for _, row in df.iterrows():
                sid = str(row.get('supplier_id', ''))
                text = ""
                if 'products' in row and pd.notna(row['products']):
                    text += " " + str(row['products'])
                text += " " + str(row.get('name', ''))
                text += " " + str(row.get('industry', ''))
                text_l = text.lower()
                hits = sum(1 for tok in qtokens if tok in text_l)
                score = hits / max(1, len(qtokens))
                fallback[sid] = float(score)
            fv = np.array(list(fallback.values())) if fallback else np.array([0.0])
            mn, mx = float(fv.min()), float(fv.max())
            if mx > mn:
                for k in fallback:
                    fallback[k] = float((fallback[k] - mn) / (mx - mn))
            else:
                for k in fallback:
                    fallback[k] = 1.0 if fallback[k] > 0 else 0.0
            semantic_map = fallback

        # final normalize
        if semantic_map:
            vals = np.array(list(semantic_map.values()))
            mn, mx = float(vals.min()), float(vals.max())
            if mx > mn:
                for k in semantic_map:
                    semantic_map[k] = float((semantic_map[k] - mn) / (mx - mn))
            else:
                for k in semantic_map:
                    semantic_map[k] = 1.0 if semantic_map[k] > 0 else 0.0

    except Exception as e:
        print("Semantic processing error:", e)
        semantic_map = {}

    df['semantic_score'] = df['supplier_id'].map(semantic_map).fillna(0.0)

    # ---- ML predictions ----
    df['ml_score'] = 0.0
    df['risk_numeric'] = df.apply(compute_risk_numeric, axis=1)

    if ml_engine:
        try:
            preds = ml_engine.predict_all()
            if preds is not None and not preds.empty and 'supplier_id' in preds.columns:
                preds['supplier_id'] = preds['supplier_id'].astype(str)
                if 'final_recommendation_score' in preds.columns:
                    raw_ml = preds['final_recommendation_score'].astype(float)
                elif 'ml_score' in preds.columns:
                    raw_ml = preds['ml_score'].astype(float)
                else:
                    num_cols = preds.select_dtypes(include=[np.number]).columns.tolist()
                    raw_ml = preds[num_cols[0]].astype(float) if num_cols else pd.Series([0.0]*len(preds))

                mean = raw_ml.mean()
                std = raw_ml.std() if raw_ml.std() > 1e-12 else 1.0
                z = (raw_ml - mean) / (std + 1e-9)
                z_min, z_max = float(z.min()), float(z.max())
                if (z_max - z_min) > 1e-12:
                    ml_norm = (z - z_min) / (z_max - z_min)
                else:
                    ml_norm = (raw_ml - raw_ml.min()) / (raw_ml.max() - raw_ml.min() + 1e-9)

                ml_map = dict(zip(preds['supplier_id'], ml_norm))
                df['ml_score'] = df['supplier_id'].map(ml_map).fillna(0.0)

                if 'risk_numeric' in preds.columns:
                    try:
                        risk_map = dict(zip(preds['supplier_id'], preds['risk_numeric'].astype(float)))
                        df['risk_numeric'] = df['supplier_id'].map(risk_map).fillna(df['risk_numeric'])
                    except Exception:
                        pass
        except Exception as e:
            print("ML prediction error:", e)

    # ---- product filtering & boosts ----
    if is_product_query:
        if 'industry' in df.columns:
            mask_elec = df['industry'].astype(str).str.lower() == 'electronics'
            if mask_elec.any():
                df = df[mask_elec].copy()
            else:
                if df['semantic_score'].max() > 0:
                    df = df[df['semantic_score'] > 0].copy()

        df['semantic_score'] = (df['semantic_score'] * 2.5).clip(0, 1.0)
        df['ml_score'] = (df['ml_score'] * 1.3).clip(0, 1.0)

        RISK_PENALTY = 0.25
        ML_WEIGHT = 0.05
        SEM_WEIGHT = 0.9
    else:
        ML_WEIGHT = 0.3
        SEM_WEIGHT = 0.65
        RISK_PENALTY = 0.15

    if industry:
        if 'industry' in df.columns:
            df = df[df['industry'].astype(str) == industry]

    if product and 'products' in df.columns:
        mask = df['products'].astype(str).str.contains(product, case=False, na=False) | (df['semantic_score'] > 0.10)
        df = df[mask]

    # ---- final scoring ----
    df['final_score'] = (ML_WEIGHT * df['ml_score'] + SEM_WEIGHT * df['semantic_score'] - RISK_PENALTY * df['risk_numeric']).clip(0, 1)

    try:
        if (df['final_score'].max() - df['final_score'].min()) < 1e-8:
            df['final_score'] = df['ml_score']
    except Exception:
        pass

    df = df.sort_values('final_score', ascending=False).head(top_n)

    # ---- build response ----
    results = []
    for _, r in df.iterrows():
        # compute a canonical overall risk level for this row (string: Low/Mid/High)
        try:
            overall_risk = compute_overall_risk_level(r)
        except Exception:
            overall_risk = map_single_risk(r.get('political_risk'))

        match_score = int(round(float(r.get('final_score') or 0.0) * 100))
        # ensure savings & tbv are numbers
        tbv = float(r.get('total_business_value') or 0.0)
        savings = float(tbv * 0.05)

        results.append({
            "supplier_id": r.get('supplier_id'),
            "name": r.get('name') or "",
            "industry": r.get('industry') or "",
            "city": r.get('city') or "",
            "country": r.get('country') or "",
            "semantic_score": float(r.get('semantic_score') or 0.0),
            "ml_score": float(r.get('ml_score') or 0.0),
            "match_score": match_score,
            "overall_risk_level": overall_risk,           # <-- template expects this key
            "risk_level": overall_risk,                  # keep legacy key too
            "hybrid_score": float(r.get('final_score') or 0.0),
            "savings_potential": savings,
            "total_business_value": tbv
        })

    return jsonify({"success": True, "recommendations": results})

# ---------------------------------------------------------
# SUPPLIERS PAGE
# ---------------------------------------------------------
@app.route('/suppliers')
def suppliers():
    q = request.args.get("search", "")
    industry = request.args.get("industry", "")
    product = request.args.get("product", "")

    # internal recommend call
    with app.test_request_context(f"/api/recommend?query={q}&industry={industry}&product={product}&top_n=300"):
        rec_json = api_recommend().get_json()

    industries = sorted(suppliers_df['industry'].dropna().unique()) if 'industry' in suppliers_df.columns else []
    product_list = []
    if 'products' in suppliers_df.columns:
        product_list = (suppliers_df['products'].dropna().astype(str).str.split(',').explode().str.strip().unique().tolist())

    return render_template(
        "suppliers.html",
        recommendations=rec_json.get('recommendations', []),
        industries=industries,
        products=product_list,
        search_query=q,
        selected_industry=industry,
        selected_product=product
    )

# ---------------------------------------------------------
# PURCHASE ORDERS
# ---------------------------------------------------------
@app.route('/purchase-orders')
def purchase_orders():
    page = int(request.args.get('page', 1))
    per_page = 50
    status = request.args.get('status')
    supplier = request.args.get('supplier')
    search = request.args.get('search')

    df = pos_df.copy() if pos_df is not None else pd.DataFrame()
    if df is None:
        df = pd.DataFrame()

    if status and 'status' in df.columns:
        df = df[df['status'] == status]

    if supplier and 'supplier_id' in df.columns:
        df = df[df['supplier_id'] == supplier]

    if search:
        q = str(search).lower()
        cond = pd.Series(False, index=df.index)
        if 'po_number' in df.columns:
            cond = cond | df['po_number'].astype(str).str.lower().str.contains(q, na=False)
        if 'supplier_name' in df.columns:
            cond = cond | df['supplier_name'].astype(str).str.lower().str.contains(q, na=False)
        if 'item_description' in df.columns:
            cond = cond | df['item_description'].astype(str).str.lower().str.contains(q, na=False)
        df = df[cond]

    total_filtered = int(len(df))
    total_all = int(len(pos_df) if pos_df is not None else 0)

    def safe_count(df_obj, col, value):
        try:
            return int((df_obj[col] == value).sum()) if col in df_obj.columns else 0
        except Exception:
            return 0

    pending_count = safe_count(pos_df if pos_df is not None else pd.DataFrame(), 'status', 'Pending')
    completed_count = safe_count(pos_df if pos_df is not None else pd.DataFrame(), 'status', 'Completed')
    delivered_count = safe_count(pos_df if pos_df is not None else pd.DataFrame(), 'status', 'Delivered')
    cancelled_count = safe_count(pos_df if pos_df is not None else pd.DataFrame(), 'status', 'Cancelled')

    # statuses list for filter dropdown
    statuses = []
    try:
        if pos_df is not None and 'status' in pos_df.columns:
            statuses = sorted([s for s in pos_df['status'].dropna().unique().tolist()])
    except Exception:
        statuses = ['Pending', 'Completed', 'Delivered', 'Cancelled']

    if 'order_date' in df.columns:
        try:
            if not np.issubdtype(df['order_date'].dtype, np.datetime64):
                df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df = df.sort_values('order_date', ascending=False, na_position='last')
        except Exception:
            pass

    start = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page] if not df.empty else df
    pos_list = [clean_record(r) for _, r in page_df.iterrows()]

    return render_template(
        "purchase_orders.html",
        pos=pos_list,
        page=page,
        total_pages=(total_filtered + per_page - 1) // per_page if total_filtered is not None else 0,
        selected_status=status,
        search_query=search,
        total_pos=total_all,
        pending_count=int(pending_count),
        completed_count=int(completed_count),
        delivered_count=int(delivered_count),
        cancelled_count=int(cancelled_count),
        statuses=statuses
    )

# API: update purchase order status (POST JSON) - keeps old path
# Replace your current api_update_po_status with this function
@app.route('/api/update-po-status', methods=['POST'])
@app.route('/api/po/<po_number>/status', methods=['POST'])
def api_update_po_status(po_number: str = None):
    """
    Robust PO status updater. Accepts either:
      - POST /api/update-po-status  with JSON { "po_number": "...", "status": "Completed", "ratings": {...} }
      - POST /api/po/<po_number>/status with JSON { "status": "Completed", "ratings": {...} }
    Returns detailed errors during development to help debug 'internal_error'.
    """
    global pos_df, PO_CSV, suppliers_df

    try:
        # parse JSON safely
        payload = None
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {}

        # Determine po_number (path param preferred; fallback to payload)
        if po_number:
            po_number = str(po_number).strip()
        else:
            po_number = str(payload.get('po_number') or payload.get('poNumber') or "").strip()

        new_status = str(payload.get('status') or payload.get('new_status') or "").strip()
        ratings = payload.get('ratings')

        if not po_number:
            return jsonify({"ok": False, "error": "missing_po_number", "message": "po_number is required"}), 400
        if not new_status:
            return jsonify({"ok": False, "error": "missing_status", "message": "status is required"}), 400

        # Defensive checks
        if pos_df is None or pos_df.empty or 'po_number' not in pos_df.columns:
            return jsonify({"ok": False, "error": "no_pos_data", "message": "Purchase order data not loaded"}), 500

        mask = pos_df['po_number'].astype(str) == po_number
        if not mask.any():
            return jsonify({"ok": False, "error": "po_not_found", "message": f"PO {po_number} not found"}), 404

        # Ensure status column exists
        if 'status' not in pos_df.columns:
            pos_df['status'] = ''

        # Normalize status string
        normalized_status = new_status.strip().title()

        # Update in-memory DF
        try:
            pos_df.loc[mask, 'status'] = normalized_status

            # Ensure actual_delivery_date column exists
            if 'actual_delivery_date' not in pos_df.columns:
                pos_df['actual_delivery_date'] = ''

            # If marking completed/delivered, set actual_delivery_date if empty
            if normalized_status.lower() in ('delivered', 'completed'):
                now = datetime.now().strftime("%Y-%m-%d")
                # only set when blank or NaN
                try:
                    cur_vals = pos_df.loc[mask, 'actual_delivery_date']
                    # if all empty or any empty, set to now
                    pos_df.loc[mask, 'actual_delivery_date'] = cur_vals.fillna('').replace('', now)
                except Exception:
                    pos_df.loc[mask, 'actual_delivery_date'] = now

        except Exception as e:
            # return a clear error about the update step
            import traceback as _tb
            tb = _tb.format_exc()
            print("Error updating dataframe:", tb)
            return jsonify({"ok": False, "error": "update_failed", "message": f"failed to update dataframe: {str(e)}", "error_trace": tb}), 500

        # If ratings are present attempt to update supplier aggregates (best-effort)
        if isinstance(ratings, dict):
            try:
                sid = str(pos_df.loc[mask].iloc[0].get('supplier_id') or "")
                if sid and suppliers_df is not None and not suppliers_df.empty and 'supplier_id' in suppliers_df.columns:
                    s_mask = suppliers_df['supplier_id'].astype(str) == sid
                    if s_mask.any():
                        sidx = suppliers_df[s_mask].index[0]
                        # increment total_orders
                        try:
                            old_total = int(suppliers_df.at[sidx].get('total_orders') or 0)
                            suppliers_df.at[sidx, 'total_orders'] = old_total + 1
                        except Exception:
                            suppliers_df.at[sidx, 'total_orders'] = suppliers_df.at[sidx].get('total_orders') or 1

                        # update rolling quality_score if provided
                        if 'quality_score' in ratings:
                            try:
                                new_q = float(ratings.get('quality_score', 0))
                                old_q = float(suppliers_df.at[sidx].get('quality_score') or 0)
                                prev_count = max(int(suppliers_df.at[sidx].get('total_orders') or 1) - 1, 0)
                                if prev_count <= 0:
                                    suppliers_df.at[sidx, 'quality_score'] = new_q
                                else:
                                    suppliers_df.at[sidx, 'quality_score'] = ((old_q * prev_count) + new_q) / (prev_count + 1)
                            except Exception:
                                pass
                # persist suppliers DF (best-effort)
                try:
                    suppliers_df.to_csv(SUPPLIERS_CSV, index=False)
                except Exception as e:
                    print("Warning: failed to persist suppliers.csv:", str(e))
            except Exception:
                # don't break the main flow
                import traceback as _tb
                print("Warning while applying ratings to supplier:", _tb.format_exc())

        # persist pos_df safely (atomic replace)
        try:
            temp_path = PO_CSV + ".tmp"
            pos_df.to_csv(temp_path, index=False)
            os.replace(temp_path, PO_CSV)
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            print("Failed to persist PO CSV:", tb)
            return jsonify({"ok": False, "error": "save_failed", "message": f"failed to save PO CSV: {str(e)}", "error_trace": tb}), 500

        updated_row = clean_record(pos_df.loc[mask].iloc[0].to_dict())
        return jsonify({"ok": True, "po": updated_row})

    except Exception as e:
        # Development-friendly response: print & return traceback
        import traceback as _tb
        tb = _tb.format_exc()
        print("api_update_po_status: unexpected error:", tb)
        return jsonify({"ok": False, "error": "internal_error", "message": "Unexpected server error", "error_trace": tb}), 500


# ---------------------------------------------------------
# CREATE PO
# ---------------------------------------------------------
@app.route('/create-po', methods=['GET', 'POST'])
def create_po():
    global suppliers_df, pos_df

    if request.method == 'POST':
        data = request.form.to_dict()
        new_po_num = generate_po_number(pos_df)
        try:
            qty = float(data.get('quantity', 0))
            price = float(data.get('unit_price', 0))
        except Exception:
            qty = 0.0
            price = 0.0
        subtotal = qty * price
        tax_rate = float(data.get('tax_rate', 0)) if data.get('tax_rate') else 0.0
        tax = subtotal * tax_rate
        total = subtotal + tax

        new_po = {
            "po_number": new_po_num,
            "supplier_id": data.get('supplier_id'),
            "supplier_name": data.get('supplier_name', ""),
            "order_date": datetime.now().strftime("%Y-%m-%d"),
            "expected_delivery_date": data.get('delivery_date', ""),
            "actual_delivery_date": "",
            "status": "Pending",
            "item_description": data.get('item_description', ""),
            "quantity": qty,
            "unit_price": price,
            "subtotal": subtotal,
            "tax_rate": tax_rate,
            "tax_amount": tax,
            "total_amount": total,
            "currency": data.get('currency', 'INR')
        }

        pos_df = pd.concat([pos_df, pd.DataFrame([new_po])], ignore_index=True)
        pos_df.to_csv(PO_CSV, index=False)

        return redirect(url_for('po_details', po_number=new_po_num))

    supplier_id = request.args.get("supplier_id")
    supplier = None
    if supplier_id and 'supplier_id' in suppliers_df.columns:
        row = suppliers_df[suppliers_df['supplier_id'] == supplier_id]
        if not row.empty:
            supplier = clean_record(row.iloc[0].to_dict())

    return render_template("create_po.html", supplier=supplier, suppliers=suppliers_df.to_dict("records"))

# ---------------------------------------------------------
# PO DETAILS
# ---------------------------------------------------------
@app.route('/po/<po_number>')
def po_details(po_number):
    global suppliers_df, pos_df
    row = pos_df[pos_df.get('po_number', pd.Series()).astype(str) == str(po_number)]
    if row.empty:
        return "PO not found", 404
    po = clean_record(row.iloc[0].to_dict())
    s = suppliers_df[suppliers_df.get('supplier_id', pd.Series()).astype(str) == po.get('supplier_id')]
    supplier = clean_record(s.iloc[0].to_dict()) if not s.empty else None
    return render_template("po_details.html", po=po, supplier=supplier)

# ---------------------------------------------------------
# API - Supplier Modal (robust)
# ---------------------------------------------------------
@app.route('/api/supplier/<supplier_id>')
def api_supplier(supplier_id):
    supplier_id = str(supplier_id)
    if suppliers_df is None or suppliers_df.empty:
        return jsonify({"error": "No suppliers loaded"}), 404

    s = suppliers_df[suppliers_df.get('supplier_id', pd.Series()).astype(str) == supplier_id]
    if s.empty:
        return jsonify({"error": "Supplier not found"}), 404

    row = s.iloc[0].to_dict()

    basic_info = {
        "supplier_id": supplier_id,
        "name": row.get("name") or "",
        "industry": row.get("industry") or "",
        "city": row.get("city") or "",
        "country": row.get("country") or "",
        "address": row.get("address") or "",
        "postal_code": row.get("postal_code") or "",
        "email": row.get("email") or "",
        "phone": row.get("phone") or "",
        "years_in_business": int(row.get("years_in_business") or row.get("years_in_service") or 0),
        "certification": row.get("certification") or row.get("certifications") or "",
        "employee_count": int(row.get("employee_count") or row.get("employees") or 0),
    }

    # match score logic: prefer stored, else blend ML+semantic
    match_score = None
    try:
        if 'match_score' in row and row.get('match_score') not in (None, "", 0):
            match_score = int(float(row.get('match_score') or 0))
    except Exception:
        match_score = None

    ml_score = 0.0
    sem_score = 0.0
    try:
        if ml_engine:
            preds = ml_engine.predict_all()
            if preds is not None and not preds.empty and 'supplier_id' in preds.columns:
                preds['supplier_id'] = preds['supplier_id'].astype(str)
                if 'final_recommendation_score' in preds.columns:
                    raw = preds.set_index('supplier_id')['final_recommendation_score'].astype(float)
                elif 'ml_score' in preds.columns:
                    raw = preds.set_index('supplier_id')['ml_score'].astype(float)
                else:
                    num_cols = preds.select_dtypes(include=[np.number]).columns.tolist()
                    raw = preds.set_index('supplier_id')[num_cols[0]].astype(float) if num_cols else None

                if raw is not None and len(raw) > 0:
                    try:
                        vals = raw.values.astype(float)
                        mn, mx = float(vals.min()), float(vals.max())
                        if mx > mn:
                            norm_series = (raw - mn) / (mx - mn)
                        else:
                            norm_series = raw.fillna(0.0)
                        ml_score = float(norm_series.get(supplier_id, 0.0))
                    except Exception:
                        ml_score = float(raw.get(supplier_id, 0.0) if supplier_id in raw.index else 0.0)
    except Exception:
        ml_score = 0.0

    try:
        if embedding_engine:
            try:
                embedding_engine.ensure(suppliers_df)
            except Exception:
                pass
            qtext = basic_info['name'] or basic_info['industry'] or basic_info['city'] or ""
            if qtext:
                try:
                    sem = embedding_engine.search(qtext, top_k=10)
                    if isinstance(sem, dict):
                        sem_score = float(sem.get(supplier_id, 0.0))
                    elif isinstance(sem, (list, tuple)):
                        for sid, sc in sem:
                            if str(sid) == supplier_id:
                                sem_score = float(sc)
                                break
                except Exception:
                    sem_score = 0.0
    except Exception:
        sem_score = 0.0

    if match_score is None or match_score == 0:
        try:
            blended = (0.7 * float(ml_score or 0.0)) + (0.3 * float(sem_score or 0.0))
            match_score = int(round(max(0.0, min(blended, 1.0)) * 100))
        except Exception:
            match_score = 0

    if match_score == 0 and ml_score > 0:
        match_score = int(round(ml_score * 100))

    try:
        revenue_growth = float(row.get("revenue_growth_yoy") or 0.0)
    except Exception:
        revenue_growth = 0.0
    try:
        net_profit_margin = float(row.get("net_profit_margin") or 0.0)
    except Exception:
        net_profit_margin = 0.0
    credit = row.get("credit_rating") or ""

    try:
        rg_score = max(min(revenue_growth / 50.0, 1.0), -1.0)
        rg_score = (rg_score + 1.0) / 2.0
        pm_score = max(min(net_profit_margin / 50.0, 1.0), -1.0)
        pm_score = (pm_score + 1.0) / 2.0
        credit_bonus = 0.0
        if isinstance(credit, str):
            cred = credit.strip().upper()
            if cred.startswith('AAA'):
                credit_bonus = 0.20
            elif cred.startswith('AA'):
                credit_bonus = 0.15
            elif cred.startswith('A'):
                credit_bonus = 0.10
            elif cred.startswith('BBB'):
                credit_bonus = 0.05
            else:
                credit_bonus = 0.0
        fin_score = (0.5 * rg_score) + (0.4 * pm_score) + credit_bonus
        fin_score = int(round(max(0.0, min(fin_score, 1.0)) * 100))
    except Exception:
        fin_score = 0

    try:
        on_time = float(row.get("on_time_rate") or row.get("actual_on_time_rate") or 0.0)
    except Exception:
        on_time = 0.0
    try:
        completion = float(row.get("completion_rate") or 0.0)
    except Exception:
        completion = 0.0
    try:
        quality = float(row.get("quality_score") or 0.0)
    except Exception:
        quality = 0.0

    try:
        on_time_n = max(min(on_time / 100.0, 1.0), 0.0)
        completion_n = max(min(completion / 100.0, 1.0), 0.0)
        quality_n = max(min(quality / 5.0, 1.0), 0.0)
        perf_score = (0.5 * on_time_n) + (0.3 * completion_n) + (0.2 * quality_n)
        perf_score = int(round(max(0.0, min(perf_score, 1.0)) * 100))
    except Exception:
        perf_score = 0

    try:
        risk_level = compute_overall_risk_level(row) if callable(compute_overall_risk_level) else None
    except Exception:
        risk_level = None
    if not risk_level:
        risk_level = map_single_risk(row.get("political_risk"))

    pos = pos_df[pos_df.get('supplier_id', pd.Series()).astype(str) == supplier_id] if pos_df is not None else pd.DataFrame()
    total_orders = int(row.get("total_orders") or (len(pos) if pos is not None else 0))
    completed_orders = int(row.get("completed_orders") or (pos.get('status', pd.Series()).eq('Completed').sum() if not pos.empty and 'status' in pos.columns else 0))
    total_business_value = float(row.get("total_business_value") or (pos.get('total_amount', pd.Series()).sum() if not pos.empty and 'total_amount' in pos.columns else 0.0))

    response = {
        "basic_info": basic_info,
        "financial_health": {
            "credit_rating": row.get("credit_rating") or "",
            "revenue_growth_yoy": round(revenue_growth, 2),
            "net_profit_margin": round(net_profit_margin, 2),
            "financial_score": fin_score
        },
        "performance_metrics": {
            "on_time_delivery_rate": round(on_time, 1),
            "quality_score": round(quality, 2),
            "response_time_hours": float(row.get("response_time_hours") or 0.0),
            "performance_score": perf_score
        },
        "risk_profile": {
            "political_risk": map_single_risk(row.get("political_risk")),
            "trade_risk": map_single_risk(row.get("trade_risk")),
            "natural_disaster_risk": map_single_risk(row.get("natural_disaster_risk")),
            "overall_risk": risk_level
        },
        "business_history": {
            "total_orders": total_orders,
            "completed_orders": int(completed_orders),
            "total_business_value": float(total_business_value)
        },
        "overall": {
            "savings_potential": float((row.get("total_business_value") or total_business_value or 0) * 0.05),
            "match_score": int(match_score or 0)
        }
    }

    return jsonify(response)

# ---------------------------------------------------------
# Supplier Details Page
# ---------------------------------------------------------
@app.route('/supplier-details/<supplier_id>')
def supplier_details(supplier_id):
    supplier_id = str(supplier_id)
    row = suppliers_df[suppliers_df.get('supplier_id', pd.Series()).astype(str) == supplier_id]
    if row.empty:
        return "Supplier not found", 404
    supplier = clean_record(row.iloc[0].to_dict())
    return render_template("supplier_details.html", supplier=supplier)

@app.route('/api/recommend_debug')
def api_recommend_debug():
    q = (request.args.get('query') or request.args.get('search') or "").strip()
    df = suppliers_df.copy() if suppliers_df is not None else pd.DataFrame()
    if 'supplier_id' not in df.columns:
        return jsonify({"error": "no suppliers loaded"}), 400

    sem_map = {}
    if q and embedding_engine:
        try:
            embedding_engine.ensure(df)
            sem = embedding_engine.search(q, top_k=len(df))
            try:
                sem_map = {str(sid): float(score) for sid, score in sem}
            except Exception:
                sem_map = {}
                for it in sem:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        sem_map[str(it[0])] = float(it[1])
        except Exception as e:
            sem_map = {"error": str(e)}

    ml_raw = {}
    ml_stats = {}
    if ml_engine:
        try:
            preds = ml_engine.predict_all()
            if preds is not None and not preds.empty:
                score_col = 'final_recommendation_score' if 'final_recommendation_score' in preds.columns else ('ml_score' if 'ml_score' in preds.columns else None)
                if score_col:
                    for _, row in preds.iterrows():
                        ml_raw[str(row['supplier_id'])] = float(row.get(score_col, 0.0))
                    import numpy as _np
                    vals = _np.array(list(ml_raw.values()), dtype=float)
                    if vals.size:
                        ml_stats = {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()), "std": float(vals.std())}
        except Exception as e:
            ml_stats = {"error": str(e)}

    return jsonify({
        "query": q,
        "semantic_map_sample": dict(list(sem_map.items())[:20]),
        "semantic_max": (max(sem_map.values()) if sem_map and all(isinstance(v, (int,float)) for v in sem_map.values()) else None),
        "ml_raw_sample": dict(list(ml_raw.items())[:20]),
        "ml_stats": ml_stats
    })

# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ml_engine_available": bool(ml_engine),
        "embedding_engine_available": bool(embedding_engine),
        "suppliers_loaded": int(not suppliers_df.empty)
    })

# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Starting app on 0.0.0.0:5000")
    print("MODEL_DIR:", MODEL_DIR)
    print("ML engine available:", bool(ml_engine))
    print("Embedding engine available:", bool(embedding_engine))
    print("Suppliers loaded:", not suppliers_df.empty)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
