# train_ml.py
"""
Train ML artifacts used by app.py.

Updated for your project:
 - Default POS file is now data/purchase_orders.csv
 - Robust path resolver so you can run from scripts/ or project root
 - Ensures project root is on sys.path so `models` can be imported
"""

import os
import sys
import argparse

# --- Ensure project root is on sys.path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../procurement_hub/scripts
PROJECT_ROOT = os.path.dirname(THIS_DIR)                # .../procurement_hub
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Helper to resolve paths relative to project root ---
def resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    candidate = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(p)

# --- Import Trainer ---
try:
    from models.ml_recommendation_engine import Trainer
except Exception as e:
    print("ERROR: Cannot import Trainer from models/ml_recommendation_engine.py")
    print(e)
    raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description="Train ML recommendation model")
    parser.add_argument("--suppliers", default="data/suppliers.csv",
                        help="Suppliers CSV (relative path allowed)")
    parser.add_argument("--pos", default="data/purchase_orders.csv",
                        help="Purchase orders CSV (relative path allowed)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to write trained model artifacts")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if artifacts exist")
    args = parser.parse_args()

    suppliers_path = resolve_path(args.suppliers)
    pos_path = resolve_path(args.pos)
    model_dir = resolve_path(args.model_dir)

    print("Training with suppliers:", suppliers_path)
    print("            pos file:", pos_path)
    print("          model dir:", model_dir)

    # Validate existence
    if not os.path.exists(suppliers_path):
        print(f"ERROR: suppliers file not found: {suppliers_path}")
        raise SystemExit(2)
    if not os.path.exists(pos_path):
        print(f"ERROR: purchase_orders file not found: {pos_path}")
        raise SystemExit(2)

    trainer = Trainer(model_dir)

    try:
        res = trainer.train_all_models(suppliers_path, pos_path, force_retrain=args.force)
    except Exception as e:
        print("Training failed:")
        import traceback
        traceback.print_exc()
        raise SystemExit(3)

    print("Training completed successfully.")
    print("Train R^2:", res.get("train_score"))
    print("Test R^2:", res.get("test_score"))
    print("Model metadata keys:", list(res.get("metadata", {}).keys()))

if __name__ == "__main__":
    main()
