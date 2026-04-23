# =============================================================================
# run_model_selection.py  — Stage 4
# =============================================================================
# Compare baseline vs tuned for each UC → save the winner as uc*_final.pkl
# → write outputs/model_selection_report.json.
#
# Run:  python run_model_selection.py
# Requires: run_training.py + run_tuning.py to have completed.
# =============================================================================

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import joblib
from src.modeling.model_selection import select_and_save_finals

# =============================================================================
# STEP 1 — Load splits
# =============================================================================
print("Loading splits...")
uc1 = joblib.load('splits/uc1.pkl')
uc3 = joblib.load('splits/uc3.pkl')
uc4 = joblib.load('splits/uc4.pkl')
uc2_cab  = joblib.load('splits/uc2_cab.pkl')
uc2_auto = joblib.load('splits/uc2_auto.pkl')
uc2_bike = joblib.load('splits/uc2_bike.pkl')

splits = {
    'uc1': uc1,
    'uc2': {'Cab': uc2_cab, 'Auto': uc2_auto, 'Bike': uc2_bike},
    'uc3': uc3,
    'uc4': uc4,
}

# =============================================================================
# STEP 2 — Compare and save finals
# =============================================================================
report = select_and_save_finals(
    splits,
    model_dir='models',
    output_dir='outputs',
)

# =============================================================================
# STEP 3 — Print report
# =============================================================================
print("\n" + "=" * 60)
print("MODEL SELECTION REPORT")
print("=" * 60)
for uc, info in report.items():
    metric  = info['metric']
    winner  = info['winner']
    b_score = info['baseline']
    t_score = info.get('tuned', 'N/A')
    print(f"\n{uc}")
    print(f"  Metric   : {metric}")
    print(f"  Baseline : {b_score}")
    print(f"  Tuned    : {t_score}")
    print(f"  Winner   : {winner}  ← saved as {uc.lower().replace('-', '_')}_final.pkl")

print("\n✅ Stage 4 complete.")
print("   Final models saved to models/")
print("   Report saved to outputs/model_selection_report.json")
print("\n   Run run_predict.py to test inference on a new booking.")
