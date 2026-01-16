import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
rf_market = joblib.load(os.path.join(BASE_DIR, "models", "rf_market.pkl"))
rf_perf = joblib.load(os.path.join(BASE_DIR, "models", "rf_performance.pkl"))
rf_match = joblib.load(os.path.join(BASE_DIR, "models", "rf_match.pkl"))

def predict_all(stats):
    stats = np.array(stats).reshape(1, -1)
    stats_scaled = scaler.transform(stats)

    market_value = rf_market.predict(stats_scaled)[0]
    performance = rf_perf.predict(stats_scaled)[0]
    match_win = rf_match.predict(stats_scaled)[0]

    return {
        "market_value": round(market_value, 2),
        "performance": round(performance, 2),
        "win_probability": "Alta" if match_win == 1 else "Baja"
    }
