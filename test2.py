import json
from predict import predict_dam_from_payload

PAYLOAD_PATH = "payload.json"

# =====================================================
# LOAD PAYLOAD
# =====================================================

with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
    payload = json.load(f)

print("=" * 80)
print("RUNNING DAM PREDICTION")
print("=" * 80)

# =====================================================
# PREDICT
# =====================================================

result = predict_dam_from_payload(payload)

# =====================================================
# SAVE
# =====================================================

OUTPUT = "dam_forecast.json"

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

# =====================================================
# SUMMARY
# =====================================================

print()
print("=" * 80)
print("PREDICTION COMPLETE")
print("=" * 80)

print("Region          :", result["region"])
print("Prediction Date :", result["prediction_date"])
print("Forecast Blocks :", len(result["forecast"]))

print()
print("First 5 Forecast Blocks")
print("-" * 80)

for row in result["forecast"][:5]:
    print(
        f"Block {row['block']:02d} | "
        f"P10={row['P10']:8.2f} | "
        f"P50={row['P50']:8.2f} | "
        f"P90={row['P90']:8.2f}"
    )

print()
print(f"Forecast saved to: {OUTPUT}")