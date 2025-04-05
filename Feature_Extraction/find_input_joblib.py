import joblib
import cloudpickle

# Try loading the model
try:
    model = joblib.load("gbr_best_Y1_compat.joblib")
except Exception as e:
    print(f"joblib.load failed: {e}")
    print("Trying cloudpickle...")
    with open("gbr_best_Y1_compat.joblib", "rb") as f:
        model = cloudpickle.load(f)

# Extract and print input feature names
if hasattr(model, "feature_names_in_"):
    print("\nModel input features:")
    for feature in model.feature_names_in_:
        print(feature)
else:
    print("No feature_names_in_ attribute found in the model.")

# Optional: Print full params
# params = model.get_params() if hasattr(model, "get_params") else vars(model)
# for key, value in params.items():
#     print(f"{key}: {value}")
