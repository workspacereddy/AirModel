import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# 1. Page Setup
# ============================================================
st.set_page_config(page_title="Air Quality Forecast App", layout="wide")

st.title("🌫️ Air Quality Forecasting Dashboard")
st.write("Predict AQI for next **7 days, 7 weeks, or 1 month** using all trained models.")

# ============================================================
# 2. Load Data & Models
# ============================================================

@st.cache_resource
def load_models():
    model_paths = {
        "GradientBoosting": "GradientBoosting.pkl",
        "DecisionTree": "DecisionTree.pkl",
        "KNN": "KNN.pkl"
    }

    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
        except:
            st.warning(f"⚠️ Missing model file: {path}")

    scaler = joblib.load("scaler.joblib")

    return models, scaler


models, scaler = load_models()

# Load dataset
df = pd.read_csv("air_quality.csv")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
df = df.sort_values("date")


# ============================================================
# 3. Feature Engineering (same as your model training)
# ============================================================

def prepare_features(data):
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        data[f"lag_{lag}"] = data["aqi_clean"].shift(lag)

    data["roll_3"] = data["aqi_clean"].rolling(3).mean()
    data["roll_12"] = data["aqi_clean"].rolling(12).mean()
    data["roll_24"] = data["aqi_clean"].rolling(24).mean()
    data["roll_72"] = data["aqi_clean"].rolling(72).mean()

    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["weekday"] = data["date"].dt.weekday

    return data.dropna()


df["aqi_clean"] = df["aqi"]  # if needed
df = prepare_features(df)


# Feature list used for prediction
FEATURES = [
    "lag_1","lag_3","lag_6","lag_12","lag_24","lag_48","lag_72",
    "roll_3","roll_12","roll_24","roll_72",
    "hour","day","month","weekday"
]


# ============================================================
# 4. Forecast Function
# ============================================================

def forecast_future(model, hours):
    future = df.copy()

    for i in range(hours):
        X = future.iloc[-1][FEATURES].values.reshape(1, -1)
        pred = model.predict(X)[0]

        new_row = {
            "date": future.iloc[-1]["date"] + pd.Timedelta(hours=1),
            "aqi_clean": pred
        }

        for lag in [1, 3, 6, 12, 24, 48, 72]:
            new_row[f"lag_{lag}"] = future["aqi_clean"].iloc[-lag]

        for w in [3, 12, 24, 72]:
            new_row[f"roll_{w}"] = future["aqi_clean"].iloc[-w:].mean()

        new_row["hour"] = new_row["date"].hour
        new_row["day"] = new_row["date"].day
        new_row["month"] = new_row["date"].month
        new_row["weekday"] = new_row["date"].weekday()

        future = pd.concat([future, pd.DataFrame([new_row])], ignore_index=True)

    return future.tail(hours)[["date", "aqi_clean"]]


# ============================================================
# 5. User Input – Forecast Duration
# ============================================================

st.sidebar.header("⏳ Forecast Settings")
forecast_choice = st.sidebar.selectbox(
    "Select Prediction Time Range",
    ["Next 7 Days", "Next 7 Weeks", "Next 30 Days"]
)

if forecast_choice == "Next 7 Days":
    horizon = 7 * 24
elif forecast_choice == "Next 7 Weeks":
    horizon = 7 * 7 * 24
else:
    horizon = 30 * 24

st.sidebar.write(f"📌 Forecast Hours: **{horizon}**")

# ============================================================
# 6. Run Forecast Button
# ============================================================

if st.sidebar.button("🔮 Generate Predictions"):
    st.subheader(f"📈 Forecast Results for {forecast_choice}")

    tabs = st.tabs(list(models.keys()))  # Create tabs for each model

    for tab, (name, model) in zip(tabs, models.items()):
        with tab:
            st.write(f"### Model: {name}")

            pred_df = forecast_future(model, horizon)

            st.dataframe(pred_df.tail(20))

            # Plot chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pred_df["date"], pred_df["aqi_clean"], label="Predicted AQI")
            ax.set_title(f"{name} — Future AQI Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("AQI")
            ax.grid()
            st.pyplot(fig)

    st.success("✔ Forecast generation completed!")


# Footer
st.write("---")
st.write("Built with ❤️ using Streamlit and Machine Learning")
