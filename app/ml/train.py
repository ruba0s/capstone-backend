import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from app.config import settings
from app.ml.features import clean_biomarker_data, build_features

def train():
    df = pd.read_csv(settings.DATA_PATH)
    df = clean_biomarker_data(df)

    X = build_features(df)
    y = df["age"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.2f} years")

    with open(settings.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train()