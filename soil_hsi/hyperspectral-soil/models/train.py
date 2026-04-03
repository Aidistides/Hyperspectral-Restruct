import argparse
import numpy as np
import joblib

from hyperspectral_soil.deep_learning import SpectralCNN, DLTrainer

from sklearn.model_selection import train_test_split, cross_val_score

from hyperspectral_soil.models import (
    PLSModel,
    RandomForestModel,
    XGBoostModel,
    NeuralNetModel,
    Trainer
)

# ------------------------
# Dummy data loader (replace later)
# ------------------------
def load_data():
    # Replace with real dataset
    X = np.random.rand(100, 200)  # 100 samples, 200 bands
    y = np.random.rand(100)       # target (e.g., soil moisture)
    return X, y


# ------------------------
# Model selector
# ------------------------
def get_model(name, input_dim=None):
    if name == "pls":
        return PLSModel(n_components=10)
    elif name == "rf":
        return RandomForestModel()
    elif name == "xgb":
        return XGBoostModel()
    elif name == "nn":
        return NeuralNetModel(input_dim=input_dim)
    elif name == "cnn":
        return "cnn"  # special case
    else:
        raise ValueError(f"Unknown model: {name}")


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf",
                        help="pls | rf | xgb | nn")
    parser.add_argument("--cv", action="store_true",
                        help="Use cross-validation")
    parser.add_argument("--save", action="store_true",
                        help="Save trained model")

    args = parser.parse_args()

    # Load data
    X, y = load_data()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model
    model = get_model(args.model, input_dim=X.shape[1])

        # ------------------------
    # CNN special case
    # ------------------------
    if args.model == "cnn":
        model = SpectralCNN(input_length=X.shape[1])
        trainer = DLTrainer(model)

        trainer.fit(X_train, y_train)
        preds = trainer.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score

        print({
            "RMSE": mean_squared_error(y_test, preds, squared=False),
            "R2": r2_score(y_test, preds)
        })

        return
        
    # ------------------------
    # Cross-validation
    # ------------------------
    if args.cv:
        print("Running cross-validation...")

        scores = cross_val_score(
            model.model if hasattr(model, "model") else model,
            X,
            y,
            scoring="neg_root_mean_squared_error",
            cv=5
        )

        print("CV RMSE:", -scores.mean())
        return

    # ------------------------
    # Train normally
    # ------------------------
    trainer = Trainer(model)
    trainer.train(X_train, y_train)

    metrics = trainer.evaluate(X_test, y_test)
    print("Evaluation:", metrics)

    # ------------------------
    # Save model
    # ------------------------
    if args.save:
        joblib.dump(model, f"{args.model}_model.pkl")
        print(f"Model saved as {args.model}_model.pkl")


if __name__ == "__main__":
    main()
