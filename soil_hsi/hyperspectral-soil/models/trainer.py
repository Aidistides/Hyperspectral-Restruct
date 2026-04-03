from sklearn.metrics import mean_squared_error, r2_score

class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        return {
            "RMSE": rmse,
            "R2": r2
        }
