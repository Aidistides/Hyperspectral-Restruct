from hyperspectral_soil.models import RandomForestModel, Trainer

model = RandomForestModel()

trainer = Trainer(model)
trainer.train(X_train, y_train)

metrics = trainer.evaluate(X_test, y_test)

print(metrics)
