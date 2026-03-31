def predict(model, scaler, input_features):
    scaled = scaler.transform([input_features])
    pred = model.predict(scaled)

    if pred[0] == 1:
        return "Mine"
    else:
        return "Rock"