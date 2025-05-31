from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("soil_classifier.joblib")
transformer = joblib.load("transformer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        input_data = request.form.to_dict()

        categorical_fields = ['Photoperiod', 'Category_pH']

        for key in input_data:
            if key not in categorical_fields:
                input_data[key] = float(input_data[key])

        df_input = pd.DataFrame([input_data])
        transformed = transformer.transform(df_input)
        pred_encoded = model.predict(transformed)
        prediction = label_encoder.inverse_transform(pred_encoded)[0]

        return redirect(url_for("result", prediction=prediction))

    return render_template("predict.html")

@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
