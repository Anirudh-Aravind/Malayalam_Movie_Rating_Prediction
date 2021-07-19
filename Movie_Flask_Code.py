from flask import Flask,render_template,request
import joblib
import numpy as np
app = Flask(__name__)
model=joblib.load("BernoulliNB_model")
@app.route('/')
def home():
    return render_template("html_code.html")
@app.route('/predict', methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    if prediction==2:
        prediction="Good"
    elif prediction==0:
        prediction="Average"
    else:
        prediction="Bad"
    return render_template("html_code.html", prediction_text=f"It will be an {prediction} movie")
if __name__=="__main__":
    app.run(debug=True)