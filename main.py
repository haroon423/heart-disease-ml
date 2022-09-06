from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict_placement():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol  = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slop = int(request.form.get('slop'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

    result = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slop, ca, thal]).reshape(1, 13))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'
    # return render_template('index.html', result=result)
    return result
if __name__ == '__main__':
    app.run(debug=True)    