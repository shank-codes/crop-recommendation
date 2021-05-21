from flask import Flask,render_template,request,jsonify
from prediction import crop_predict

app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def hello_world():
    return render_template('index.html')
    
@app.route('/predict',methods=["POST"])
def recommend():
    n = int(request.form['nitrogen'])
    p = int(request.form['phosphorous'])
    k = int(request.form['potassium'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    row = [n,p,k,temp,humidity,ph,rainfall]
    result = crop_predict(row)
    return jsonify({"result":result,"status":200})


if __name__ == '__main__':
    app.run(debug=True,port=8001)