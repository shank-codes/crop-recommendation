from flask import Flask,render_template,request,jsonify
from prediction import crop_predict

app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def hello_world():
    return render_template('index.html')
    
@app.route('/predict',methods=["POST"])
def recommend():
    input = request.json
    print(input)
    n = int(input['nitrogen'])
    p = int(input['phosphorous'])
    k = int(input['potassium'])
    temp = float(input['temperature'])
    humidity = float(input['humidity'])
    ph = float(input['ph'])
    rainfall = float(input['rainfall'])
    row = [n,p,k,temp,humidity,ph,rainfall]
    result = crop_predict(row)
    return jsonify({"result": result,"status":200})


if __name__ == '__main__':
    app.run(debug=True,port=8001)