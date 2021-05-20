from flask import Flask,render_template,request


app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def hello_world():
    '''
    n,p,k,temp,humidity,ph,rainfall = 0,0,0,0.0,0.0,0.0,0.0
    if request.method=='POST':
        n = int(request.form['nitrogen'])
        p = int(request.form['phosphorous'])
        k = int(request.form['potassium'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        #res = model.lr.predict((model.scaler.transform([[battery,bluetooth,clkspee]])))
        '''
    return render_template('index.html')
    #return render_template(n,p,k,temp,humidity,ph,rainfall)

@app.route('/result',methods=["POST"])
def recommend():
    n = int(request.form['nitrogen'])
    p = int(request.form['phosphorous'])
    k = int(request.form['potassium'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    return (n,p,k,temp,humidity,ph,rainfall)


if __name__ == '__main__':
    app.run(debug=True,port=8001)