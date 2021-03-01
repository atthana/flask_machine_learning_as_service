from flask import Flask, request
from flask_cors import CORS, \
    cross_origin  # ติดตั้งตัวนี้เพิ่มเพื่อให้สามารถเรียกใช้งานผ่านจากภายนอกได้ กรณีคนละ network
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/')  # เพิ่ม route หรือ วิธีการในการเรียก
@cross_origin()
def helloworld():
    return 'helloworld'


@app.route('/area', methods=['GET'])
@cross_origin()  # ใส่เพื่อให้ใช้ API จากภายนอกได้
def area():
    width = float(request.values['w'])
    height = float(request.values['h'])
    return str(width * height)


@app.route('/bmi', methods=['GET'])
@cross_origin()  # ใส่เพื่อให้ใช้ API จากภายนอกได้
def bmi():
    weight = float(request.values['weight'])
    height = float(request.values['height'])
    bmi = weight/((height/100) * (height/100))
    return str(bmi)


@app.route('/iris', methods=['POST'])  # ใช้ method POST บ้าง
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')  # วาง model ไว้ที่ตำแหน่งเดียวกันนะ ไม่งั้นต้องใส่ path
    req = request.values['param']  # ใส่ค่าทั้งหมดในตัวแปร param ตัวเดียวเลย เด๋วค่อยไปแยกเอา
    inputs = np.array(req.split(','), dtype=np.float32).reshape(1, -1)  # จัดการกับ req ที่รับเข้ามาโดยแยกด้วย ',' และ reshape เป็น 1 กับ -1
    predict_target = model.predict(inputs) # พอได้ format ที่ตรงกับ model แล้วก้อส่งค่าเข้าไป predict เลย
    if predict_target == 0:  # output จะออกมาเป็น 0-2 เราก้อมาแยกอีกที
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
