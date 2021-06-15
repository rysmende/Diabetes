import json
from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

def vectorize(data):
    age = data['age']
    if age < 40:
        age = 0
    elif age >= 40 & age < 50:
        age = 1
    elif age >= 50 & age < 59:
        age = 2
    else:
        age = 3
    
    pa = data['physical_activity']
    if pa > 0.1 & pa <= 0.5:
        pa = 0.33
    elif pa > 0.5 & pa < 1.0:
        pa = 0.66
    elif pa >= 1.0:
        pa = 1.0
    else:
        pa = 0
        
    height_m = data['height_m'] / 100
    
    bmi = round(data['weight_kg'] / height_m ** 2, 1)
    
    jf = data['junk_food']
    junk_food = 0.0
    if jf == 0:
        junk_food = 0.1
    elif jf == 1:
        junk_food = 0.35
    elif jf == 2:
        junk_food = 0.65
    else:
        junk_food = 0.9

    st = data['stress']
    stress = 0.0
    if st == 1:
        stress = 0.33
    elif st == 2:
        stress = 0.66
    elif st == 3:
        stress = 1.0
        
    bp = data['blood_pressure']
    blood_pressure = 0.0
    if bp == 1:
        blood_pressure = 0.5
    elif bp == 2:
        blood_pressure = 1.0
        
    return np.array([age, data['gender'], data['family_diabetes'], pa, bmi, data['smoking'], data['alcohol'], data['sleep'], data['sound_sleep'], data['regular_medicine'], junk_food, stress, blood_pressure, data['pregnancies'], data['gestational_diabetes'], data['urination_frequency']])


app = Flask(__name__)

model = None 

path = 'lib/model/Classifier.pkl'
with open(path, 'rb') as f:
    model = pickle.load(f)

# argument parsing
@app.route('/', methods=['POST'])
def diagnose():
    record = json.loads(request.data)
    
    x = vectorize(record)
    y = model.predict_proba(x)
    
    probability = y[1]
    return jsonify({"Probability" : probability})

# Setup the Api resource routing here
# Route the URL to the resource

if __name__ == '__main__':
    app.run(debug=True)
    


d = {
	"age" : 21,
	"gender" : 0,
	"family_diabetes" : 1,
	"physical_activity" : 1.0,
	"height_cm" : 190,
	"weight_kg" : 101,
	"smoking" : 0,
	"alcohol" : 1,
	"sleep" : 9,
	"sound_sleep" : 8,
	"regular_medicine" : 0,
	"junk_food" : 2,
	"stress" : 1,
	"blood_pressure" : 1,
	"pregnancies" : 0,
	"gestational_diabetes" : 0,
	"urination_frequency" : 0
}
