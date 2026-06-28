from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model/crop_yield_model.pkl")

label_encoders = joblib.load("model/label_encoders.pkl")

# =====================================================
# WEATHER API
# =====================================================

API_KEY = "ab0ea0fd5e5aedddd01fd47691b19e40"

# =====================================================
# HOME PAGE
# =====================================================

@app.route('/')

def home():
    return render_template("index.html")

# =====================================================
# PREDICTION
# =====================================================

@app.route('/predict', methods=['POST'])

def predict():

    area = request.form['area']

    item = request.form['item']

    year = int(request.form['year'])

    pesticides = float(request.form['pesticides'])

    city = request.form['city']

    user_rainfall = request.form['rainfall']
    # =================================================
    # FETCH WEATHER DATA
    # =================================================
    try:

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

        response = requests.get(url, timeout=10)
        
        weather_data = response.json()

        print(weather_data)

        if response.status_code != 200:

            return f"Weather API Error: {weather_data}"

        avg_temp = weather_data['main']['temp']


        if user_rainfall.strip() != "":

            rainfall = float(user_rainfall)

        else:

            rainfall = weather_data.get('rain', {}).get('1h', 0)

    except Exception as e:

        print("Weather API Failed")

        print(e)

        # DEFAULT VALUES
        avg_temp = 30

        if user_rainfall.strip() != "":

            rainfall = float(user_rainfall)

        else:

            rainfall = 0

    # =================================================
    # ENCODE INPUTS
    # =================================================

    area_encoded = label_encoders['Area'].transform([area])[0]

    item_encoded = label_encoders['Item'].transform([item])[0]

    # =================================================
    # CREATE INPUT DATAFRAME
    # =================================================

    input_data = pd.DataFrame([[
        area_encoded,
        item_encoded,
        year,
        rainfall,
        pesticides,
        avg_temp
    ]], columns=[
        'Area',
        'Item',
        'Year',
        'average_rain_fall_mm_per_year',
        'pesticides_tonnes',
        'avg_temp'
    ])

    # =================================================
    # PREDICTION
    # =================================================

    prediction = model.predict(input_data)

    result = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction=result,
        temperature=avg_temp,
        rainfall=rainfall
    )

# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)