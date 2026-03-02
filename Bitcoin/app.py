import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template, send_file
import io, os, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and process data
df = pd.read_csv("Bitcoin Historical Data (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Clean numeric columns
for col in ['Price', 'Open', 'High', 'Low', 'Vol.']:
    df[col] = df[col].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# Train model
X = df[['Open', 'High', 'Low', 'Vol.']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

app = Flask(__name__)
CSV_PATH = 'predictions.csv'

def is_valid_input(val):
    try:
        f = float(val)
        if f < 0:
            return False
        return True
    except:
        return False

@app.route('/')
def home():
    return render_template("index.html", prediction=None, show_plot=False, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    error = None
    try:
        # Validate inputs
        inputs = {}
        for field in ['Open', 'High', 'Low', 'Vol.']:
            val = request.form.get(field)
            if val is None or not is_valid_input(val):
                error = f"Invalid input for {field}. Please enter a non-negative number."
                return render_template("index.html", prediction=None, show_plot=False, error=error)
            inputs[field] = float(val)

        input_data = np.array([inputs['Open'], inputs['High'], inputs['Low'], inputs['Vol.']]).reshape(1, -1)
        prediction = round(model.predict(input_data)[0], 2)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = pd.DataFrame([[timestamp, inputs['Open'], inputs['High'], inputs['Low'], inputs['Vol.'], prediction]],
                           columns=['Timestamp', 'Open', 'High', 'Low', 'Volume', 'Predicted Price'])

        if os.path.exists(CSV_PATH):
            row.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            row.to_csv(CSV_PATH, index=False)

        # Plotting with better date formatting
        data = pd.read_csv(CSV_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        plt.figure(figsize=(10, 4))
        plt.plot(data['Timestamp'], data['Predicted Price'], marker='o', color='gold')
        plt.title('Bitcoin Predicted Price Over Time', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/plot.png')
        plt.close()

        return render_template("index.html", prediction=prediction, show_plot=True, error=None)

    except Exception as e:
        error = f"Error: {e}"
        return render_template("index.html", prediction=None, show_plot=False, error=error)

@app.route('/download_csv', methods=['GET'])
def download_csv():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True)
    else:
        return "CSV file not found."

@app.route('/download_data', methods=['GET'])
def download_data():
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", download_name="historical_data.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True