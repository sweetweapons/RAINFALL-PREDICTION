# import pandas as pd
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('rf_model_annual.pkl', 'rb'))

# # Load the dataset for subdivisions
# df = pd.read_csv("sub-division_rainfall_act_dep_1901-2015.csv")
# df_actual = df[df['Parameter'] == 'Actual']
# subdivisions = df_actual['SUBDIVISION'].unique()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     if request.method == 'POST':
#         subdivision = request.form['subdivision']
#         year = int(request.form['year'])
#         jun = float(request.form['jun'])
#         jul = float(request.form['jul'])
#         aug = float(request.form['aug'])
#         may = float(request.form['may'])
#         sep = float(request.form['sep'])

#         # Preprocess input
#         subdivision_code = pd.Series([subdivision]).astype('category').cat.codes[0]

#         # Prepare input for the model
#         input_data = np.array([[subdivision_code, jun, jul, aug, may, sep]])

#         # Make prediction
#         prediction = model.predict(input_data)[0]

#     return render_template('index.html', prediction=prediction, subdivisions=subdivisions)

# if __name__ == '__main__':
#     app.run(debug=True)



import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('rf_model_annual.pkl', 'rb'))

# Load the dataset for subdivisions
df = pd.read_csv("sub-division_rainfall_act_dep_1901-2015.csv")
df_actual = df[df['Parameter'] == 'Actual']
subdivisions = df_actual['SUBDIVISION'].unique()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    year = None
    plot_url = None

    if request.method == 'POST':
        subdivision = request.form['subdivision']
        year = int(request.form['year'])
        jun = float(request.form['jun'])
        jul = float(request.form['jul'])
        aug = float(request.form['aug'])
        may = float(request.form['may'])
        sep = float(request.form['sep'])

        # Preprocess input
        subdivision_code = pd.Series([subdivision]).astype('category').cat.codes[0]

        # Prepare input for the model
        input_data = np.array([[subdivision_code, jun, jul, aug, may, sep]])

        # Make prediction
        prediction = model.predict(input_data)[0]

                # Plot the predicted value
        plt.figure(figsize=(6, 4))
        months = ['May', 'June', 'July', 'August', 'September']
        values = [may, jun, jul, aug, sep]

        # Plot the monthly rainfall values
        plt.plot(months, values, marker='o', color='blue', label='Monthly Rainfall')
        # Plot the predicted annual rainfall as a horizontal line
        plt.axhline(y=prediction, color='r', linestyle='--', label='Predicted Annual Rainfall')

        # Adding text annotation for the predicted value
        plt.text(x=2, y=prediction + 20, s=f'Predicted: {prediction:.2f} mm', color='red')

        plt.title(f'Rainfall Prediction for {subdivision} in {year}')
        plt.xlabel('Months')
        plt.ylabel('Rainfall (mm)')
        plt.xticks(months)  # Ensures all month labels are shown
        plt.legend()


        # Save the plot as a PNG image to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # Debugging prints
        print("Plot URL length:", len(plot_url))  # Check the length of the plot URL
        print("Plot URL preview:", plot_url[:30])  # Print a preview of the plot URL

    return render_template('index.html', prediction=prediction, year=year, subdivisions=subdivisions, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
