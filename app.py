import joblib
from flask import Flask, request, jsonify,  render_template
import pandas as pd

app = Flask(__name__, static_url_path='/static')

model , dummy_columns= joblib.load('model.pkl')

val_X_columns = ['HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals', 'year', 'month'] + dummy_columns



@app.route('/')
def home():
   return  render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    HomeTeam = data['HomeTeam']
    AwayTeam = data['AwayTeam']
    year = data['year']
    month = data['month']
    HTHG = data['HTHG']
    HTAG = data['HTAG']
    
    # Create a DataFrame with zeros
    dataframe = pd.DataFrame(0, index=[0], columns=dummy_columns)
    
    # Set the specific input values
    dataframe['HalfTimeHomeTeamGoals'] = HTHG
    dataframe['HalfTimeAwayTeamGoals'] = HTAG
    dataframe['year'] = year
    dataframe['month'] = month

    for team_col in dummy_columns:
        if team_col.startswith(f'HomeTeam_{HomeTeam}'):
            dataframe[team_col] = 1 if team_col == f'HomeTeam_{HomeTeam}' else 0
        elif team_col.startswith(f'AwayTeam_{AwayTeam}'):
            dataframe[team_col] = 1 if team_col == f'AwayTeam_{AwayTeam}' else 0
        else:
            dataframe[team_col] = 0  # Ensure all dummy columns are initialized
    
    dataframe = dataframe[val_X_columns]

    # Predict the result
    prediction = model.predict_proba(dataframe)
    
    return jsonify({'prediction': prediction.tolist()})


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content response



if __name__ == '__main__':
    app.run(debug=True)