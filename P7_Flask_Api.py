from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

API_Flask = Flask(__name__)
API_Flask.config["DEBUG"] = True
best_thresh = 0.08

loaded_model = joblib.load('finalized_model.sav')
loaded_X2_test_std = joblib.load('file_X2_test_std.sav')


@API_Flask.route('/', methods=['GET'])
def score():
    y_predict_proba = loaded_model.predict_proba(loaded_X2_test_std)
    class_predict = np.where(y_predict_proba[:, 1]>best_thresh, 1, 0)
    df_scores = pd.DataFrame(y_predict_proba[:, 1],columns=['proba'])
    df_scores['class'] = class_predict
    return df_scores.to_json(date_format='iso', orient='split')



if __name__ == "__main__":
    API_Flask.run(debug=True, use_reloader=False)