

import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import gc
import json
import joblib
import dash_bootstrap_components as dbc

from flask import Flask, render_template, jsonify
import requests

Threshold = 0.08

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#server = app.server

# load model
SCORE_API_URL ='http://127.0.0.1:5000/'

# send request to score API in order to get scores dataFrame

response = requests.get(SCORE_API_URL)

jsonified_df_scores = response.content.decode('utf-8')

df_scores = pd.read_json(jsonified_df_scores, orient='split')

df_X2_train = joblib.load('file_X2_train.sav')
df = pd.read_csv('features.csv')

dff = df[['PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']]


df_f_importance = pd.read_csv('feature_importances.csv').head(10)


# Reading rhe Shap values that have been preprocessed earlier.
# Could be put in preprocessing step BuildDataFromZipFile as well
shapValues1 = np.load("shapValues20K.npy")

#available_features = df_scores.columns

def getCustomerFeatures(CustId, NbFeatures = 10):
    maxFeatureId = sorted(range(len(shapValues1[CustId])),
                          key=lambda x: abs(shapValues1[CustId][x]))[-NbFeatures:]
    FeatureNames = np.empty(NbFeatures, dtype=object)
    FeatureShapValues = np.empty(NbFeatures, dtype=float)
    #FeatureStdValues = np.empty(NbFeatures, dtype=float)
    for i, Id in enumerate(maxFeatureId):
        FeatureNames[i] = df_X2_train.columns[Id]
        FeatureShapValues[i] = shapValues1[CustId][Id]
        #FeatureStdValues[i] = df_X2_train.iloc[CustId][Id]
    positive = FeatureShapValues > 0
    colors = list(map(lambda x: 'red' if x else 'blue', positive))
    return (FeatureNames, FeatureShapValues, colors)


def sortSecond(val):
    return val[1]



# Coefficient Figure
coef_fig = px.bar(df_f_importance, y=df_f_importance['feature'], x=df_f_importance['importance'], orientation="h",
                 title="Mean Feature Importance")

coef_fig.update_traces(marker_color='orange')
coef_fig.update_layout(width=700, height=300, bargap=0.05,
                       margin=dict(l=100, r=100, t=50, b=50))

# Data for probability distribution
NbBins = 500
dist = pd.cut(df_scores['proba'], bins=NbBins).value_counts()
dist.sort_index(inplace=True)
ticks = np.linspace(0, 1, NbBins)
DashBoradTH=int(Threshold*len(dist))

# Layout of the Dashboard
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H6("Customer Selection"),
            html.Div([
                   html.P("Custumer Id: "),
                   dbc.Input(id="customer-id", value=0, type="number", min=0, max=df_scores.shape[0]),
                   dbc.Card(
                     [
                        html.H3(id='cust-answer', className="card-title"),
                        html.H6(id='cust-score', className="card-title"),
                     ]),
                   html.Br(),
                  ], style={'width': '48%', 'display': 'inline-block'}),
                  html.Div([dcc.Graph(id='proba-score')
                  ], style={'width': '48%', 'display': 'inline-block'}),

                  html.Div([
                    dcc.Graph(id='feature-graphic1')],
                    style={'width': '100%', 'display': 'inline-block'}),

    ],
    style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(figure=coef_fig),
    ],
    style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            
            dcc.Graph(id='tip-graph'),
            
            dcc.Dropdown(
                id='column-dropdown', clearable=False,
                options=[{'label': i, 'value': i} for i in dff.drop(['TARGET'], axis=1).columns.unique()],
                value='EXT_SOURCE_1'
             ),
            

        ],
        style={'width': '48%', 'display': 'inline-block'})

      
    ])

    
])

# Callbacks functions

# Callback for the customer feature importance
@app.callback(
    Output('feature-graphic1', 'figure'),
    [Input('customer-id', 'value')])
def update_graph(customerId):
    FeatureNames, FeatureShapValues, colors = getCustomerFeatures(customerId)
    cust_coef_fig = px.bar(
        y=FeatureNames,
        x=FeatureShapValues,
        orientation="h",
#        color=colors,
        labels={"x": "Weight on Prediction", "y": "Features"},
        title="Customer Feature Importance",
    )
    cust_coef_fig.update_traces(marker_color=colors)
    cust_coef_fig.update_layout(width=700, height=300, bargap=0.05,
                                margin=dict(l=100, r=100, t=50, b=50))

    return cust_coef_fig

# Callback for the prediction and score of the selected customer
@app.callback(
    Output('cust-answer', 'children'),
    [Input('customer-id', 'value')])
def update_score(customerId):
    Score = df_scores.loc[customerId, 'proba']
    Answer = 'Accepted' if Score <= Threshold else 'Refused'
    return f"{Answer}"

@app.callback(
    Output('cust-score', 'children'),
    [Input('customer-id', 'value')])
def update_score(customerId):
    Score = df_scores.loc[customerId, 'proba']
    return f"Score: {Score:.2f}"

# Callback for the probability distribution 
@app.callback(
    Output('proba-score', 'figure'),
    [Input('customer-id', 'value')])
def update_graph(customerId):
    fig = go.Figure(data = go.Scatter(x=ticks[:DashBoradTH],
                                      y=dist[:DashBoradTH],
                                      mode='lines',
                                      marker=dict(color='blue'),
                                      name='Success'))
    fig.add_trace(go.Scatter(x=ticks[DashBoradTH:],
                             y=dist[DashBoradTH:],
                             mode='lines',
                             marker=dict(color='red'),
                             name='Default'))
    Score = df_scores.loc[customerId, 'proba']
    rank = int(Score*len(dist))-1
    fig.add_trace(go.Scatter(x=[Score], y=[dist[rank]], mode='markers',
                             marker=dict(size=15, color='black'),
                             name='Customer'))
    fig.update_layout(
        title="Prediction Distribution",
        margin=dict(l=20, r=20, t=40, b=20),
        width=600, height=150,
        paper_bgcolor="LightSteelBlue",
    )

    return fig

# Callback for the X axis feature distribution for accepted and refused customers 
@app.callback(
    Output('tip-graph', 'figure'),
    [Input('column-dropdown', 'value'),
     Input('customer-id', 'value')])

def update_tip_figure(selection, customerId):
    hist_data = [df.loc[df['TARGET'] == 0, selection], df.loc[df['TARGET'] == 1, selection]]
    group_labels = ['target == 0', 'target == 1']
    fig = ff.create_distplot(hist_data, group_labels,

                             show_hist=False,

                             show_rug=False)

    title="{} Feature Distribution".format(selection)

    fig.update_layout(shapes=[

        dict(

            type= 'line',

            yref= 'paper',

            y0= 0,

            y1= 1,

            xref= 'x',

            x0= dff.iloc[customerId][selection],

            x1= dff.iloc[customerId][selection]

        )],

        title=title,

        margin=dict(l=20, r=20, t=40, b=20),

        width=400, height=200,

        paper_bgcolor="LightSteelBlue",

)
    return fig






if __name__ == '__main__':
    app.run_server(debug=True)