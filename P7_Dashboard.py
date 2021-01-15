import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import joblib
from flask import Flask, render_template, jsonify
import requests

df = pd.read_csv('df2.csv')
df_X2_test = pd.read_csv('df_X2_test.csv')

list_feature = ['PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']
dff1 = df[['PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']]
dff=dff1.head(10000)
df_X2_test = pd.read_csv('df_X2_test.csv')

df_f_importance = pd.read_csv('feature_importances.csv')

df_f_importance = df_f_importance.head(10)

#scoreApi = Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,  external_stylesheets=external_stylesheets)

SCORE_API_URL ='http://127.0.0.1:5000/'

# send request to score API in order to get scores dataFrame

response = requests.get(SCORE_API_URL)

jsonified_df_scores = response.content.decode('utf-8')

df_scores = pd.read_json(jsonified_df_scores, orient='split')

fig1 = px.bar(df_f_importance, x=df_f_importance['feature'], y=df_f_importance['importance'])

fig1.update_layout(transition_duration=500)




app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='Feature Importance'),

        html.Div(children='''
            Les dix premières meilleures features.
        '''),

        dcc.Graph(
            id='graph',
            figure=fig1
        ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Courbe de densité selon classe 0 et 1'),
        dcc.Graph(id='tip-graph'),
        html.Label([
            "selection",
            dcc.Dropdown(
                id='column-dropdown', clearable=False,
                options=[
                    {'label': c, 'value': c}
                    for c in dff.drop(['TARGET'], axis=1).columns.unique()
                ])
        ]),
        
        html.Label([
            "customerId",
            dcc.Dropdown(
                id='column-dropdown1', clearable=False,
                options=[
                    {'label': c, 'value': c}
                    for c in dff.index
                ])
        ]),
    ]),
    
    # New Div for all elements in the new 'row' of the page
    html.Div([ 
        html.H1(children='Calcul et affichage du score pour un client'),
        dcc.Graph(id='tip1-graph'),
        html.Label([
            "indexclient",
            dcc.Dropdown(
                id='index-dropdown', clearable=False,
                options=[
                    {'label': c, 'value': c}
                    for c in df_X2_test.index
                ])
        ]),
    ]),
    
    html.Div([ 
        html.H1(children='Courbe du score clients'),
        dcc.Graph(id='tip2-graph'),
        html.Label([
            "indexclient1",
            dcc.Dropdown(
                id='index-dropdown1', clearable=False,
                options=[
                    {'label': c, 'value': c}
                    for c in df_X2_test.index
                ])
        ]),
    ])
    
    
])

# Callback function that automatically updates the tip-graph based on chosen colorscale
@app.callback(
    Output('tip-graph', 'figure'),
    [Input("column-dropdown", "value"),
    Input("column-dropdown1", "value")]
)
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

@app.callback(
    Output('tip1-graph', 'figure'),
    [Input("index-dropdown", "value")]
)
def update_tip_figure(indexclient):
    
   
    fig = go.Figure(data=[go.Table(header=dict(values=['Index_client', 'Score']),
                 cells=dict(values=[[indexclient], [df_scores.iloc[indexclient]['proba']]]))
                     ])
    
    
    
    return  fig 


@app.callback(
    Output('tip2-graph', 'figure'),
    [Input("index-dropdown1", "value")]
    
)

def update_tip_figure(indexclient1):
    
    fig = ff.create_distplot([df_scores['proba']], ['Score_client'],

                             show_hist=False, 

                             show_rug=False)

    title="{} Feature Distribution".format('proba')

    fig.update_layout(shapes=[

        dict(

            type= 'line',

            yref= 'paper',

            y0= 0,

            y1= 1,

            xref= 'x',

            x0= df_scores.iloc[indexclient1]['proba'],

            x1= df_scores.iloc[indexclient1]['proba']

        )],

        title=title,

        margin=dict(l=20, r=20, t=40, b=20),

        width=400, height=200,

        paper_bgcolor="LightSteelBlue",

)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)