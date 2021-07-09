import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime
import pandas as pd
import yahoo_fin.stock_info as yf

# Get data from csv files
test_data = pd.read_csv('test.csv')
start = test_data['Unnamed: 0'][0]
data_df = pd.read_csv('AAPL.csv')
data_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
dates = data_df.loc[start:]['Date']
test_data['Date'] = dates.values

app = dash.Dash()
fig_1 = {'x': data_df['Date'], 'y': data_df['close'], 'type': 'line', 'name': 'Close (Real)'}
fig_2 = {'x': test_data['Date'], 'y': test_data['Predicted'], 'type': 'line', 'name': 'Close (Predicted)'}

app.layout = html.Div([dcc.Tabs([dcc.Tab(label='AAPL LSTM Model', children=
                                                                    [html.H2('LSTM Model',style={"textAlign": "center"}), 
                                                                    dcc.Graph(figure={'data': [fig_1, fig_2],
                                                                    'layout':{'title': 'LSTM Prediction vs Real Price', 
                                                                    'xaxis':{'title':'Date'
            },
            'yaxis':{'title':'Close Price'
            }}
                                                                    }
                                                                    )
                                                                    ]),
                                dcc.Tab(label='Stock Charts', children = [html.H2('Ticker to Graph ', style={"textAlign": "center"}), 
                                html.Div(dcc.Input(id='input', value='', type='text'), style={"display":'flex', "justifyContent":'center'}),
                                html.Div(id='Graph')
                                ])]
                                       )])

@app.callback(
    Output(component_id='Graph', component_property='children'), 
    Input(component_id='input', component_property='value')
    )
def update_graph(input):
    start = datetime.datetime(2014, 2,1)
    end = datetime.datetime.now()
    try:
        input_df = yf.get_data(input, start_date=start, end_date=end)
        return dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': input_df.index, 'y': input_df['close'], 'type': 'line', 'name': input.upper()},
            ],
            'layout': {
                'title': input.upper(), 'xaxis':{
                'title':'Date'
            },
            'yaxis':{
                 'title':'Close Price'
            }
            }
        }
    )
    except:
        if input == '':
            return 'Enter a Ticker'
        else:
            return 'Invalid Ticker'

if __name__ == '__main__':
    app.run_server(debug=True)