from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

app = Dash(__name__)
server = app.server

# Fetching and preparing the data
input_columns = ['Hmin', 'Hmax', 'eps_p', 'eps_v','COMP_DoOrganizeOnlyAfterHowManyCycles', 
                 'max_N_stencil', 'ord_gradient', 'ord_laplace', 'ord_eval', 
                 'DIFFOP_kernel_Laplace', 'DIFFOP_kernel_Neumann']
output_columns = ['cD-value_Direct', 'cL-value_Direct', 'WallTimeTotal_h']

input_df = pd.read_csv('data/input.csv')
output_df = pd.read_csv('data/results.csv')

input_df = pd.DataFrame(input_df, columns=input_columns)
output_df = pd.DataFrame(output_df, columns=output_columns)

input_df['Hmin'] = input_df['Hmin'].round(decimals = 8)
input_df['Hmax'] = input_df['Hmax'].round(decimals = 8)

data = pd.concat([input_df, output_df], axis=1)

#Predictor part
test_data = pd.read_csv('data/test_data.csv')

inp_cols_cD = ['Hmin', 'Hmax', 'COMP_DoOrganizeOnlyAfterHowManyCycles', 'max_N_stencil', 'ord_laplace', 'DIFFOP_kernel_Neumann']
inp_cols_cL = ['Hmin', 'Hmax', 'max_N_stencil', 'ord_gradient', 'ord_laplace', 'DIFFOP_kernel_Neumann']
inp_cols_Time = ['Hmin', 'Hmax', 'DIFFOP_kernel_Laplace']

X_test_cD = pd.DataFrame(test_data, columns=inp_cols_cD)
X_test_cL = pd.DataFrame(test_data, columns=inp_cols_cL)
X_test_Time = pd.DataFrame(test_data, columns=inp_cols_Time)

y_test = pd.DataFrame(test_data, columns=output_columns)

drag_model = pickle.load(open('models/drag_trained_model.pickle', 'rb'))
lift_model = pickle.load(open('models/lift_trained_model.pickle', 'rb'))
time_model = pickle.load(open('models/time_trained_model.pickle', 'rb'))

y_test_cD_pred_interval = pd.DataFrame(drag_model.predict(X_test_cD, alpha=0.15)[1].reshape(-1,2), index=X_test_cD.index, columns=["left", "right"])
y_test_cL_pred_interval = pd.DataFrame(lift_model.predict(X_test_cL, alpha=0.2)[1].reshape(-1,2), index=X_test_cL.index, columns=["left", "right"])
y_test_Time_pred_interval = pd.DataFrame(time_model.predict(X_test_Time, alpha=0.1)[1].reshape(-1,2), index=X_test_Time.index, columns=["left", "right"])

test_data['cD-value_Direct_min'] = y_test_cD_pred_interval['left']
test_data['cD-value_Direct_max'] = y_test_cD_pred_interval['right']
test_data['cL-value_Direct_min'] = y_test_cL_pred_interval['left']
test_data['cL-value_Direct_max'] = y_test_cL_pred_interval['right']
test_data['WallTimeTotal_h_min'] = y_test_Time_pred_interval['left']
test_data['WallTimeTotal_h_max'] = y_test_Time_pred_interval['right']

#Dashboard Layout

shadow_style = {
    'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
    'borderRadius': '5px',
    'padding': '10px',
    'margin': '10px'
}

recommendations = ['Hmin: Between 0.005 and 0.007',
                   'Hmax: Ideally between 0.03 and 0.046, maximum upto 0.06',
                   'COMP_DoOrganizeOnlyAfterHowManyCycles: 3 or 4 for reduced computation time',
                   'max_N_stencil: 40',
                   'eps_p and eps_v: Higher value for slightly reduced computation time',
                   'ord_laplace: 2.9 if Hmax<=0.046, otherwise 3',
                   'DIFFOP_kernel_Neumann: If Hmax<=0.046 and ord_laplace=2.9, a value of 5 is recommended']

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='MESHFREE Data Exploration', children=[
                       html.Hr(),
                       html.Div(['Select Feature Parameter:',
                                 dcc.Dropdown(input_columns, input_columns[0], id='tab-1_input_dropdown')],
                                 style={"margin-left": "225px", 'width': '30%', 'display': 'inline-block'}),
                       html.Div(['Select Target:',
                                dcc.Dropdown(output_columns, output_columns[0], id='tab-1_output_dropdown')],
                                style={"margin-left": "50px",'width': '30%', 'display': 'inline-block'}),
                       dcc.Graph(id='tab-1_relation_graph')
                       ]),
        dcc.Tab(label='MESHFREE Test Set Exploration', children=[
            html.Hr(),
            html.Div(['Select Feature Parameter:',
                      dcc.Dropdown(input_columns, input_columns[0], id='tab-2_input_dropdown')],
                      style={"margin-left": "225px", 'width': '30%', 'display': 'inline-block'}),
            html.Div(['Select Target:',
                      dcc.Dropdown(output_columns, output_columns[0], id='tab-2_output_dropdown')],
                      style={"margin-left": "50px",'width': '30%', 'display': 'inline-block'}),
            dcc.Graph(id='tab-2_relation_graph')
            ]),
        dcc.Tab(label='Drag-Lift Range Predictor', children=[
            html.Hr(),
            html.Div([
                html.Div([
                    html.Div(['Hmin:',
                              dcc.Slider(min=0.005, max=0.01, step=0.0001, value=0.005, id='tab-3_input_hmin', marks=None, tooltip={"placement": "bottom", "always_visible": True})
                    ], style=dict(flex='100%', **shadow_style)),
                    html.Div(['Hmax:',
                              dcc.Slider(min=0.03, max=0.1, step=0.001, value=0.03, id='tab-3_input_hmax', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                    ], style=dict(flex='100%', **shadow_style)),
                    html.Div([
                        html.Div(['COMP_DoOrganizeOnlyAfterHowManyCycles:',
                                  dcc.RadioItems([1, 2, 3, 4], 1, id='tab-3_input_cycles')
                                  ], style=dict(flex='40%', **shadow_style)),
                        html.Div(['max_N_stencil:',
                                  dcc.RadioItems([30, 40], 30, id='tab-3_input_stencil')
                                  ], style=dict(flex='30%', **shadow_style)),
                    ], style=dict(flex='100%', display='flex', flexDirection='row')),
                    html.Div([
                        html.Div(['ord_gradient:',
                                  dcc.RadioItems([2, 3], 2, id='tab-3_input_ord_grad')
                                  ], style=dict(flex='10%', **shadow_style)),
                        html.Div(['ord_laplace:',
                                  dcc.RadioItems([2, 2.9, 3], 2, id='tab-3_input_ord_lapl')
                                  ], style=dict(flex='10%', **shadow_style)),
                        html.Div(['DIFFOP_kernel_Laplace:',
                                  dcc.RadioItems([2, 6], 2, id='tab-3_input_kernel_lapl')
                                  ], style=dict(flex='35%', **shadow_style)),
                        html.Div(['DIFFOP_kernel_Neumann:',
                                  dcc.RadioItems([2, 5], 2, id='tab-3_input_kernel_neumann')
                                  ], style=dict(flex='35%', **shadow_style))
                    ], style=dict(flex='100%', display='flex', flexDirection='row'))
                   ], style=dict(flex='50%')),
            html.Div(children=[
                html.Div(children=[
                    'Drag coefficient is estimated to lie between:',
                    html.P(id='tab-3_drag'),
                    'Lift coefficient is estimated to lie between:',
                    html.P(id='tab-3_lift'),
                    'Estimated computation Time is approximately:',
                    html.P(id='tab-3_time')
                ], style=dict(flex='30%', **shadow_style)),
                html.Div(children=[
                    'Recommended parameter settings for simulation:',
                    html.Ul(children=[html.Li(i) for i in recommendations])
                ], style=dict(flex='70%', **shadow_style)),
                ], style=dict(flex='50%', display='flex', flexDirection='column'))
                ], style=dict(display='flex', justifyContent='space-between'))
                ])
        ])
    ])


@app.callback(
    Output('tab-1_relation_graph', 'figure'),
    [Input('tab-1_input_dropdown', 'value'),
    Input('tab-1_output_dropdown', 'value')])
def update_graph_tab1(input_column, output_column):

    if(input_column in ['Hmin', 'Hmax']):
        data_sorted = data.sort_values(by=[input_column])
        hover_cols = [col for col in input_columns if col!=input_column]
        
        fig = px.scatter(data_sorted, x=input_column, y=output_column, hover_data=hover_cols)
    else:
        fig = px.violin(data, x=input_column, y=output_column, box=True, points=False)
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=np.sort(data[input_column].unique()))
        
    if(output_column == 'cD-value_Direct'):
        fig.add_hline(y=7, line_width=1, line_dash='dash', line_color='red')
        fig.add_hline(y=5.5, line_width=1, line_dash='dash', line_color='red')
        fig.update_yaxes(title_text='Drag')
    elif(output_column == 'cL-value_Direct'):
        fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='red')
        fig.add_hline(y=0.02, line_width=1, line_dash='dash', line_color='red')
        fig.update_yaxes(title_text='Lift')
    elif(output_column == 'WallTimeTotal_h'):
        fig.update_yaxes(title_text='Runtime (hours)')
    return fig

@app.callback(
    Output('tab-2_relation_graph', 'figure'),
    [Input('tab-2_input_dropdown', 'value'),
    Input('tab-2_output_dropdown', 'value')])
def update_graph_tab2(input_column, output_column):
    fig = go.Figure()
    if(input_column in ['Hmin', 'Hmax']):
        data_sorted = test_data.sort_values(by=[input_column])
        hover_col_list = [col for col in input_columns if col!=input_column]
        
        fig.add_scatter(x=test_data[input_column], y=test_data[output_column], mode='markers', name='Real Value', customdata=test_data[hover_col_list],
                        hovertemplate='<br>'.join([
                            input_column + ': %{x:.5f}',
                            output_column + ': %{y:.5f}',
                            hover_col_list[0] + ': %{customdata[0]}',
                            hover_col_list[1] + ': %{customdata[1]}',
                            hover_col_list[2] + ': %{customdata[2]}',
                            hover_col_list[3] + ': %{customdata[3]}',
                            hover_col_list[4] + ': %{customdata[4]}',
                            hover_col_list[5] + ': %{customdata[5]}',
                            hover_col_list[6] + ': %{customdata[6]}',
                            hover_col_list[7] + ': %{customdata[7]}',
                            hover_col_list[8] + ': %{customdata[8]}',
                            hover_col_list[9] + ': %{customdata[9]}'
                            ]))
        fig.add_scatter(x=[test_data[input_column][0], test_data[input_column][0]],
                    y=[test_data[output_column+'_min'][0], test_data[output_column+'_max'][0]],
                    mode='lines',
                    line_color='#d8c8e5', 
                    name='Predicted Range',
                    showlegend=True)
        for i in range(1,len(test_data)):
            fig.add_scatter(x=[test_data[input_column][i], test_data[input_column][i]],
                    y=[test_data[output_column+'_min'][i], test_data[output_column+'_max'][i]],
                    mode='lines',
                    line_color='#d8c8e5', 
                    showlegend=False)
        fig.update_traces(marker_size=10)
        fig.update_xaxes(title_text=input_column)
    else:
        
        fig.add_trace(go.Violin(x=test_data[input_column], y=test_data[output_column], box_visible=True, meanline_visible=True, name='Real Value'))
        fig.add_trace(go.Violin(x=test_data[input_column], y=test_data[output_column+'_max'], box_visible=True, meanline_visible=True, name='Prediction Upper Limit'))
        fig.add_trace(go.Violin(x=test_data[input_column], y=test_data[output_column+'_min'], box_visible=True, meanline_visible=True, name='Prediction Lower Limit'))

        fig.update_xaxes(title_text=input_column, type='category', categoryorder='array', categoryarray=np.sort(test_data[input_column].unique()))
        
    if(output_column == 'cD-value_Direct'):
        fig.add_hline(y=7, line_width=1, line_dash='dash', line_color='red')
        fig.add_hline(y=5.5, line_width=1, line_dash='dash', line_color='red')
        fig.update_yaxes(title_text='Drag')
    elif(output_column == 'cL-value_Direct'):
        fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='red')
        fig.add_hline(y=0.02, line_width=1, line_dash='dash', line_color='red')
        fig.update_yaxes(title_text='Lift')
    elif(output_column == 'WallTimeTotal_h'):
        fig.update_yaxes(title_text='Runtime (hours)')
    return fig

@app.callback(
    [Output('tab-3_drag', 'children'),
     Output('tab-3_lift', 'children'),
     Output('tab-3_time', 'children')],
    [Input('tab-3_input_hmin', 'value'),
     Input('tab-3_input_hmax', 'value'),
     Input('tab-3_input_cycles', 'value'),
     Input('tab-3_input_stencil', 'value'),
     Input('tab-3_input_ord_grad', 'value'),
     Input('tab-3_input_ord_lapl', 'value'),
     Input('tab-3_input_kernel_lapl', 'value'),
     Input('tab-3_input_kernel_neumann', 'value')])
def update_prediction(hmin, hmax, cycles, stencil, ord_grad, ord_lapl, kernel_lapl, kernel_neumann):
    
    drag_input = pd.DataFrame([[hmin, hmax, cycles, stencil, ord_lapl, kernel_neumann]], columns=inp_cols_cD)
    lift_input = pd.DataFrame([[hmin, hmax, stencil, ord_grad, ord_lapl, kernel_neumann]], columns=inp_cols_cL)
    time_input = pd.DataFrame([[hmin, hmax, kernel_lapl]], columns=inp_cols_Time)

    drag_output = pd.DataFrame(drag_model.predict(drag_input, alpha=0.15)[1].reshape(-1,2), index=[0], columns=['left', 'right'])
    lift_output = pd.DataFrame(lift_model.predict(lift_input, alpha=0.2)[1].reshape(-1,2), index=[0], columns=['left', 'right'])
    time_output = pd.DataFrame(time_model.predict(time_input, alpha=0.1)[1].reshape(-1,2), index=[0], columns=['left', 'right'])

    drag_label = str(round(drag_output.iloc[0,0], 4)) + " - " + str(round(drag_output.iloc[0,1], 4))
    lift_label = str(round(lift_output.iloc[0,0], 4)) + " - " + str(round(lift_output.iloc[0,1], 4))
    time_label = str(round(time_output.iloc[0,0], 4)) + " - " + str(round(time_output.iloc[0,1], 4))

    return drag_label, lift_label, time_label


if __name__ == '__main__':
    app.run(debug=True)
