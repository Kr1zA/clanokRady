import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
    
def watchout():
  print("computers are transforming into a noose and a yoke for humans" )  

def draw(x, datum_alebo_cisla="datum", y1_orig=None, y2_Percentage=None, y3_Log=None):
    fig = go.Figure()
    x1 = x.index.get_level_values(datum_alebo_cisla)
    if y1_orig is not None:
        fig.add_trace(go.Scatter(x=x1, y=y1_orig,
                            mode='lines',
                            name='orig'))
    if y2_Percentage is not None:
        fig.add_trace(go.Scatter(x=x1, y=y2_Percentage,
                            mode='lines+markers',
                            name='predicted'))

    fig.show()
    
def draw_cisla(x, y1_orig=None, y2_Percentage=None, y3_Log=None):
    fig = go.Figure()
    x1 = x.index
    if y1_orig is not None:
        fig.add_trace(go.Scatter(x=x1, y=y1_orig,
                            mode='lines',
                            name='orig'))
    if y2_Percentage is not None:
        fig.add_trace(go.Scatter(x=x1, y=y2_Percentage,
                            mode='lines+markers',
                            name='predicted'))

    fig.show()
    
def draw_multistep(x, y=None, y_pred=None):
    fig = go.Figure()
    x_index = x.index
    fig.add_trace(go.Scatter(x=x_index, y=x,
                            mode='lines',
                            name='x'))
    if y is not None:
        y_index = y.index
        fig.add_trace(go.Scatter(x=y_index, y=y,
                            mode='lines',
                            name='y'))
    if y_pred is not None:
        y_pred_index = y_pred.index
        fig.add_trace(go.Scatter(x=y_pred_index, y=y_pred,
                            mode='lines+markers',
                            name='y_predicted'))
    fig.show()
    
def draw_more(title_text, names, y, x=None, y_pred=None):
    fig = make_subplots(rows=len(y), cols=1, subplot_titles=names)
    for i in range(len(y)):
        if x is not None:
            x_index = x[i].index
            fig.append_trace(go.Scatter(
                x=x_index,
                y=x[i],
                name='orig',
                line_color='blue'
            ), row=i+1, col=1)
        y_index = y[i].index
        fig.append_trace(go.Scatter(
            x=y_index,
            y=y[i],
            name='orig',
            line_color='blue'
        ), row=i+1, col=1)
        if y_pred is not None:
            fig.append_trace(go.Scatter(
                x=y_index,
                y=y_pred[i],
                name='pred',
                line_color='red'
            ), row=i+1, col=1)
        
    fig.update_layout(
        height=300*len(y), 
#         width=2000, 
        title_text=title_text)
    fig.show()


def create_percentage_change(df, alfa=1):
    df1 = df.copy()
    df1 = df1.drop(df1.index[0])
    df1 = df1.iloc[:,:]-df.iloc[:-1,:].values
    df1 = df1.iloc[:,:]/(df.iloc[:-1,:].values + alfa)
    return df1

def create_log_difference(df, alfa=1):
    df1 = df.copy()
    df1 = df1.drop(df1.index[0])
    df1 = np.log( (df1 + alfa) / (df.iloc[:-1,:].values + alfa) )
    return df1

def back_from_percentage_change(df, pred, alfa=1):
    return pred*(df + alfa)+df

def back_from_log_difference(df, pred, alfa=1):
    return np.e**(pred + np.log(df + alfa))-alfa

def make_dataset(df):
    casy = pd.read_csv("casy.csv")
    df["Date Time"]=casy
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    datetime_series = pd.to_datetime(date_time)
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df=df.set_index(datetime_index)
    arrays = [list(range(52417)), list(df.index )]
    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["cislo", "datum"])
    aa = pd.DataFrame(df.values, index=index)
    aa.columns = df.columns
    df = aa
    return df

def create_x_y(df, input_length, output_length):
    X, y = list(), list()
    for i in range(len(df)):
        # find the end of this pattern
        id_x = i + input_length
        id_y = i + input_length + output_length
        # check if we are beyond the sequence
        if id_y > len(df):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df[i:id_x], df[id_x:id_y]
        X.append(seq_x)
        y.append(seq_y)
    
    return X, y