import efinance as ef
import pandas as pd
import numpy as np
import os
import sys
import datetime
import sklearn.linear_model as lm

os.chdir(sys.path[0])

def pre_time():
    time = datetime.datetime.now()
    train_stt = (time + datetime.timedelta(days=-1461)).strftime("%Y%m%d")
    test_stt = (time + datetime.timedelta(days=-5)).strftime("%Y%m%d")

    curt = (time).strftime("%Y%m%d")

    return [curt,train_stt,test_stt]

def get_data(stock_code):
    tlst = pre_time()
    df = ef.stock.get_quote_history(stock_code, beg=tlst[1], end=tlst[0])
    #print(df)
    arr = df[["开盘","收盘"]].to_numpy()
    train_op_x,train_op_y,train_ed_x,train_ed_y = [],[],[],[]
    for i in range(len(arr)-5):
        train_op_x.append([arr[i][0],arr[i+1][0],arr[i+2][0],arr[i+3][0],arr[i+4][0]])
        train_op_y.append(arr[i+5][0])
        train_ed_x.append([arr[i][1],arr[i+1][1],arr[i+2][1],arr[i+3][1],arr[i+4][1]])
        train_ed_y.append(arr[i+5][1])
    name = df["股票名称"].unique()

    return[name,np.array(train_op_x),np.array(train_op_y),np.array(train_ed_x),np.array(train_ed_y),np.array(arr[-5:,0].reshape(1,-1)),np.array(arr[-5:,1].reshape(1,-1))]

def perround(x):
    return (round(x*100)/100)

def train(datax,datay):
    model = lm.LinearRegression()
    model.fit(datax,datay)
    return model

if __name__ == "__main__":

    stock_code = '000300'

    stock_code = input("输入股票代码：")

    [name,datax_op,datay_op,datax_ed,datay_ed,test_op,test_ed] = get_data(stock_code)

    model_op = train(datax_op,datay_op)
    pred_op = model_op.predict(test_op)

    model_ed = train(datax_ed,datay_ed)
    pred_ed = model_ed.predict(test_ed)


    print("股票",name[0],"开盘价预测：",perround(pred_op[0]),"收盘价预测：",perround(pred_ed[0]))
