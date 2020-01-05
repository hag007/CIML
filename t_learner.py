import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calc_t_learner(file_name):
    df=pd.read_csv("datasets/{}.csv".format(file_name), index_col=0)
    df_treated=df[df["T"]==1]
    df_control = df[df["T"] == 0]

    model_1 = LinearRegression()
    model_1.fit(df_treated.drop(["Y"],axis=1), df_treated["Y"])
    res_1=model_1.predict(df_treated.drop(["Y"],axis=1))

    model_2 = LinearRegression()
    model_2.fit(df_control.drop(["Y"], axis=1), df_control["Y"])
    res_2=model_2.predict(df_treated.drop(["Y"],axis=1))

    att=(res_1-res_2).mean()
    print att
    return att

if __name__=="__main__":

    att_1=calc_t_learner("data1_p")
    att_2=calc_t_learner("data2_p")
    try:
        df=pd.read_csv("output/agg_result.csv",index_col=0)
    except:
        df =pd.DataFrame()

    df.loc[3,"data1"]=att_1
    df.loc[3, "data2"]=att_2
    df.to_csv("output/agg_result.csv")