import pandas as pd
from sklearn.linear_model import LinearRegression

def calc_s_learner(file_name):
    df = pd.read_csv("datasets/{}.csv".format(file_name), index_col=0)
    df_treated = df[df["T"] == 1]
    df_control = df[df["T"] == 0]


    model = LinearRegression()
    model.fit(df.drop(["Y"], axis=1), df["Y"])
    sample=df_treated.drop(["Y"], axis=1)
    res_1 = model.predict(sample)
    sample["T"]=0
    res_2 = model.predict(sample)

    att = (res_1 - res_2).mean()
    print att
    return att

if __name__=="__main__":

    att_1=calc_s_learner("data1_p")
    att_2=calc_s_learner("data2_p")
    try:
        df=pd.read_csv("output/agg_result.csv",index_col=0)
    except:
        df =pd.DataFrame()

    df.loc[2,"data1"]=att_1
    df.loc[2, "data2"]=att_2
    df.to_csv("output/agg_result.csv")