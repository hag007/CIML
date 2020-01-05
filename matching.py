import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

def calc_matching(file_name):
    df=pd.read_csv("datasets/{}.csv".format(file_name), index_col=0)
    df_treated=df[df["T"]==1]
    df_control = df[df["T"] == 0]

    topn=5
    n_jobs=1

    knn = NearestNeighbors(n_neighbors=topn + 1, metric='euclidean', n_jobs=n_jobs)
    knn.fit(df_control.drop(["T","Y"], axis=1))

    distances, indices = knn.kneighbors(df_treated.drop(["T","Y"], axis=1))
    vfunc = np.vectorize(lambda a: df_control.iloc[a].loc["Y"])
    res=vfunc(indices)
    ite=df_treated.loc[:,"Y"]-res.mean(axis=1)
    ate=ite.sum()/float(len(ite))

    print ate
    return ate

if __name__=="__main__":

    att_1=calc_matching("data1_p")
    att_2=calc_matching("data2_p")
    try:
        df=pd.read_csv("output/agg_result.csv",index_col=0)
    except:
        df =pd.DataFrame()

    df.loc[4,"data1"]=att_1
    df.loc[4, "data2"]=att_2
    df.to_csv("output/agg_result.csv")