import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

poi_dis = np.load("data/50_poi_dis.npz")["arr_0"]

house_sit = pd.read_excel("data/house_situation.xlsx")
# classify the green level of different residential community.
house_green = house_sit["绿化率"]
house_green = house_green.reset_index()
house_green['绿化率'] = house_green['绿化率'].replace("暂无资料",0)
house_green["绿化率"] = house_green['绿化率'].astype(float)
house_green["green_level"] = pd.cut(house_green['绿化率'],5,labels=[0,1,2,3,4])
green_label = np.array(house_green["green_level"].values)

#upload
average_emb = defaultdict()
poi_dis = poi_dis.reshape((poi_dis.shape[0],50000))
for ind in range(poi_dis.shape[0]):
    if green_label[ind] not in average_emb.keys():
        average_emb[str(green_label[ind])+"_count"] = 1
        average_emb[green_label[ind]] = poi_dis[ind]
    else:
        average_emb[green_label[ind]] = average_emb[green_label[ind]] + poi_dis[ind]
    average_emb[str(green_label[ind])+"_count"] += 1

green_mean_emb = defaultdict()
for i in range(5):
    green_mean_emb[i] = average_emb[i]/average_emb[str(i)+"_count"]
    print("green level :",i," max value: ",green_mean_emb[i].max())

#save the distribution of urban planning solution under different green level settings.
with open("./data/green_standards_50.pkl","wb") as f:
    pickle.dump(green_mean_emb,f)