import numpy as np
import pandas as pd
from collections import defaultdict

poi_dis = np.load("data/100_poi_dis.npz")["arr_0"]
func_zone1 = np.load("data/func1_100.npz")["arr_0"]
func_zone2 = np.load("data/func2_100.npz")["arr_0"]
env_rep = np.load("data/env_embedding1.npz")["arr_0"]
house_sit = pd.read_excel("data/house_situation.xlsx")

# classify the green level of different residential community.
house_green = house_sit["绿化率"]
house_green = house_green.reset_index()
house_green['绿化率'] = house_green['绿化率'].replace("暂无资料",0)
house_green["绿化率"] = house_green['绿化率'].astype(float)
print("green level max value:",house_green['绿化率'].max())
print("green level max value:",house_green['绿化率'].min())
house_green["green_level"] = pd.cut(house_green['绿化率'],5,labels=[0,1,2,3,4])
green_emb = pd.get_dummies(house_green["green_level"]).values
np.savez_compressed("data/green_level_emb.npz",green_emb)
print("The green level distribution of all communities as follows: ",house_green.groupby('green_level')['index'].count())
