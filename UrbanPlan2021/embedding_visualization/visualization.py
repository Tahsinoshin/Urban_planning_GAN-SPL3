import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

urban_embedding = np.load("urban_embedding.npz")["arr_0"]
urban_label = np.load("green_label.npz")["arr_0"]
urban_label = urban_label.reshape((2691,1))
final_embedding = np.hstack([urban_embedding,urban_label])
final_embedding = pd.DataFrame(final_embedding)
final_embedding.to_csv("final_embedding.csv",index=False)
final_embedding[2] = final_embedding[2].astype(int)
final_embedding[2].replace(0,"green_0",inplace=True)
final_embedding[2].replace(1,"green_1",inplace=True)
final_embedding[2].replace(2,"green_2",inplace=True)
final_embedding[2].replace(3,"green_3",inplace=True)
final_embedding[2].replace(4,"green_4",inplace=True)
final_embedding.rename(columns={0:"x",1:"y",2:"label"},inplace=True)
sns.scatterplot(data=final_embedding,x="x",y="y",hue="label")
plt.savefig("embedding.jpg",dpi=600)
plt.show()