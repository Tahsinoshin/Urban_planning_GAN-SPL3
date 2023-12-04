import pickle
import glob
import numpy as np
from collections import defaultdict
from gensim.matutils import kullback_leibler
from scipy.stats import wasserstein_distance
from gensim.matutils import hellinger
from scipy.spatial import distance


def kl_divergence(y_true,y_pred):
    return kullback_leibler(y_true,y_pred)

def js_divergence(y_true,y_pred):
    return distance.jensenshannon(y_true,y_pred)

def cos_distance(y_true,y_pred):
    return distance.cosine(y_true,y_pred)

def wd_distance(y_true,y_predict):
    return wasserstein_distance(y_true,y_predict)

def hd_distance(y_true,y_predict):
    return hellinger(y_true,y_predict)

def evaluate_solution(standards,predicts):
    '''

    :param standards: {green_level:average_embedding} which indicates the distribution information of each green level of urban planning solution
    :param predict_solution: {green_level:predict_embedding} which indicates the distribution information of each green level of generated urban planning solution
    :return: [kl_dis, js_dis, wd_dis, hd_dis]
    kl_dis = \sum_i^N (kl(real,predict))/N
    js_dis = \sum_i^N (js(real,predict))/N
    wd_dis = \sum_i^N (wd(real,predict))/N
    hd_dis = \sum_i^N (hd(real,predict))/N
    '''
    eval_result = []
    for green_level, standard_solution in standards.items():
        predict_solution = predicts[green_level]
        kl_dis = kl_divergence(standard_solution,predict_solution)
        js_dis = js_divergence(standard_solution,predict_solution)
        wd_dis = wd_distance(standard_solution,predict_solution)
        hd_dis = hd_distance(standard_solution,predict_solution)
        cos_dis = cos_distance(standard_solution,predict_solution)
        eval_result.append([kl_dis,js_dis,wd_dis,hd_dis,cos_dis])
    eval_result = np.array(eval_result)
    eval_result = np.mean(eval_result,axis=0)
    return eval_result

def print_all_metrics(model_name, con_label, standards,predicts):
    #according to the green level to get the mean value of prediction.
    mappings = defaultdict(list)
    for ind, pred_solu in enumerate(predicts):
        mappings[con_label[ind]].append(pred_solu)

    #calculate the distance between prediction and golden-standard
    predict_dict = defaultdict(np.array)
    for green_level,preds in mappings.items():
        tmp_emb = np.array(preds).reshape(len(preds),-1)
        avg_emb = np.mean(tmp_emb,axis=0)
        avg_emb[avg_emb<=0] = 0.00001
        predict_dict[green_level] = avg_emb
    diverges = evaluate_solution(standards,predict_dict)
    print("The ",model_name,"kl_dis:",diverges[0],"js_dis:",diverges[1],"wd_dis:",diverges[2],"hd_dis:",diverges[3],"cos_dis:",diverges[4])


with open("../data/green_standards.pkl","rb") as f:
    green_s = pickle.load(f)
con_label = np.load("../data/con_label.npz")["arr_0"]
cluvae_data1 = np.load("./tmp/cluvae*_generate_result.npz")["arr_0"]
cluvae_data2 = np.load("./tmp/cluvae^_generate_result.npz")["arr_0"]
cluvae_data3 = np.load("./tmp/cluvae-_generate_result.npz")["arr_0"]
cluvae_data4 = np.load("./tmp/cluvae#_generate_result.npz")["arr_0"]
cluvae_data5 = np.load("./tmp/cluvae^_generate_result1.npz")["arr_0"]
cluvae_data6 = np.load("./tmp/cluvae^_generate_result2.npz")["arr_0"]
cluvae_data7 = np.load("./tmp/cluvae^_generate_result3.npz")["arr_0"]
cluvae_data8 = np.load("./tmp/cluvae^_generate_result4.npz")["arr_0"]


print_all_metrics("cluvae*",con_label,green_s,cluvae_data1)
print_all_metrics("cluvae^",con_label,green_s,cluvae_data2)
print_all_metrics("cluvae-",con_label,green_s,cluvae_data3)
print_all_metrics("cluvae#",con_label,green_s,cluvae_data4)
print_all_metrics("cluvae^1",con_label,green_s,cluvae_data5)
print_all_metrics("cluvae^2",con_label,green_s,cluvae_data6)
print_all_metrics("cluvae^3",con_label,green_s,cluvae_data7)
print_all_metrics("cluvae^4",con_label,green_s,cluvae_data8)

print("---------------------------------------------------")

cluvae_data9 = np.load("./tmp/cluvae_generate_result_20.npz")["arr_0"]
cluvae_data10 = np.load("./tmp/cluvae_generate_result_21.npz")["arr_0"]
cluvae_data11 = np.load("./tmp/cluvae_generate_result_22.npz")["arr_0"]
cluvae_data12 = np.load("./tmp/cluvae_generate_result_23.npz")["arr_0"]
cluvae_data13 = np.load("./tmp/cluvae_generate_result_24.npz")["arr_0"]



print_all_metrics("cluvae20",con_label,green_s,cluvae_data9)
print_all_metrics("cluvae21",con_label,green_s,cluvae_data10)
print_all_metrics("cluvae22",con_label,green_s,cluvae_data11)
print_all_metrics("cluvae23",con_label,green_s,cluvae_data12)
print_all_metrics("cluvae24",con_label,green_s,cluvae_data13)


