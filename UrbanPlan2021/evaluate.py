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


with open("./data/green_standards.pkl","rb") as f:
    green_s = pickle.load(f)
con_label = np.load("./data/con_label.npz")["arr_0"]
our_data = np.load("./model/tmp/cluvae_generate_result_50.npz")["arr_0"]
cvae_data = np.load("./model/tmp/cvae_generate_result.npz")["arr_0"]
dcgan_data = np.load("./model/tmp/dcgan_generate_result.npz")["arr_0"]
wgan_data = np.load("./model/tmp/wgan_generate_result.npz")["arr_0"]
wgan_gp_data = np.load("./model/tmp/wgangp_generate_result.npz")["arr_0"]
cgan_data = np.load("./model/tmp/cgan_generate_result.npz")["arr_0"]
lucgan_data = np.load("./model/tmp/lucgan_generate_result.npz")["arr_0"]

print_all_metrics("cluvae",con_label,green_s,our_data)
print_all_metrics("cvae",con_label,green_s,cvae_data)
print_all_metrics("dcgan",con_label,green_s,dcgan_data)
print_all_metrics("wgan",con_label,green_s,wgan_data)
print_all_metrics("wgan-gp",con_label,green_s,wgan_gp_data)
print_all_metrics("cgan",con_label,green_s,cgan_data)
print_all_metrics("luccgan",con_label,green_s,lucgan_data)




