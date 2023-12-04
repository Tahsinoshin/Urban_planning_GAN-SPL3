import pickle
import glob
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def calculate_POI_vector(poi_dis,dim,poi_cat):
    F_dict1 = []
    F_dict2 = []
    for bid, urban_solution in enumerate(poi_dis):
        F = []
        for m in range(dim):
            for n in range(dim):
                f_i = []
                for poi_category in range(poi_cat):
                    n_j = urban_solution[poi_category][m][n]
                    N_i = np.sum(urban_solution[:, m, n])
                    if N_i == 0:
                        N_i = 0.0001
                    R = dim * dim
                    n_r_j = np.sum(urban_solution[poi_category, :, :])
                    if n_r_j == 0:
                        n_r_j = 0.0001
                    v_ij = (n_j / N_i) * np.log(R / n_r_j)
                    f_i.append(v_ij)
                F.append(f_i)
        F = np.array(F)
        u1, _, _ = np.linalg.svd(F, full_matrices=True)
        u2, _, _ = np.linalg.svd(F, full_matrices=False)
        F_dict1.append(u1)
        F_dict2.append(u2)
        print("The ", bid, " residential community has been done!")

        if bid % 5 == 0:
            F1_d = np.array(F_dict1)
            F2_d = np.array(F_dict2)
            np.savez_compressed("./tmp/"+str(bid)+"_tf_idf_5.npz",f1_d=F1_d,f2_d=F2_d)
            print("The middle file has been saved!")
            F_dict1.clear()
            F_dict2.clear()
    if len(F_dict1) != 0 and F_dict2 != 0:
        F1_d = np.array(F_dict1)
        F2_d = np.array(F_dict2)
        np.savez_compressed("./tmp/" + str(len(poi_dis)) + "_tf_idf_5.npz", f1_d=F1_d, f2_d=F2_d)
        print("The middle file has been saved!")
        F_dict1.clear()
        F_dict2.clear()
    return

def find_func_zones_by_kmeans(path,n,save_path1,save_path2):
    files = glob.glob(path)
    files = sorted(files, key=lambda x: int(x.split('/')[2].split('_')[0]))
    f1_count = 0
    f2_count = 0
    f1_dict = defaultdict(list)
    f2_dict = defaultdict(list)
    for i, f in enumerate(files):
        F1_d = np.load(f)["f1_d"]
        F2_d = np.load(f)["f2_d"]

        for solution in F1_d:
            kmeans = KMeans(n_clusters=n, n_jobs=-1, precompute_distances=True, algorithm='full', n_init=1,
                            random_state=0).fit(solution)
            f1_dict[f1_count] = kmeans.labels_
            f1_count += 1
            print("In F1 the ", f1_count - 1, " solution. The class distribution is ",
                  np.sum(kmeans.labels_ == 0),
                  np.sum(kmeans.labels_ == 1),
                  np.sum(kmeans.labels_ == 2),
                  np.sum(kmeans.labels_ == 3),
                  np.sum(kmeans.labels_ == 4),
                  )

        for solution in F2_d:
            kmeans = KMeans(n_clusters=n, n_jobs=-1, precompute_distances=True, algorithm='full', n_init=1,
                            random_state=0).fit(solution)
            f2_dict[f2_count] = kmeans.labels_
            f2_count += 1
            print("In F2 the ", f2_count - 1, " solution. The class distribution is ",
                  np.sum(kmeans.labels_ == 0),
                  np.sum(kmeans.labels_ == 1),
                  np.sum(kmeans.labels_ == 2),
                  np.sum(kmeans.labels_ == 3),
                  np.sum(kmeans.labels_ == 4),
                  )

    with open(save_path1, "wb") as f:
        pickle.dump(f1_dict, f)
    with open(save_path2, "wb") as f:
        pickle.dump(f2_dict, f)

def convert_functional_zone_shape(data, dim):
    result = []
    for key, value in data.items():
        sample_tmp = []
        for i in range(dim):
            line_tmp = []
            for j in range(dim):
                line_tmp.append(value[i * dim + j])
            sample_tmp.append(line_tmp)
        result.append(np.array(sample_tmp))
    return np.array(result)


if __name__ == "__main__":
    poi_dis = np.load("./data/data/5_poi_dis.npz")["arr_0"]
    #calculate_POI_vector(poi_dis,5,20) # calculate poi vector using tf-idf value and svd algorithm
    find_func_zones_by_kmeans("./tmp/*idf_5.npz",5,"./tmp/f1_cluster_5.pkl","./tmp/f2_cluster_5.pkl") #find functional zone by kmeans algorithm

    with open("./tmp/f1_cluster_5.pkl", "rb") as f:
        f1_dict = pickle.load(f)
    with open("./tmp/f2_cluster_5.pkl", "rb") as f:
        f2_dict = pickle.load(f)
    f1_result = convert_functional_zone_shape(f1_dict, 5)
    f2_result = convert_functional_zone_shape(f2_dict, 5)
    np.savez_compressed("./data/func1_5.npz", f1_result)
    np.savez_compressed("./data/func2_5.npz", f2_result)