import numpy as np
import tensorflow as tf


def read_data():
    env_rep = np.load("../data/env_embedding1.npz")["arr_0"]
    green_emb = np.load("../data/green_level_emb.npz")["arr_0"]
    con_emb = np.hstack([env_rep,green_emb])
    urban_sol = np.load("../data/100_poi_dis.npz")["arr_0"]
    func_zone = np.load("../data/func1_100.npz")["arr_0"]
    urban_sol = urban_sol.reshape((urban_sol.shape[0],-1))
    func_zone = func_zone.reshape((func_zone.shape[0],-1))
    return urban_sol, func_zone, con_emb

def generate_data_batch(urban_sol,func_zone,con_emb,batch_size,ratio):
    data_len = urban_sol.shape[0]
    urban_sol = tf.data.Dataset.from_tensor_slices(urban_sol)
    func_zone = tf.data.Dataset.from_tensor_slices(func_zone)
    con_emb = tf.data.Dataset.from_tensor_slices(con_emb)

    train_sol, test_sol = urban_sol.take(int(data_len*ratio)),urban_sol.skip(int(data_len*ratio))
    train_func, test_func = func_zone.take(int(data_len*ratio)),func_zone.skip(int(data_len*ratio))
    train_con, test_con = con_emb.take(int(data_len*ratio)),con_emb.skip(int(data_len*ratio))

    con_label = []
    for test_sample in test_con:
        con_label.append(np.argmax(test_sample.numpy()[-5:]))
    con_label = np.array(con_label)
    np.savez_compressed("../data/con_label.npz",con_label)

    train_sol = train_sol.batch(batch_size=batch_size)
    train_func = train_func.batch(batch_size=batch_size)
    train_con = train_con.batch(batch_size=batch_size)

    test_sol = test_sol.batch(batch_size=batch_size)
    test_func = test_func.batch(batch_size=batch_size)
    test_con = test_con.batch(batch_size=batch_size)

    train_dataset = tf.data.Dataset.zip((train_sol,train_func,train_con))
    test_dataset = tf.data.Dataset.zip((test_sol, test_func, test_con))

    return train_dataset,test_dataset




