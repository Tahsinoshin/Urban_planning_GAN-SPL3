import sys
sys.path.append(".")
import numpy as np
from tensorflow.keras import layers
import time
import tensorflow as tf
import random
from functools import partial

def read_data():
    env_rep = np.load("../../../data/env_embedding1.npz")["arr_0"]
    green_emb = np.load("../../../data/green_level_emb.npz")["arr_0"]
    con_emb = np.hstack([env_rep,green_emb])
    urban_sol = np.load("../../../data/5_poi_dis.npz")["arr_0"]
    func_zone = np.load("../../../data/func1_5.npz")["arr_0"]
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
    np.savez_compressed("../../../data/con_label.npz",con_label)

    train_sol = train_sol.batch(batch_size=batch_size)
    train_func = train_func.batch(batch_size=batch_size)
    train_con = train_con.batch(batch_size=batch_size)

    test_sol = test_sol.batch(batch_size=batch_size)
    test_func = test_func.batch(batch_size=batch_size)
    test_con = test_con.batch(batch_size=batch_size)

    train_dataset = tf.data.Dataset.zip((train_sol,train_func,train_con))
    test_dataset = tf.data.Dataset.zip((test_sol, test_func, test_con))

    return train_dataset,test_dataset


#设计生成器generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*256, use_bias=False, input_shape=(10,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((5, 5, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (7, 7), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(20, (7, 7), strides=(1, 1), padding='same', use_bias=False,activation="tanh"))


    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[5, 5, 20]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def critic_loss(good_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(good_output)

# add generator loss
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

# Gradient Penalty (GP)
def gradient_penalty(generator, good_plan, generated_plan):
    batch_size = 1
    good_plan = tf.cast(good_plan, tf.float32)
    generated_plan = tf.cast(generated_plan, tf.float32)
    alpha = tf.random.uniform([batch_size, 5, 5, 20], 0., 1.)
    diff = generated_plan - good_plan
    inter = good_plan + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = generator(inter)
    gradients = tape.gradient(predictions, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((slopes - 1.) ** 2)

def train_step(good_plan,context_feature):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gp_weight = 10.

      generated_plan = generator(context_feature, training=True)

      good_output = discriminator(good_plan, training=True)
      fake_output = discriminator(generated_plan, training=True)
      cri_loss = critic_loss(good_output, fake_output)
      # add generator loss
      gen_loss = generator_loss(fake_output)

      gp = gradient_penalty(partial(discriminator, training=True),
                            good_plan, generated_plan)
      cri_loss += gp * gp_weight

    gradients_of_discriminator = disc_tape.gradient(cri_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

urban_sol,func_zone, con_emb = read_data()
train_dataset, test_dataset = generate_data_batch(urban_sol,func_zone,con_emb, 4 , 0.9)

#测试generator能否生成对应shape的规划方案
generator = make_generator_model()
#测试对应的discriminator的作用
discriminator = make_discriminator_model()

# cross_entropy损失为了后续的正负类的分类任务。
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#定义generator和discriminator的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#training phase
for epoch in range(1, 50 + 1):
  start_time = time.time()
  for batch_ind, data in enumerate(train_dataset):
      batch_urban = tf.reshape(data[0], (-1, 5, 5, 20))
      batch_env = data[2][:, :-5]
      train_step(batch_urban,batch_env)
  end_time = time.time()
  print('Epoch: {}, time elapse for current epoch: {}'
        .format(epoch, end_time - start_time))

result = []
for data in test_dataset:
    test_con = data[2][:,:-5]
    generated_sol = generator(test_con)
    result.append(np.maximum(generated_sol.numpy(),0.00001))

generate_us = result[0]
for i in range(1,len(result)):
    generate_us = np.vstack((generate_us,result[i]))
generate_us = generate_us.reshape((generate_us.shape[0], 5, 5, 20))
np.savez_compressed("../tmp/5_wgangp_generate_result.npz", generate_us)
print("WGAN-GP model is ended!")



