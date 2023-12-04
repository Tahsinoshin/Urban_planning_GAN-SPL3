import sys
sys.path.append(".")
import numpy as np
from tensorflow.keras import layers
import time
import tensorflow as tf
from utils import read_data,generate_data_batch

#设计生成器generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(10*10*256, use_bias=False, input_shape=(15,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((10, 10, 256)))
    assert model.output_shape == (None, 10, 10, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 10, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 20, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (7, 7), strides=(5, 5), padding='same', use_bias=False))
    assert model.output_shape == (None, 100, 100, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(20, (7, 7), strides=(1, 1), padding='same', use_bias=False,activation="tanh"))
    assert model.output_shape == (None, 100, 100, 20)

    return model

def make_discriminator_model():
    input_d = tf.keras.Input(shape=(100,100,20))
    input_con = tf.keras.Input(shape=(15,))
    conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[100, 100, 20])(input_d)
    relu1 = tf.keras.layers.LeakyReLU()(conv1)
    relu1 = tf.keras.layers.Dropout(0.3)(relu1)

    conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(relu1)
    relu2 = tf.keras.layers.LeakyReLU()(conv2)
    relu2 = tf.keras.layers.Dropout(0.3)(relu2)

    fl1 = tf.keras.layers.Flatten()(relu2)
    con1 = tf.keras.layers.concatenate([fl1,input_con],axis=1)
    output = tf.keras.layers.Dense(1)(con1)
    dis_model = tf.keras.Model([input_d,input_con],output)
    return dis_model

#最大化优秀的规划方案的loss，对于差的规划方案和generated的规划方案都要减小
def discriminator_loss(good_output, fake_output):
    good_loss = cross_entropy(tf.ones_like(good_output), good_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = good_loss + fake_loss
    return total_loss

#要让生成器生成的结果越优秀越好
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

def train_step(good_plan,context_feature):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_plan = generator(context_feature, training=True)

      good_output = discriminator([good_plan,context_feature], training=True)
      fake_output = discriminator([generated_plan, context_feature], training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(good_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#测试generator能否生成对应shape的规划方案
generator = make_generator_model()
#测试对应的discriminator的作用
discriminator = make_discriminator_model()
# cross_entropy损失为了后续的正负类的分类任务。
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#定义generator和discriminator的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


urban_sol,func_zone, con_emb = read_data()
train_dataset, test_dataset = generate_data_batch(urban_sol,func_zone,con_emb, 4 , 0.9)

#training phase
for epoch in range(1, 50 + 1):
  start_time = time.time()
  for batch_ind, data in enumerate(train_dataset):
      batch_urban = tf.reshape(data[0], (-1, 100, 100, 20))
      batch_env = data[2]
      train_step(batch_urban,batch_env)
  end_time = time.time()
  print('Epoch: {}, time elapse for current epoch: {}'
        .format(epoch, end_time - start_time))

result = []
for data in test_dataset:
    test_con = data[2]
    generated_sol = generator(test_con)
    result.append(np.maximum(generated_sol.numpy(),0.00001))

generate_us = result[0]
for i in range(1,len(result)):
    generate_us = np.vstack((generate_us,result[i]))
generate_us = generate_us.reshape((generate_us.shape[0], 100, 100, 20))
np.savez_compressed("./tmp/cgan_generate_result.npz", generate_us)
print("CGAN model is ended!")