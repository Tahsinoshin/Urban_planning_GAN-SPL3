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
    model.add(layers.Dense(10*10*256, use_bias=False, input_shape=(10,)))
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
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[100, 100, 20]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

#add critic loss
def critic_loss(good_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(good_output)

#要让生成器生成的结果越优秀越好
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

def train_step(good_plan,context_feature):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_plan = generator(context_feature, training=True)
      good_output = discriminator(good_plan, training=True)
      fake_output = discriminator(generated_plan, training=True)

      # change discriminator loss to critic loss
      cri_loss = critic_loss(good_output, fake_output)
      gen_loss = generator_loss(fake_output)

    gradients_of_discriminator = disc_tape.gradient(cri_loss, discriminator.trainable_variables)

    # for WGAN model all the gradients should clip to (-0.01,0.01)
    for idx, grad in enumerate(gradients_of_discriminator):
        gradients_of_discriminator[idx] = tf.clip_by_value(grad, -0.01, 0.01)
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
      batch_urban = tf.reshape(data[0], (-1, 100, 100, 20))
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
generate_us = generate_us.reshape((generate_us.shape[0], 100, 100, 20))
np.savez_compressed("./tmp/wgan_generate_result.npz", generate_us)
print("WGAN model is ended!")

