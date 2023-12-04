from IPython import display
import numpy as np
import tensorflow as tf
import time
import argparse
from utils import read_data,generate_data_batch

def get_params():
    parser = argparse.ArgumentParser(description="TF for urban planning solution.")
    parser.add_argument("--epochs", type=int, default=2, help="train epochs.")
    parser.add_argument("--latent_dim", type=int, default=2, help="the dimension of latent embedding.")
    parser.add_argument("--cond_dim", type=int, default=15, help="the dimension of conditional vector.")
    parser.add_argument("--input_dim", type=int, default=200015, help="the dimension of conditional vector.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    args , _ = parser.parse_known_args()
    return args

#control the activation to output the solution. Only decoder "relu" for large value
def encoder_model(input_dim,latent_dim):
    input_e = tf.keras.Input(shape=(input_dim,))
    h_e = tf.keras.layers.Dense(units=2048,activation="tanh")(input_e)
    out_e = tf.keras.layers.Dense(latent_dim + latent_dim)(h_e)
    encoder = tf.keras.Model(input_e,out_e)
    return encoder

def decoder_model(latent_dim, input_dim, condition_dim):
    input_d = tf.keras.Input(shape=(latent_dim + condition_dim,))
    h_d = tf.keras.layers.Dense(units=2048,activation="tanh")(input_d)
    x_rec = tf.keras.layers.Dense(units=input_dim - condition_dim,activation="relu")(h_d)
    decoder = tf.keras.Model(input_d,x_rec)
    return decoder

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_dim, condition_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = encoder_model(input_dim,latent_dim)
    self.decoder = decoder_model(latent_dim, input_dim, condition_dim)


  @tf.function
  def sample(self, eps=None):
    return self.decode(eps)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    logits = self.decoder(z)
    return logits

  def func_constrain(self,x):
      return self.function_zoner(x)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x, func_zone, cond_dim):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  c_vec = x[:,-cond_dim:]
  z_c = tf.concat([z,c_vec],1)
  x_logit = model.decode(z_c)
  x_origin = x[:,:-cond_dim]
  rec_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x_origin,x_logit)
  logpx_z = -tf.reduce_sum(rec_loss,axis=[0]) #maximize this item equals minimize the reconstruction loss
  logpz = log_normal_pdf(z, 0., 0.) #make the distribution of z close to the normal distribution
  logqz_x = log_normal_pdf(z, mean, logvar) #make each z belong to the normal distribution with the mean value and variance value
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

# @tf.function
def train_step(model, x, func_zone, cond_dim, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x, func_zone, cond_dim)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_solutions(model, test_cond):
  z = tf.random.normal((test_cond.shape[0],2))
  z = tf.concat((z,test_cond),1)
  predictions = model.sample(z)
  return predictions

args = get_params()
urban_sol,func_zone, con_emb = read_data()
train_dataset, test_dataset = generate_data_batch(urban_sol,func_zone,con_emb, args.batch_size, 0.9)

model = CVAE(args.latent_dim,args.input_dim, args.cond_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)

#training phase
for epoch in range(1, args.epochs + 1):
  start_time = time.time()
  for i,train_x in enumerate(train_dataset):
    plan_solution = tf.cast(train_x[0],tf.float32)
    cond_vec = tf.cast(train_x[2],tf.float32)
    x = tf.concat([plan_solution,cond_vec],1)
    func_zone = train_x[1]
    train_step(model, x, func_zone, args.cond_dim, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    plan_solution = tf.cast(test_x[0], tf.float32)
    cond_vec = tf.cast(test_x[2], tf.float32)
    x = tf.concat([plan_solution, cond_vec], 1)
    func_zone = test_x[1]
    loss(compute_loss(model, x, func_zone, args.cond_dim))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))


result = []
for test_x in test_dataset:
    generate_solution = generate_solutions(model,test_x[2])
    result.append(generate_solution.numpy())

generate_us = result[0]
for i in range(1,len(result)):
    generate_us = np.vstack((generate_us,result[i]))
generate_us = np.array(generate_us)
result = generate_us.reshape((generate_us.shape[0],100,100,20))
np.savez_compressed("./tmp/cluvae-_generate_result.npz",result)
