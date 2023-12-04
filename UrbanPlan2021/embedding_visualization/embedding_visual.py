from IPython import display
import numpy as np
import tensorflow as tf
from utils import read_data,generate_data_batch
import time
import argparse

def get_params():
    parser = argparse.ArgumentParser(description="TF for urban planning solution.")
    parser.add_argument("--epochs", type=int, default=50, help="train epochs.")
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

def func_zone_classifier():
    input_data = tf.keras.Input(shape=(20,))
    h1 = tf.keras.layers.Dense(2048)(input_data)
    output_prob = tf.keras.layers.Dense(5,activation=tf.nn.softmax)(h1)
    model = tf.keras.Model(input_data,output_prob)
    return model


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_dim, condition_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = encoder_model(input_dim,latent_dim)
    self.decoder = decoder_model(latent_dim, input_dim, condition_dim)
    self.function_zoner = func_zone_classifier()


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
  x_origin = tf.reshape(x_origin,(-1,20))
  func_rec = model.func_constrain(x_origin)
  func_loss = tf.keras.losses.SparseCategoricalCrossentropy()(func_zone,func_rec)
  logpx_z = -tf.reduce_sum(rec_loss,axis=[0]) #maximize this item equals minimize the reconstruction loss
  logpz = log_normal_pdf(z, 0., 0.) #make the distribution of z close to the normal distribution
  logqz_x = log_normal_pdf(z, mean, logvar) #make each z belong to the normal distribution with the mean value and variance value
  return -tf.reduce_mean(logpx_z + logpz - logqz_x) + 0. * func_loss

@tf.function
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

checkpoint_path = "../model/weights/epoch50.ckpt"
model.load_weights(checkpoint_path)

result = []
for i, train_x in enumerate(train_dataset):
    plan_solution = tf.cast(train_x[0], tf.float32)
    cond_vec = tf.cast(train_x[2], tf.float32)
    x = tf.concat([plan_solution, cond_vec], 1)
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    result.append(z.numpy())

final = result[0]
for i in range(1,len(result)):
    final = np.vstack((final,result[i]))
np.savez_compressed("urban_embedding.npz",final)
print("The embedding has been collected completely!")
