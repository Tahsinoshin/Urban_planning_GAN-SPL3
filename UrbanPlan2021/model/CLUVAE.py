from IPython import display
import numpy as np
import tensorflow as tf
import time


def min_max_scaler(data, min_value, max_value):
    return (data-min_value)/(max_value-min_value)

def inverse_min_max_scaler(data,min_value,max_value):
    return (max_value-min_value)*data + min_value

def read_data():
    env_rep = np.load("../data/env_embedding1.npz")["arr_0"]
    green_emb = np.load("../data/green_level_emb.npz")["arr_0"]
    con_emb = np.hstack([env_rep,green_emb])
    urban_sol = np.load("../data/100_poi_dis.npz")["arr_0"]
    func_zone = np.load("../data/func1_100.npz")["arr_0"]

    urban_sol = urban_sol.reshape((urban_sol.shape[0],-1))
    func_zone = func_zone.reshape((func_zone.shape[0],-1))

    #对数据进行最大最小化放缩
    min_val = urban_sol.min()
    max_val = urban_sol.max()
    # urban_sol = min_max_scaler(urban_sol,min_val,max_val)
    return urban_sol, func_zone, con_emb, min_val, max_val

def generate_data_batch(urban_sol,func_zone,con_emb,batch_size,ratio):

    data_len = urban_sol.shape[0]
    urban_sol = tf.data.Dataset.from_tensor_slices(urban_sol)
    func_zone = tf.data.Dataset.from_tensor_slices(func_zone)
    con_emb = tf.data.Dataset.from_tensor_slices(con_emb)

    train_sol, test_sol = urban_sol.take(int(data_len*ratio)),urban_sol.skip(int(data_len*ratio))
    train_func, test_func = func_zone.take(int(data_len*ratio)),func_zone.skip(int(data_len*ratio))
    train_con, test_con = con_emb.take(int(data_len*ratio)),con_emb.skip(int(data_len*ratio))

    train_sol = train_sol.batch(batch_size=batch_size)
    train_func = train_func.batch(batch_size=batch_size)
    train_con = train_con.batch(batch_size=batch_size)

    test_sol = test_sol.batch(batch_size=batch_size)
    test_func = test_func.batch(batch_size=batch_size)
    test_con = test_con.batch(batch_size=batch_size)

    train_dataset = tf.data.Dataset.zip((train_sol,train_func,train_con))
    test_dataset = tf.data.Dataset.zip((test_sol, test_func, test_con))

    return train_dataset,test_dataset

def decoder_model(latent_dim, input_dim, condition_dim):
    input_c = tf.keras.Input(shape=(latent_dim + condition_dim,))
    h_c = tf.keras.layers.Dense(units=2048)(input_c)
    x_rec = tf.keras.layers.Dense(units=input_dim - condition_dim,activation=tf.nn.relu)(h_c)
    model = tf.keras.Model(input_c,x_rec)
    return model

def func_class_model():
    input_data = tf.keras.Input(shape=(20,))
    h1 = tf.keras.layers.Dense(2048)(input_data)
    output_prob = tf.keras.layers.Dense(5,activation=tf.nn.softmax)(h1)
    model = tf.keras.Model(input_data,output_prob)
    return model

batch_size = 4
urban_sol,func_zone, con_emb, min_value, max_value = read_data()
train_dataset, test_dataset = generate_data_batch(urban_sol,func_zone,con_emb, batch_size, 0.9)

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_dim, condition_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = decoder_model(latent_dim, input_dim, condition_dim)
    self.func_class = func_class_model()


  @tf.function
  def sample(self, eps=None,apply_sigmoid=False):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=apply_sigmoid)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def func_cls(self,x):
      return self.func_class(x)

optimizer = tf.keras.optimizers.Adam(1e-4)


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
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_origin)
  # rec_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x_origin,x_logit)
  x_origin = tf.reshape(x_origin,(-1,20))
  func_rec = model.func_cls(x_origin)
  func_loss = tf.keras.losses.SparseCategoricalCrossentropy()(func_zone,func_rec)
  logpx_z = -tf.reduce_sum(cross_ent,axis=[0])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x) + 0.5 * func_loss


@tf.function
def train_step(model, x, func_zone, cond_dim, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x, func_zone, cond_dim)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 50
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
cond_dim = 15

model = CVAE(latent_dim,200015, cond_dim)

#training phase
for epoch in range(1, epochs + 1):
  start_time = time.time()
  for i,train_x in enumerate(train_dataset):
    plan_solution = tf.cast(train_x[0],tf.float32)
    cond_vec = tf.cast(train_x[2],tf.float32)
    x = tf.concat([plan_solution,cond_vec],1)
    func_zone = train_x[1]
    train_step(model, x, func_zone, cond_dim, optimizer)
  end_time = time.time()

  checkpoint_path = "./weights/epoch"+str(epoch)+".ckpt"
  model.save_weights(checkpoint_path)

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    plan_solution = tf.cast(test_x[0], tf.float32)
    cond_vec = tf.cast(test_x[2], tf.float32)
    x = tf.concat([plan_solution, cond_vec], 1)
    func_zone = test_x[1]
    loss(compute_loss(model, x, func_zone, cond_dim))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))

def generate_solutions(model, test_cond):
  z = tf.random.normal((test_cond.shape[0],2))
  z = tf.concat((z,test_cond),1)
  predictions = model.sample(z,apply_sigmoid=False)
  return predictions

for ind in range(1,epochs+1):
    model.load_weights('./weights/epoch'+str(ind)+'.ckpt')
    # testing phase
    result = []
    for test_x in test_dataset:
        generate_solution = generate_solutions(model,test_x[2])
        result.append(generate_solution.numpy())

    generate_us = result[0]
    for i in range(1,len(result)):
        generate_us = np.vstack((generate_us,result[i]))
    generate_us = np.array(generate_us)
    # result = inverse_min_max_scaler(generate_us,min_value,max_value)
    result = generate_us.reshape((generate_us.shape[0],100,100,20))
    np.savez_compressed("./tmp/cluvae_generate_result.npz",result)
    print('epoch:',ind)
    print("max value: ", result.max())
    print("min value: ", result.min())