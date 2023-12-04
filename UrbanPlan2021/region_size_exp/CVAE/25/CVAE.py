from IPython import display
import numpy as np
import tensorflow as tf
import time
import argparse

def read_data():
    env_rep = np.load("../../../data/env_embedding1.npz")["arr_0"]
    green_emb = np.load("../../../data/green_level_emb.npz")["arr_0"]
    con_emb = np.hstack([env_rep,green_emb])
    urban_sol = np.load("../../../data/25_poi_dis.npz")["arr_0"]
    func_zone = np.load("../../../data/func1_25.npz")["arr_0"]
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


def get_params():
    parser = argparse.ArgumentParser(description="TF for urban planning solution.")
    parser.add_argument("--epochs", type=int, default=50, help="train epochs.")
    parser.add_argument("--latent_dim", type=int, default=2, help="the dimension of latent embedding.")
    parser.add_argument("--cond_dim", type=int, default=15, help="the dimension of conditional vector.")
    parser.add_argument("--input_dim", type=int, default=12515, help="the dimension of conditional vector.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    args, _ = parser.parse_known_args()
    return args


def encoder_model(input_dim, latent_dim):
    input_e = tf.keras.Input(shape=(input_dim,))
    h_e = tf.keras.layers.Dense(units=2048)(input_e)
    out_e = tf.keras.layers.Dense(latent_dim + latent_dim)(h_e)
    encoder = tf.keras.Model(input_e, out_e)
    return encoder


def decoder_model(latent_dim, input_dim, condition_dim):
    input_d = tf.keras.Input(shape=(latent_dim + condition_dim,))
    h_d = tf.keras.layers.Dense(units=2048)(input_d)
    x_rec = tf.keras.layers.Dense(units=input_dim - condition_dim, activation="relu")(h_d)
    decoder = tf.keras.Model(input_d, x_rec)
    return decoder



class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, input_dim, condition_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder_model(input_dim, latent_dim)
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



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x, cond_dim):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    c_vec = x[:, -cond_dim:]
    z_c = tf.concat([z, c_vec], 1)
    x_logit = model.decode(z_c)
    x_origin = x[:, :-cond_dim]
    rec_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x_origin, x_logit)
    logpx_z = -tf.reduce_sum(rec_loss, axis=[0])  # maximize this item equals minimize the reconstruction loss
    logpz = log_normal_pdf(z, 0., 0.)  # make the distribution of z close to the normal distribution
    logqz_x = log_normal_pdf(z, mean, logvar)  # make each z belong to the normal distribution with the mean value and variance value
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, cond_dim, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x,  cond_dim)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_solutions(model, test_cond):
    z = tf.random.normal((test_cond.shape[0], 2))
    z = tf.concat((z, test_cond), 1)
    predictions = model.sample(z)
    return predictions


args = get_params()
urban_sol, func_zone, con_emb = read_data()
train_dataset, test_dataset = generate_data_batch(urban_sol, func_zone, con_emb, args.batch_size, 0.9)

model = CVAE(args.latent_dim, args.input_dim, args.cond_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)

# training phase
for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    for i, train_x in enumerate(train_dataset):
        plan_solution = tf.cast(train_x[0], tf.float32)
        cond_vec = tf.cast(train_x[2], tf.float32)
        x = tf.concat([plan_solution, cond_vec], 1)
        train_step(model, x, args.cond_dim, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        plan_solution = tf.cast(test_x[0], tf.float32)
        cond_vec = tf.cast(test_x[2], tf.float32)
        x = tf.concat([plan_solution, cond_vec], 1)
        loss(compute_loss(model, x, args.cond_dim))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))

# testing phase
result = []
for test_x in test_dataset:
    generate_solution = generate_solutions(model, test_x[2])
    result.append(generate_solution.numpy())

generate_us = result[0]
for i in range(1, len(result)):
    generate_us = np.vstack((generate_us, result[i]))
generate_us = np.array(generate_us)
result = generate_us.reshape((generate_us.shape[0], 25, 25, 20))
np.savez_compressed("../tmp/25_cvae_generate_result.npz", result)
print("CVAE model is ended!")