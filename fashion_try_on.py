import glob
import os
import time
from random import shuffle

import cv2
import keras

# import function from keras
import numpy as np
import tensorflow as tf
from keras import Function as function
from keras import Model
from keras.applications import *
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, Conv2D, LeakyReLU
from PIL import Image
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["KERAS_BACKEND"] = "tensorflow"


# Configuration
channel_axis = -1
channel_first = False

nc_in = 9
nc_out = 4
ngf = 64
ndf = 64
use_lsgan = False
use_nsgan = False  # Non-saturating GAN
A = 10 if use_lsgan else 100

# CAGAN config
nc_G_inp = 9  # [x_img,y_img,y_j]
nc_G_out = 4  # [alpha, x_i_j (RGB)
nc_D_inp = 6  # pos : [x_i,y_i]; neg1: [G_out(x_i),y_i; neg2: [x_i,y_j]
nc_D_out = 1
gamma_i = 0.2
use_instancenorm = True
loadSize = 143
image_size = 256
batchsize = 16
lrD = 2e-4
lrG = 2e-4

# Define models
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1, 0.02)


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a)  ##for convolution kernel
    k.conv_weight = True
    return k


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1, 0.02)  # for batch normalization


# Basic Discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return BatchNormalization(
        momentum=0.9, axis=channel_axis, epsilon=1.01e-5, gamma_initializer=gamma_init
    )


def instance_norm():
    return BatchNormalization(
        axis=channel_axis, epsilon=1.01e-5, gamma_initializer=gamma_init
    )


def BASIC_D(nc, ndf, max_layers=3, use_sigmoid=True):
    """
    DCGAN discriminator model.

    Args:
        nc (int): Number of input channels.
        ndf (int): Number of filters of the first layer.
        max_layers (int): Maximum number of hidden layers.
        use_sigmoid (bool): Whether to use sigmoid activation in the final layer.

    Returns:
        keras.Model: DCGAN discriminator model.
    """
    # Define input layer
    if channel_first:
        input_a = Input(shape=(nc, None, None))
    else:
        input_a = Input(shape=(None, None, nc))
    _ = input_a

    # First convolutional layer
    _ = Conv2D(ndf, kernel_size=4, strides=2, padding="same")(_)
    _ = LeakyReLU(negative_slope=0.2)(_)

    # Intermediate convolutional layers
    for layer in range(1, max_layers):
        out_feat = ndf * min(2**layer, 8)
        _ = Conv2D(out_feat, kernel_size=4, strides=2, padding="same", use_bias=False)(
            _
        )
        _ = BatchNormalization()(_, training=True)
        _ = LeakyReLU(negative_slope=0.2)(_)

    # Last convolutional layer
    out_feat = ndf * min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = Conv2D(out_feat, kernel_size=4, use_bias=False)(_)
    _ = BatchNormalization()(_, training=True)
    _ = LeakyReLU(negative_slope=0.2)(_)

    # Final layer
    _ = ZeroPadding2D(1)(_)
    _ = Conv2D(1, kernel_size=4, activation="sigmoid" if use_sigmoid else None)(_)
    # Define model
    return Model(inputs=[input_a], outputs=_)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True, use_batchnorm=True):
    s = isize if fixed_input_size else None
    inputs = Input(shape=(s, s, nc_in))
    x_i = Lambda(lambda x: x[:, :, :, 0:3], name="x_i")(inputs)
    y_i = Lambda(lambda x: x[:, :, :, 3:6], name="y_i")(inputs)
    xi_and_y_i = Concatenate(name="xi_yi")([x_i, y_i])
    xi_yi_s64 = AveragePooling2D(pool_size=2)(xi_and_y_i)
    xi_yi_s32 = AveragePooling2D(pool_size=4)(xi_and_y_i)
    xi_yi_s16 = AveragePooling2D(pool_size=8)(xi_and_y_i)
    xi_yi_s8 = AveragePooling2D(pool_size=16)(xi_and_y_i)
    layer1 = Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=not use_batchnorm,
        name="layer1",
    )(inputs)
    layer1 = LeakyReLU(negative_slope=0.2)(layer1)
    layer1 = Concatenate(axis=-1)([layer1, xi_yi_s64])
    layer2 = Conv2D(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=not use_batchnorm,
        name="layer2",
    )(layer1)
    if use_batchnorm:
        layer2 = BatchNormalization()(layer2, training=True)
    layer2 = LeakyReLU(negative_slope=0.2)(layer2)
    layer2 = Concatenate(axis=-1)([layer2, xi_yi_s32])
    layer3 = Conv2D(
        256,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=not use_batchnorm,
        name="layer3",
    )(layer2)
    if use_batchnorm:
        layer3 = BatchNormalization()(layer3, training=True)
    layer3 = LeakyReLU(negative_slope=0.2)(layer3)
    layer3 = Concatenate(axis=-1)([layer3, xi_yi_s16])
    layer4 = Conv2D(
        512,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=not use_batchnorm,
        name="layer4",
    )(layer3)
    if use_batchnorm:
        layer4 = BatchNormalization()(layer4, training=True)
    layer4 = LeakyReLU(negative_slope=0.2)(layer4)
    layer4 = Concatenate(axis=-1)([layer4, xi_yi_s8])
    layer9 = Conv2DTranspose(
        256,
        kernel_size=4,
        strides=2,
        use_bias=not use_batchnorm,
        kernel_initializer=RandomNormal(0, 0.02),
        name="layer9",
    )(layer4)
    layer9 = Cropping2D(((1, 1), (1, 1)))(layer9)
    if use_batchnorm:
        layer9 = BatchNormalization()(layer9, training=True)
    layer9 = Concatenate(axis=-1)([layer9, layer3])
    layer9 = Activation("relu")(layer9)
    layer9 = Concatenate(axis=-1)([layer9, xi_yi_s16])
    layer10 = Conv2DTranspose(
        128,
        kernel_size=4,
        strides=2,
        use_bias=not use_batchnorm,
        kernel_initializer=RandomNormal(0, 0.02),
        name="layer10",
    )(layer9)
    layer10 = Cropping2D(((1, 1), (1, 1)))(layer10)
    if use_batchnorm:
        layer10 = BatchNormalization()(layer10, training=True)
    layer10 = Concatenate(axis=-1)([layer10, layer2])
    layer10 = Activation("relu")(layer10)
    layer10 = Concatenate(axis=-1)([layer10, xi_yi_s32])
    layer11 = Conv2DTranspose(
        64,
        kernel_size=4,
        strides=2,
        use_bias=not use_batchnorm,
        kernel_initializer=RandomNormal(0, 0.02),
        name="layer11",
    )(layer10)
    layer11 = Cropping2D(((1, 1), (1, 1)))(layer11)
    if use_batchnorm:
        layer11 = BatchNormalization()(layer11, training=True)
    layer11 = Activation("relu")(layer11)
    layer12 = Concatenate(axis=-1)([layer11, xi_yi_s64])
    layer12 = Activation("relu")(layer12)
    layer12 = Conv2DTranspose(
        32,
        kernel_size=4,
        strides=2,
        use_bias=not use_batchnorm,
        kernel_initializer=RandomNormal(0, 0.02),
        name="layer12",
    )(layer12)
    layer12 = Cropping2D(((1, 1), (1, 1)))(layer12)
    if use_batchnorm:
        layer12 = BatchNormalization()(layer12, training=True)
    layer12 = Conv2D(
        4,
        kernel_size=4,
        strides=1,
        padding="same",
        use_bias=not use_batchnorm,
        name="out128",
    )(layer12)
    alpha = Lambda(lambda x: x[:, :, :, 0:1], name="alpha")(layer12)
    x_i_j = Lambda(lambda x: x[:, :, :, 1:], name="x_i_j")(layer12)
    alpha = Activation("sigmoid", name="alpha_sigmoid")(alpha)
    x_i_j = Activation("tanh", name="x_i_j_tanh")(x_i_j)
    out = Concatenate(axis=-1, name="out128_concat")([alpha, x_i_j])
    return Model(inputs=inputs, outputs=out)


netGA = UNET_G(image_size, nc_G_inp, nc_G_out, ngf)
netGA.summary()

netDA = BASIC_D(nc_D_inp, ndf, use_sigmoid=not use_lsgan)


def cycle_variables(netG1):
    """
    Params:
        x_i : human w/ cloth i shape = (128,96,3)
        y_i : stand alone cloth i, shape =(128,96,3)
        y_j : stand alone cloth i, shape =(128,96,3)
        alpha: mask for x_i_j, shape =(128,96,1)
        x_i_j: generated fake human swapping cloth i to j, shape =(128,96,3)

    Returns:
        real_input : concat[x_i,y_i,y_j], shape = (128,96,9)
        fake_output : masked_x_i_j = alpha*x_i_j + (1-alpha)*x_i,shape = (128,96,3)
        rec_input : output of the second generator (generating similar images to x_i), shape = (128,96,3)
        fn_generator : a path from input to G_out and cyclic G_out
    """
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]

    x_i = Lambda(lambda x: x[:, :, :, 0:3])(real_input)
    y_i = Lambda(lambda x: x[:, :, :, 3:6])(real_input)
    y_j = Lambda(lambda x: x[:, :, :, 6:])(real_input)

    alpha = Lambda(lambda x: x[:, :, :, 0:1])(fake_output)
    x_i_j = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

    fake_output = alpha * x_i_j + (1 - alpha) * x_i
    concat_input_G2 = concatenate([x_i, y_j, y_i], axis=-1)  # swap y_i and y_j
    rec_input = netG1([concat_input_G2])

    rec_alpha = Lambda(lambda x: x[:, :, :, 0:1])(rec_input)
    rec_x_i_j = Lambda(lambda x: x[:, :, :, 1:])(rec_input)

    rec_input = rec_alpha * rec_x_i_j + (1 - rec_alpha) * fake_output

    fn_generate = function([real_input], [fake_output, rec_input])

    return real_input, fake_output, rec_input, fn_generate, alpha


# Assuming netGA is defined elsewhere in your code
real_A, fake_B, rec_A, cycleA_generate, alpha_A = cycle_variables(netGA)


# Define discriminator loss function
@tf.function
def D_loss(netD, real, fake, rec, use_nsgan=False):
    x_i = real[..., 0:3]  # Slicing the channels from real inputs
    y_i = real[..., 3:6]  # Slicing the channels from real inputs
    y_j = real[..., 6:]  # Slicing the channels from real inputs
    x_i_j = fake  # Using fake inputs

    # Calculating outputs for different combinations of inputs
    output_real = netD(tf.concat([x_i, y_i], axis=-1))  # Positive sample
    output_fake = netD(tf.concat([x_i_j, y_j], axis=-1))  # Negative sample
    output_fake2 = netD(tf.concat([x_i, y_j], axis=-1))  # Negative sample 2

    # Calculating discriminator losses
    loss_D_real = tf.keras.losses.mean_squared_error(
        tf.ones_like(output_real), output_real
    )
    loss_D_fake = tf.keras.losses.mean_squared_error(
        tf.zeros_like(output_fake), output_fake
    )
    loss_D_fake2 = tf.keras.losses.mean_squared_error(
        tf.zeros_like(output_fake2), output_fake2
    )

    if not use_nsgan:
        loss_G = tf.keras.losses.mean_squared_error(
            tf.ones_like(output_fake), output_fake
        )
    else:
        loss_G = K.mean(K.log(output_fake))

    loss_D = loss_D_real + (loss_D_fake + loss_D_fake2)
    loss_cyc = tf.keras.losses.mean_absolute_error(rec, x_i)

    return loss_D, loss_G, loss_cyc


# Calculate losses
loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_B, rec_A)
loss_cyc = loss_cycA
loss_id = tf.reduce_mean(tf.abs(alpha_A))  # Loss of alpha
loss_G = loss_GA + 1 * (1 * loss_cyc + gamma_i * loss_id)
loss_D = loss_DA * 2  # Multiply by 2 as specified in the original code


# Get trainable weights
weightsD = netDA.trainable_weights
weightsG = netGA.trainable_weights


gradients = tf.GradientTape.gradient(loss_D, netDA.trainable_variables)
training_updates = keras.optimizers.legacy.Adam(
    learning_rate=lrD, beta_1=0.5
).apply_gradients(zip(gradients, netDA.trainable_variables))
netD_train = function([real_A], [loss_DA / 2], training_updates)
gradients = tf.GradientTape.gradient(loss_G, netGA.trainable_variables)
training_updates = keras.optimizers.legacy.Adam(
    learning_rate=lrG, beta_1=0.5
).apply_gradients(zip(gradients, netGA.trainable_variables))
netG_train = function([real_A], [loss_GA, loss_cyc], training_updates)

# # Load Image
#
# Filenames:
#
#     "./imgs/1/fileID_1.jpg" for human images.
#     "./imgs/5/fileID_5.jpg" for article images.

# In[17]:


isRGB = True
apply_da = False


# In[18]:


def load_data(file_pattern):
    """
    Load image files matching the given pattern.

    Args:
        file_pattern (str): File pattern to match.

    Returns:
        List: List of file paths matching the pattern.
    """
    return glob.glob(file_pattern)


def crop_img(img, large_size, small_size):
    # only apply DA to human images
    img_width = small_size[0]
    img_height = small_size[1]
    diff_size = (large_size[0] - small_size[0], large_size[1] - small_size[1])

    x_range = [i for i in range(diff_size[0])]
    y_range = [j for j in range(diff_size[1])]
    x0 = np.random.choice(x_range)
    y0 = np.random.choice(y_range)

    img = np.array(img)

    img = img[y0 : y0 + img_height, x0 : x0 + img_width, :]

    return img


def read_image(fn, fn_y_i=None):
    input_size = (image_size, image_size)
    cropped_size = (96, 128)

    # Load human picture
    im = cv2.imread(fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    im = cv2.resize(im, input_size, interpolation=cv2.INTER_LINEAR)  # Resize

    if apply_da:
        im = crop_img(im, input_size, cropped_size)

    arr = im.astype(np.float32) / 255 * 2 - 1
    img_x_i = arr

    if channel_first:
        img_x_i = np.moveaxis(img_x_i, 2, 0)

    # Load article picture y_i
    if fn_y_i is None:
        fn_y_i = os.path.join(
            data_dir, "5", os.path.basename(fn)
        )  # Replace "1" with "5" in the path

    im = cv2.imread(fn_y_i)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    im = cv2.resize(im, input_size, interpolation=cv2.INTER_LINEAR)  # Resize

    arr = im.astype(np.float32) / 255 * 2 - 1
    img_y_i = arr

    if channel_first:
        img_y_i = np.moveaxis(img_y_i, 2, 0)

    # Load article picture y_j randomly
    fn_y_j = np.random.choice(filenames_5)
    while fn_y_j == fn_y_i:
        fn_y_j = np.random.choice(filenames_5)

    im = cv2.imread(fn_y_j)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    im = cv2.resize(im, input_size, interpolation=cv2.INTER_LINEAR)  # Resize

    arr = im.astype(np.float32) / 255 * 2 - 1
    img_y_j = arr

    if np.random.randint(0, 2):
        img_y_j = img_y_j[:, ::-1]

    if channel_first:
        img_y_j = np.moveaxis(img_y_j, 2, 0)

    if np.random.randint(0, 2):
        img_x_i = img_x_i[:, ::-1]
        img_y_i = img_y_i[:, ::-1]

    img = np.concatenate([img_x_i, img_y_i, img_y_j], axis=-1)
    assert img.shape[-1] == 9

    return img


# Get filenames

# In[19]:

data_dir = "MVC_image_pairs_resize_new"
person_dir = "1"
clothes_dir = "5"

train_A = load_data(os.path.join(data_dir, person_dir, "*.jpg"))
filenames_1 = load_data(os.path.join(data_dir, person_dir, "*.jpg"))
filenames_5 = load_data(os.path.join(data_dir, clothes_dir, "*.jpg"))

assert len(train_A)


# ## Other utilities

# In[20]:


def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize

        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = [read_image(data[j]) for j in range(i, i + size)]
        i += size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, batchsize):
    batchA = minibatch(dataA, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A


# In[21]:


# from IPython.display import display
def showX(X, rows=1):
    assert X.shape[0] % rows == 0
    int_X = ((X + 1) / 2 * 255).clip(0, 255).astype("uint8")
    # print (int_X.shape)
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1, 3, image_size, image_size), 1, 3)
    else:
        if X.shape[-1] == 9:
            img_x_i = int_X[:, :, :, :3]
            img_y_i = int_X[:, :, :, 3:6]
            img_y_j = int_X[:, :, :, 6:9]
            int_X = np.concatenate([img_x_i, img_y_i, img_y_j], axis=1)
        else:
            int_X = int_X.reshape(-1, image_size, image_size, 3)
    int_X = (
        int_X.reshape(rows, -1, image_size, image_size, 3)
        .swapaxes(1, 2)
        .reshape(rows * image_size, -1, 3)
    )
    if not isRGB:
        int_X = cv2.cvtColor(int_X, cv2.COLOR_LAB2RGB)

    Image.fromarray(int_X).save(os.path.join("results_512", str(time.time()) + ".jpg"))
    # display(Image.fromarray(int_X))


# In[22]:


def showG(A):
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i : i + 1]]) for i in range(X.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]

    rA = G(cycleA_generate, A)
    arr = np.concatenate(
        [A[:, :, :, :3], A[:, :, :, 3:6], A[:, :, :, 6:9], rA[0], rA[1]]
    )
    showX(arr, 5)


# # Demo
#
# Show 8 results on the same target article.

# In[ ]:


def minibatch_demo(data, batchsize, fn_y_i=None):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = [read_image(data[j], fn_y_i) for j in range(i, i + size)]
        i += size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB_demo(dataA, batchsize, fn_y_i=None):
    batchA = minibatch_demo(dataA, batchsize, fn_y_i=fn_y_i)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A


# # Training
#
# Show results every 50 iterations.

# In[ ]:

if __name__ == "_main_":
    t0 = time.time()
    niter = 150
    gen_iterations = 0
    epoch = 0
    errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0

    display_iters = 1000
    train_batch = minibatchAB(train_A, batchsize)

    iteration_num = 500000

    # while epoch < niter:
    while gen_iterations < iteration_num:
        epoch, A = next(train_batch)
        errDA = netD_train([A])
        errDA_sum += errDA[0]

        # epoch, trainA, trainB = next(train_batch)
        errGA, errCyc = netG_train([A])
        errGA_sum += errGA
        errCyc_sum += errCyc
        gen_iterations += 1
        if gen_iterations % display_iters == 0:
            """
            if gen_iterations%(10*display_iters)==0: # clear_output every 500 iters
                clear_output()
            """
            print(
                "[%d/%d][%d] Loss_D: %f Loss_G: %f loss_cyc: %f"
                % (
                    epoch,
                    niter,
                    gen_iterations,
                    errDA_sum / display_iters,
                    errGA_sum / display_iters,
                    errCyc_sum / display_iters,
                ),
                time.time() - t0,
            )
            _, A = train_batch.send(4)
            showG(A)
            errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0
            netGA.save(os.path.join("models_512", "netG" + str(time.time()) + ".h5"))
            netDA.save(os.path.join("models_512", "netD" + str(time.time()) + ".h5"))

    # DEMO
    len_fn = len(filenames_5)
    assert len_fn > 0
    idx = np.random.randint(len_fn)
    fn = filenames_5[idx]

    demo_batch = minibatchAB_demo(train_A, batchsize, fn)
    epoch, A = next(demo_batch)

    _, A = demo_batch.send(8)
    showG(A)
