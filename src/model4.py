import time
import cv2
import numpy as np
from scipy.misc import imread, imresize
import math
from itertools import izip
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, merge, UpSampling2D, core, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras import backend as K
from deepsense import neptune
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

from data_unet import train_generator, val_generator
# from data_unet_small import train_generator, val_generator


seed = 1
s = 128
channels = 20
batch = 16

prefix = "/mnt/ml-team/satellites/max.sokolowski/data/"

# Neptune
ctx = neptune.Context()

logs_channel = ctx.job.create_channel(
    name='logs',
    channel_type=neptune.ChannelType.TEXT)


train_image_channel = ctx.job.create_channel(
    name='train_image',
    channel_type=neptune.ChannelType.IMAGE)

val_image_channel = ctx.job.create_channel(
    name='val_image',
    channel_type=neptune.ChannelType.IMAGE)


def get_neptune_image(raw_image, epoch_number):
    neptune_image = Image.fromarray(raw_image)
    image_name = '(epoch {})'.format(epoch_number)
    return neptune.Image(
        name=image_name,
        description=u"",
        data=neptune_image)


def load_image(path):
    img = imread(path)
    img = imresize(img, [s, s])
    return img


train_gen = train_generator(2, s, s)
train_sample = train_gen.next()

val_gen = val_generator(2, s, s)
val_sample = val_gen.next()

X_train_model = train_sample[0]
y_train_model = train_sample[1]

X_train = train_sample[0][0]
y_train = train_sample[1][0]

X_val_model = val_sample[0]
y_val_model = val_sample[1]

X_val = val_sample[0][0]
y_val = val_sample[1][0]


def prepare_20_channels(img):
    img = np.copy(img)
    img = img.astype(np.float16)

    img = img[:, :, 0:3]  # first rbg

    img /= 3
    img += 1
    img *= 127.5
    img = img.clip(0, 255)
    img = np.round(img).astype(np.uint8)

    return img

X_train_prepared = prepare_20_channels(X_train)
X_val_prepared = prepare_20_channels(X_val)


class EpochEndCallback(Callback):
    def __init__(self):
        self.v_jaccard = 0

    def on_train_begin(self, logs={}):
        image = get_neptune_image(X_train_prepared, -1)
        train_image_channel.send(x=-1, y=image)
        image = get_neptune_image(y_train * 255, 0)
        train_image_channel.send(x=0, y=image)

        image = get_neptune_image(X_val_prepared, -1)
        val_image_channel.send(x=-1, y=image)
        image = get_neptune_image(y_val * 255, 0)
        val_image_channel.send(x=0, y=image)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            pred = self.model.predict(X_train_model)
            pred = np.array(pred[0]*255, dtype='uint8')
            image = get_neptune_image(pred, epoch)
            train_image_channel.send(x=epoch, y=image)

            pred = self.model.predict(X_val_model)
            pred = np.array(pred[0] * 255, dtype='uint8')
            image = get_neptune_image(pred, epoch)
            val_image_channel.send(x=epoch, y=image)

        if float(logs.get('val_jaccard_coef')) > self.v_jaccard:
            self.v_jaccard = float(logs.get('val_jaccard_coef'))
            action_save_model("max_val_jaccard")

        if epoch % 1000 == 0:
            action_save_model(str(epoch))


def action_save_model(name):
    model.save("model_" + str(name) + '.h5')
    return "model_" + str(name) + '.h5'


ctx.job.register_action(name='Save model', handler=action_save_model)


tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=True)
ctx.integrate_with_tensorflow()


def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred))


def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def binary_crossentropy_weight(target, output):
    class1 = 300
    class2 = 1
    smooth = 1e-12

    crossentropy = -K.sum(class1 * target * K.log(output + smooth) + class2 * (1.0 - target) * K.log(1.0 - output + smooth), axis=[0, 1, 2])

    return K.mean(crossentropy)


def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return LeakyReLU(alpha=.001)(
        BatchNormalization()(Convolution2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))
    # return Activation(activation='relu')(BatchNormalization()(Convolution2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))

def get_unet(n_ch, patch_height, patch_width):
    inputs = Input((patch_height, patch_width, n_ch))
    conv1 = Conv2DReluBatchNorm(32, 3, 3, inputs)
    conv1 = Conv2DReluBatchNorm(32, 3, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2DReluBatchNorm(64, 3, 3, pool1)
    conv2 = Conv2DReluBatchNorm(64, 3, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2DReluBatchNorm(128, 3, 3, pool2)
    conv3 = Conv2DReluBatchNorm(128, 3, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2DReluBatchNorm(256, 3, 3, pool3)
    conv4 = Conv2DReluBatchNorm(256, 3, 3, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2DReluBatchNorm(512, 3, 3, pool4)
    conv5 = Conv2DReluBatchNorm(512, 3, 3, conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2DReluBatchNorm(256, 3, 3, up6)
    conv6 = Conv2DReluBatchNorm(256, 3, 3, conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2DReluBatchNorm(128, 3, 3, up7)
    conv7 = Conv2DReluBatchNorm(128, 3, 3, conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2DReluBatchNorm(64, 3, 3, up8)
    conv8 = Conv2DReluBatchNorm(64, 3, 3, conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2DReluBatchNorm(32, 3, 3, up9)
    conv9 = Conv2DReluBatchNorm(32, 3, 3, conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=l2(0.01))(conv9)

    reshaped = core.Reshape((patch_height, patch_width))(conv10)
    # conv10 = core.Permute((2, 1))(conv10)

    model = Model(input=inputs, output=reshaped)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy', jaccard_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',
    #               metrics=['accuracy', jaccard_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss=jaccard_loss, metrics=['accuracy', jaccard_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=binary_crossentropy_weight, metrics=['accuracy', jaccard_coef])

    return model


def get_unet3(n_ch, patch_height, patch_width):

    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2DReluBatchNorm(32, 3, 3, inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2DReluBatchNorm(32, 3, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2DReluBatchNorm(64, 3, 3, pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2DReluBatchNorm(64, 3, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2DReluBatchNorm(128, 3, 3, pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2DReluBatchNorm(128, 3, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2DReluBatchNorm(256, 3, 3, pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2DReluBatchNorm(256, 3, 3, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2DReluBatchNorm(256, 3, 3, pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2DReluBatchNorm(256, 3, 3, conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2DReluBatchNorm(256, 3, 3, pool5)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2DReluBatchNorm(256, 3, 3, conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2DReluBatchNorm(256, 3, 3, pool6)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2DReluBatchNorm(256, 3, 3, conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv6], mode='concat', concat_axis=1)
    conv8 = Conv2DReluBatchNorm(256, 3, 3, up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2DReluBatchNorm(256, 3, 3, conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv5], mode='concat', concat_axis=1)
    conv9 = Conv2DReluBatchNorm(256, 3, 3, up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2DReluBatchNorm(256, 3, 3, conv9)

    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv4], mode='concat', concat_axis=1)
    conv10 = Conv2DReluBatchNorm(256, 3, 3, up10)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2DReluBatchNorm(256, 3, 3, conv10)

    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv3], mode='concat', concat_axis=1)
    conv11 = Conv2DReluBatchNorm(128, 3, 3, up11)
    conv11 = Dropout(0.2)(conv11)
    conv11 = Conv2DReluBatchNorm(128, 3, 3, conv11)

    up12 = merge([UpSampling2D(size=(2, 2))(conv11), conv2], mode='concat', concat_axis=1)
    conv12 = Conv2DReluBatchNorm(64, 3, 3, up12)
    conv12 = Dropout(0.2)(conv12)
    conv12 = Conv2DReluBatchNorm(64, 3, 3, conv12)

    up13 = merge([UpSampling2D(size=(2, 2))(conv12), conv1], mode='concat', concat_axis=1)
    conv13 = Conv2DReluBatchNorm(32, 3, 3, up13)
    conv13 = Dropout(0.2)(conv13)
    conv13 = Conv2DReluBatchNorm(32, 3, 3, conv13)

    conv14 = Convolution2D(1, 1, 1, activation='sigmoid')(conv13)
    # conv14 = core.Permute((2, 1))(conv14)

    model = Model(input=inputs, output=conv14)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss=jaccard_loss, metrics=['accuracy', jaccard_coef])

    return model


model = get_unet(channels, s, s)

path = "/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/miscSatellites/201702270812/0/jobs/945f9bcc-f0e2-4e8d-82e0-5ccaf1ec8c3b/src/model_6.h5"
model.load_weights(path, by_name=True)
# model = load_model(path)

history = model.fit_generator(
    train_generator(batch, s, s),
    samples_per_epoch=1024,
    nb_epoch=30000,
    validation_data=val_generator(batch, s, s),
    nb_val_samples=160,
    verbose=2,
    callbacks=[tensorboard, EpochEndCallback()])

model.save('model.h5')
