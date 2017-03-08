import h5py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, merge, UpSampling2D, core, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2
from PIL import Image
import data

from deepsense import neptune
#
ctx = neptune.Context()
train_image_channel = ctx.job.create_channel(
     name='train_image',
     channel_type=neptune.ChannelType.IMAGE)
s = 128

f = h5py.File("predictions.h5", "w")

LARGE_VEHICLES = 9
SMALL_VEHICLES = 10

hf = h5py.File('/mnt/ml-team/satellites/files/dstl-test-20channels_ver2.h5')
print("Read h5")
crop_width, crop_height = 128, 128


means = np.array([433.63702393, 470.16989136, 336.4604187, 505.84042358,
                  295.93945312, 336.45907593, 470.146698, 476.11135864,
                  433.62802124, 520.07617188, 692.80700684, 521.65710449,
                  4398.10644531, 4629.50585938, 4317.39990234, 3875.63598633,
                  3026.12573242, 2718.59057617, 2659.33374023, 2568.97021484], dtype=np.float16)
std = np.array([218.58804321, 172.68545532, 109.00450134, 166.98269653,
                40.36304474, 109.01099396, 172.72686768, 175.36598206,
                218.5912323, 158.89251709, 232.90725708, 142.55833435,
                1896.10668945, 2496.32739258, 2206.86889648, 2129.56494141,
                1881.29187012, 1666.98059082, 1668.1427002, 1703.44152832], dtype=np.float16)


def get_neptune_image(raw_image, epoch_number):
    neptune_image = Image.fromarray(raw_image)
    image_name = '(epoch {})'.format(epoch_number)
    return neptune.Image(
        name=image_name,
        description=u"",
        data=neptune_image)

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def binary_crossentropy_weight(target, output):
    class1 = 400
    class2 = 1
    smooth = 1e-12

    crossentropy = -K.sum(class1 * target * K.log(output + smooth) + class2 * (1.0 - target) * K.log(1.0 - output + smooth), axis=[0, 1, 2])

    return K.mean(crossentropy)

def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return Activation(activation='relu')(BatchNormalization()(Convolution2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))

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

    reshaped = core.Reshape((s, s))(conv10)
    # conv10 = core.Permute((2, 1))(conv10)

    model = Model(input=inputs, output=reshaped)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy', jaccard_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',
    #               metrics=['accuracy', jaccard_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss=jaccard_loss, metrics=['accuracy', jaccard_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=binary_crossentropy_weight, metrics=['accuracy', jaccard_coef])

    return model


def expand_crop(image_crop):
    crop = np.zeros((crop_width, crop_height, 20))
    image = np.asarray(image_crop)
    # print(crop.shape)
    # print(image.shape)

    image = image.astype(np.float16)
    image -= means
    image /= std

    crop[:image.shape[0], :image.shape[1]] = image

    # print(crop)

    return crop


def read_masks(path):
    masks = dict()
    hf = h5py.File(path)
    for ImageId in sorted(hf.keys()):
        masks[ImageId] = hf[ImageId].value
    hf.close()
    return masks


def read_hdf5(path):
    hf = h5py.File(path, 'r')
    def read(hf):
        if type(hf) == h5py.Dataset: return hf.value
        d = {}
        for key, value in hf.iteritems():
            d[key] = read(value)
        return d
    d = read(hf)
    hf.close()
    return d

def read_buildings():
    pred = dict()
    PROBA_PATH_1024 = '/mnt/ml-team/satellites/an/src-1-stage1/predictions/testprobaclassid1scale1024wclass3ret3BEST.h5'
    PROBA_PATH_2048 = '/mnt/ml-team/satellites/an/src-1-stage2/predictions/testprobaclassid1scale2048wclass3ret3BEST.h5'
    PROBA1024 = read_hdf5(PROBA_PATH_1024)
    PROBA2048 = read_hdf5(PROBA_PATH_2048)
    for ImageId in sorted(PROBA1024.keys()):
        proba1 = PROBA1024[ImageId].astype(np.uint16)
        proba2 = PROBA2048[ImageId].astype(np.uint16)
        pred[ImageId] = ((proba1 + proba2) >= 330).astype(np.uint8)
    return pred

def read_roads():
    pred = dict()
    ROADS = read_hdf5('/mnt/ml-team/satellites/an/src-3-stage1/predictions/testclassid3scale1024wclass3retBEST.h5')
    for ImageId in sorted(ROADS.keys()):
        pred[ImageId] = ROADS[ImageId].astype(np.uint8)
    return pred


BUILDINGS = read_buildings()
ROADS = read_roads()

print len(BUILDINGS)
print len(ROADS)

def generate_mask():
    model = get_unet(20, crop_width, crop_height)
    print("get_unet")

    path = "/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/smallVehiclesSatellites/201703021209/0/jobs/6e105506-17fb-4d0b-9f2d-e5e2b1438dea/src/model_max_val_jaccard.h5"
    model.load_weights(path, by_name=False)
    print("model loaded")

    MASK_WATER = read_masks("/mnt/ml-team/satellites/an/src-78-stage1/water-stage1-test-predictions.h5")



    # for image_id, image in data.test_data_iterator(3):
    #     print image_id, image.shape

    border = 6
    counter = 0
    # for key in hf.keys():
    for key, image in data.test_data_iterator():
        # image = hf[key].value
        print(key)

        image_width = image.shape[0]
        image_height = image.shape[1]

        mask = np.zeros((image_width, image_height))

        buildings = read_buildings()
        roads = read_roads()

        for i in xrange(0, image_width, crop_width-2*border):
            x = min(i, image_width - crop_width - 1)
            for j in xrange(0, image_height, crop_height-2*border):
                y = min(j, image_height - crop_height - 1)
                image_crop = image[x:x+crop_width, y:y+crop_height]
                crop = expand_crop(image_crop)
                # crop = image_crop
                crop = crop.reshape((1,) + crop.shape)

                pred = model.predict(crop)
                # print("Prediction")
                # print(i, j)
                cut_pred = pred[0]
                # print("Crop: ", np.count_nonzero(cut_pred))
                cut_pred = cut_pred[border:-border, border:-border]
                # print cut_pred.shape
                mask[x+border:x+crop_width-border, y+border:y+crop_height-border] = cut_pred

                # cut_pred = np.array(cut_pred * 255, dtype='uint8')
                # neptune_image = get_neptune_image(cut_pred, 1000*i+j)
                # train_image_channel.send(x=1000*i+j, y=neptune_image)
                # print(i, j)


        # print(mask)
        # mask = K.round(K.clip(mask, 0, 1))
        # cut_mask = np.array(mask * 255, dtype='uint8')
        # neptune_image = get_neptune_image(cut_mask, -1)
        # train_image_channel.send(x=-1, y=neptune_image)
        # print("Mask: ", np.count_nonzero(mask))
        # mask = (np.rint(mask-0.2)).astype('uint8')

        mask_water = MASK_WATER[key]
        mask = mask - mask_water
        mask = mask.clip(min=0)
        (width, height) = mask.shape

        extension_size = 100
        threshold = 0.5
        buildings_mask = buildings[key]
        extended_building_mask = np.ones((height + 2*extension_size, width + 2*extension_size))

        for x in xrange(height):
            for y in xrange(width):
                if buildings_mask[y, x] > threshold:
                    extended_building_mask[y:y+2*extension_size, x:x+2*extension_size] = 0

        extended_building_mask = extended_building_mask[extension_size:width+extension_size, extension_size:height+extension_size]
        mask = mask - extended_building_mask
        mask = mask.clip(min=0)

        # extension_size = 2
        # threshold = 0.5
        # road_mask = roads[key]
        # extended_road_mask = np.ones((height + 2 * extension_size, width + 2 * extension_size))
        #
        # for x in xrange(height):
        #     for y in xrange(width):
        #         if road_mask[y, x] > threshold:
        #             extended_road_mask[y:y + 2 * extension_size, x:x + 2 * extension_size] = 0
        #
        # extended_road_mask = extended_road_mask[extension_size:width + extension_size,
        #                          extension_size:height + extension_size]
        # mask = mask - extended_road_mask
        # mask = mask.clip(min=0)

        mask = (mask*255).astype('uint8')
        print("Mask int: ", np.count_nonzero(mask))

        f.create_dataset(key, data=mask)

        counter += 1
        # if counter > 3:
        #     break


    # import multipolygons
    # import converters
    # import submit_func
    #
    #
    # submit = dict()
    #
    # for key in f:
    #     print(key)
    #     mask = f[key].value
    #     print(np.count_nonzero(mask))
    #     polygon = converters.mask_to_multipolygon(mask)
    #     submit_func.set_prediction_polygon(submit, polygon, key, 10)
    #
    # submit_func.save_submit('submit170221_SmallVehicles.csv', submit)


if __name__ == "__main__":
    generate_mask()
