import cv2
import numpy as np
import pandas as pd
import subprocess
import h5py
import os
# -------------------------------------------------------------------------------------------------
HDF5FILE = '/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/vehiclesSatellites/201702221600/0/jobs/ae62e507-4f65-488f-8f81-25c3e3b325f2/src/predictions.h5'
OUTDIR = 'test_contours'
# -------------------------------------------------------------------------------------------------
KEYS = {
'6020_0_0','6020_0_1','6020_0_2','6020_0_3','6020_0_4','6020_1_3','6020_1_4','6020_3_3','6020_3_4',
'6030_0_3','6030_1_0','6030_4_4','6050_0_0','6050_1_0','6050_4_3','6050_4_4','6060_0_1','6060_0_3',
'6060_1_2','6060_2_0','6060_2_1','6060_3_1','6070_0_1','6070_2_1','6070_3_3','6070_3_4','6080_0_3',
'6080_3_4','6080_4_0','6080_4_2','6080_4_3','6090_1_2','6090_1_3','6090_2_1','6090_2_2','6090_2_3',
'6090_2_4','6090_3_0','6100_0_1','6100_0_3','6100_1_0','6100_1_1','6100_2_0','6100_4_1','6110_0_1',
'6110_0_2','6110_1_1','6110_3_3','6110_4_1','6120_1_4','6120_3_1','6120_3_2','6130_0_2','6130_0_4',
'6130_1_3','6130_3_4','6130_4_2','6140_0_2','6140_1_1','6150_0_4','6150_2_1','6150_2_2','6150_3_0',
'6150_3_3','6140_1_0',
}
# -------------------------------------------------------------------------------------------------
def execute(command):
    print ">> {}".format(command)
    err = subprocess.call(command, shell=True)
 
def read_from_bash(command):
    print ">> {}".format(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out = p.communicate()[0]
    return out

def read_jpg(ImageId):
    return cv2.imread('/mnt/ml-team/satellites/files/three_bands_in_jpg/3test{}.jpg'.format(ImageId))[:,:,[2,1,0]]

def save_png(filename, img):
    cv2.imwrite('{}.png'.format(filename),img[:,:,[2,1,0]])

def print_image(ImageId):
    original = read_jpg(ImageId)
    pred = PRED[ImageId]
    edges = cv2.Canny(np.uint8(pred*255), 0, 1)
    whereedges = edges>0
    cv2.dilate(src=pred,dst=pred,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(3,3)),iterations=60)
    original[:,:,0] = original[:,:,0]*pred
    original[:,:,1] = original[:,:,1]*pred
    original[:,:,2] = original[:,:,2]*pred
    original[:,:,0][whereedges] = edges[whereedges]
    original[:,:,1][whereedges] = edges[whereedges]*0
    original[:,:,2][whereedges] = edges[whereedges]*0
    save_png('{}/{}'.format(OUTDIR,ImageId), original)      


# -------------------------------------------------------------------------------------------------
hf = h5py.File(HDF5FILE)
test_keys = sorted(hf.keys())
PRED = dict()
for ImageId in test_keys:
    if ImageId in KEYS:
        PRED[ImageId] = hf[ImageId].value.clip(0,1)
hf.close()
if os.path.exists(OUTDIR): execute('rm -rf {}'.format(OUTDIR))
execute('mkdir -p {}'.format(OUTDIR))

for ImageId in sorted(PRED.keys()):
    print_image(ImageId)
