"""
import numpy as np
import h5py
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.affinity
import shapely.wkt


def mask_to_polygons(mask):
    all_polygons=[]
    for shape, value in rasterio.features.shapes(mask,mask,connectivity=4,
                                transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def convert_mask_to_multipolygon(mask, Xmax, Ymin):
    assert len(mask.shape) == 2
    height, width = mask.shape
    print height, width
    dh = Ymin/float(height*height)*(height+1)
    dw = Xmax/float(width*width)*(width+1)
    mp = mask_to_polygons(mask)
    mp = shapely.affinity.scale(mp, xfact = dw, yfact= dh, origin=(0,0,0))
    return mp


def read_grid_sizes():
    with open('/mnt/ml-team/satellites/data/grid_sizes.csv') as r:
        gs = dict()
        header = r.readline()
        for line in r:
            ImageId, Xmax, Ymin = line.split(',')
            gs[ImageId] = dict()
            gs[ImageId]['Xmax'] = float(Xmax)
            gs[ImageId]['Ymin'] = float(Ymin)
    return gs


def read_submit(path):
    with open(path) as r:
        moved_from_test_to_train = {'6010_1_2', '6040_4_4', '6070_2_3', '6100_2_2'}
        submit = dict()
        header = r.readline()
        for line in r:
            ImageId,classid = line.split(',')[:2]
            assert line[10] == ',' or line[11] == ','
            prevmp = ''
            if line[10] == ',': prevmp = line[11:].rstrip()
            if line[11] == ',': prevmp = line[12:].rstrip()
            if ImageId in moved_from_test_to_train: prevmp = "MULTIPOLYGON EMPTY"
            if ImageId not in submit: submit[ImageId] = dict()
            submit[ImageId][int(classid)] = prevmp
    return submit


def make_predictions(path, submit, classid):
    gs = read_grid_sizes()
    hf = h5py.File(path)
    for i,ImageId in enumerate(hf.keys(),1):
        print(hf[ImageId].shape)
        mp = convert_mask_to_multipolygon(hf[ImageId].value.squeeze(),gs[ImageId]['Xmax'],gs[ImageId]['Ymin'])
        mp = mp.simplify(tolerance=1e-7,preserve_topology=True)
        mp = '"{}"'.format(shapely.wkt.dumps(mp,rounding_precision=7))
        if mp == '"GEOMETRYCOLLECTION EMPTY"': mp = '"MULTIPOLYGON EMPTY"'
        if classid != 10: mp = '"MULTIPOLYGON EMPTY"'
        print classid, i, ImageId, len(mp)
        submit[ImageId][classid] = '{}'.format(mp)
    hf.close()
    return submit


def save_submit(filename, submit):
    def read_ss_order():
        with open('/mnt/ml-team/satellites/data/sample_submission.csv') as r:
            ss = list()
            header = r.readline()
            for line in r:
                ImageId = line.split(',')[0]
                if ImageId not in ss: ss.append(ImageId)
        return header,ss
    header, ss = read_ss_order()
    with open(filename, 'w') as w:
        w.write(header)
        for ImageId in ss:
            for classid in sorted(submit[ImageId]):
                line = '{},{},{}\n'.format(ImageId,classid,submit[ImageId][classid])
                w.write(line)

submit = read_submit('/mnt/ml-team/satellites/data/sample_submission.csv')
# submit = make_predictions('predictions/testclassid1scale1024wclass1.h5',submit, 1)
# submit = make_predictions('predictions/testclassid2scale1024wclass3.h5',submit, 2)
# submit = make_predictions('predictions/testclassid3scale512wclass1.h5',submit, 3)
# submit = make_predictions('predictions/testclassid4scale512wclass3.h5',submit, 4)
# submit = make_predictions('predictions/testclassid5scale1024wclass3.h5',submit, 5)
# submit = make_predictions('predictions/testclassid6scale512wclass1.h5',submit, 6)
# submit = make_predictions('predictions/testclassid7scale512wclass3.h5',submit, 7)
# submit = make_predictions('predictions/testclassid8scale512wclass3.h5',submit, 8)
# submit = make_predictions('predictions/testclassid9scale512wclass10.h5',submit, 9)
# submit = make_predictions('/home/maksymilian/Desktop/testclassid910wclass9retBEST.h5',submit, 10)
submit = make_predictions('/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/vehiclesSatellites/201702220929/0/jobs/02608406-7eef-4398-8452-b8489ab70202/src/mytestfile.h5',submit, 10)
save_submit('/home/maksymilian/Desktop/submit201702220929_SmallVehicles_S_AREK.csv', submit)"""


import numpy as np
import h5py
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.affinity
import shapely.wkt
from skimage.morphology import remove_small_objects, remove_small_holes
import csv

def read_masks(path):
    masks = dict()
    hf = h5py.File(path)
    for ImageId in sorted(hf.keys()):
        masks[ImageId] = hf[ImageId].value
    hf.close()
    return masks

def read_grid_sizes(path):
    gs = dict()
    with open(path) as r:
        header = r.readline()
        for line in r:
            ImageId, Xmax, Ymin = line.split(',')
            gs[ImageId] = dict()
            gs[ImageId]['Xmax'] = float(Xmax)
            gs[ImageId]['Ymin'] = float(Ymin)
    return gs

def make_valid(mp):
    while not mp.is_valid:
        mp = mp.buffer(0)
        if mp.type == 'Polygon':
            mp = shapely.geometry.MultiPolygon([mp])
    return mp

def make_valid_string(mp, rounding_precision):
    mp = shapely.wkt.dumps(mp, rounding_precision=rounding_precision)
    mp2 = shapely.wkt.loads(mp)
    while not mp2.is_valid:
        mp = make_valid(mp2)
        mp = shapely.wkt.dumps(mp, rounding_precision=rounding_precision)
        mp2 = shapely.wkt.loads(mp)
    return mp

def mask_to_polygons_via_shapely(mask):
    all_polygons=[]
    for shape, value in rasterio.features.shapes(mask,mask,connectivity=4, transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))
    mp = shapely.geometry.MultiPolygon(all_polygons)
    mp = make_valid(mp)
    return mp

def rescale_polygons(mp, mask, Xmax, Ymin):
    assert len(mask.shape) == 2
    height, width = mask.shape
    dh = Ymin/float(height*height)*(height+1)
    dw = Xmax/float(width*width)*(width+1)
    mp = shapely.affinity.scale(mp, xfact = dw, yfact= dh, origin=(0,0,0))
    return mp

def convert_masks_to_polygons(MASKS_LS, MASKS_S, GS, classid):
    MP = dict()
    for ImageId in sorted(MASKS_LS.keys()):
        mask_LS = MASKS_LS[ImageId]
        mask_S = MASKS_S[ImageId]
        mask = mask_LS - mask_S

        mask = mask.clip(min=0)

        threshold = 0.6
        mask[mask < threshold * 255] = 0
        mask = remove_small_objects(mask.astype(bool), min_size=8).astype(np.uint8)
        # mask = remove_small_holes(mask.astype(bool), min_size=128).astype(np.uint8)
        mp = mask_to_polygons_via_shapely(mask)
        mp = rescale_polygons(mp, mask, GS[ImageId]['Xmax'], GS[ImageId]['Ymin'])
        mp = mp.simplify(tolerance=1e-5, preserve_topology=True)
        mp = make_valid_string(mp, rounding_precision=6)
        MP[(ImageId,classid)] = mp
        print ImageId, classid, len(mp)
    return MP

def create_empty_submit(path):
    f = open(path)
    g = csv.reader(f)
    submit = dict()
    order = []
    moved_from_test_to_train = {'6010_1_2', '6040_4_4', '6070_2_3', '6100_2_2'}
    header = g.next()
    for line in g:
        ImageId,classid,mp = line
        submit[(ImageId,int(classid))] = 'MULTIPOLYGON EMPTY'
        order.append((ImageId,int(classid)))
    f.close()
    return submit,order

def add_predictions(submit, MP, classid):
    for key in MP:
        submit[key] = MP[key]
    return submit

def save_submit(name, submit, order):
    f = open(name,'w')
    g = csv.writer(f)
    g.writerow(['ImageId','ClassType','MultipolygonWKT'])
    for key in order:
        g.writerow(key + (submit[key],))
    f.close()

GS = read_grid_sizes('/mnt/ml-team/satellites/data/grid_sizes.csv')
MASKS_LS = read_masks('/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/miscSatellites/201702281155/0/jobs/dc41555c-7a2b-4488-87de-4f488d33c445/src/predictions.h5')
MASKS_S = read_masks('/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/miscSatellites/201702281016/0/jobs/d3862e05-41e2-40a4-bcf1-4d46c74b5642/src/predictions.h5')

MP = convert_masks_to_polygons(MASKS_LS, MASKS_S, GS, 9)
submit,order = create_empty_submit('/mnt/ml-team/satellites/data/sample_submission.csv')
submit = add_predictions(submit, MP, 9)

# rounding_precision=6
save_submit('/home/maksymilian/Desktop/submit201702281155_201702281016_09Large_0.csv',submit,order)
