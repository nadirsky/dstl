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

def convert_masks_to_polygons(MASKS, GS, classid):
    MP = dict()
    for ImageId in sorted(MASKS.keys()):
        mask = MASKS[ImageId]
        threshold = 0.7
        mask[mask < threshold * 255] = 0
        mask = remove_small_objects(mask.astype(bool), min_size=2).astype(np.uint8)
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
MASKS = read_masks('/mnt/ml-team/satellites/max.sokolowski/satellites/maksymilian.sokolowski@codilime.com/smallVehiclesSatellites/201703061752/0/jobs/5a042f25-96de-46eb-85a9-13aa5fb37925/src/predictions.h5')
MP = convert_masks_to_polygons(MASKS, GS, 10)
submit,order = create_empty_submit('/mnt/ml-team/satellites/data/sample_submission.csv')
submit = add_predictions(submit, MP, 10)

# rounding_precision=6
save_submit('/home/maksymilian/Desktop/submit201703061752_10Small_0.csv', submit, order)
