import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.geometry
import shapely.wkt

import numpy as np


def mask_to_multipolygon(mask):
    # Returns polygon. Cordinates are pixels positions from masks
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


def multipolygon_to_mask(multipolygon, out_shape = None):
    if multipolygon.is_empty:
        return np.zeros((10, 10, 3))

    if out_shape is None:
        (_, _, width, height) = multipolygon.bounds
        out_shape = (int(width), int(height))

    return rasterio.features.rasterize(multipolygon, out_shape)

# TODO Handle cords

