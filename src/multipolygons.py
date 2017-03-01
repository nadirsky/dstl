import shapely


def jaccard(multipolygon1, multipolygon2):
    try:
        intersection = multipolygon1.intersection(multipolygon2)
        union = multipolygon1.union(multipolygon2)
        return intersection.area / union.area
    except:
        # Problem calculating jaccard. Let's ignore it for now
        return None


def _read_grid_sizes():
    grid_sizes_path = '/mnt/ml-team/satellites/data/grid_sizes.csv'
    with open(grid_sizes_path) as r:
        gs = dict()
        header = r.readline()
        for line in r:
            image_id, x_max, y_min = line.split(',')
            gs[image_id] = dict()
            gs[image_id]['x_max'] = float(x_max)
            gs[image_id]['y_min'] = float(y_min)
    return gs


_grid_size_by_image_id = _read_grid_sizes()


def pixel_cords_to_world_cords(multipolygon, image_shape_pixels, image_id):
    y_min = _grid_size_by_image_id[image_id]['y_min']
    x_max = _grid_size_by_image_id[image_id]['x_max']

    # https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/details/evaluation
    # https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#data-processing-tutorial

    (W, H) = image_shape_pixels
    W, H = float(W), float(H)

    W_ = W * (W / (W + 1))
    H_ = H * (H / (H + 1))

    x_scaler = x_max / W_
    y_scaler = y_min / H_

    return shapely.affinity.scale(multipolygon, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def world_cords_to_pixel_cords(multipolygon, pixel_shape, image_id):
    y_min = _grid_size_by_image_id[image_id]['y_min']
    x_max = _grid_size_by_image_id[image_id]['x_max']

    # https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/details/evaluation
    # https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection#data-processing-tutorial

    (W, H) = pixel_shape
    W, H = float(W), float(H)

    W_ = W * (W / (W + 1))
    H_ = H * (H / (H + 1))

    x_scaler = W_ / x_max
    y_scaler = H_ / y_min

    return shapely.affinity.scale(multipolygon, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def drop_smallest_exteriors(multipolygon, polygon_area_cutoff):
    filtered_polygons = filter(lambda p1: p1.area > polygon_area_cutoff, multipolygon.geoms)
    return shapely.geometry.MultiPolygon(filtered_polygons)


def drop_interiors(multipolygon):
    polygons_without_interiors = [_drop_interiors_single_polygon(p) for p in multipolygon.geoms]
    return shapely.geometry.MultiPolygon(polygons_without_interiors)


def _drop_interiors_single_polygon(polygon):
    exterior = polygon.exterior
    without_interiors = shapely.geometry.Polygon(exterior)
    return without_interiors


# TODO Simplify
# smooth


def drop_smallest_interiors(multipolygon, interior_area_cutoff):
    polygons_without_interiors = [_drop_smallest_interiors_single_polygon(p, interior_area_cutoff) for p in
                                  multipolygon.geoms]
    return shapely.geometry.MultiPolygon(polygons_without_interiors)


def _drop_smallest_interiors_single_polygon(polygon, interior_area_cutoff):
    exterior = polygon.exterior
    interiors = polygon.interiors
    filtered_interiors = filter(lambda i: _area_of_interior(i) > interior_area_cutoff, interiors)
    return shapely.geometry.Polygon(exterior, filtered_interiors)


def _area_of_interior(interior):
    # interiors are Linear Rings, which by definition has no area
    interior_as_polygon = shapely.geometry.Polygon(interior)
    return interior_as_polygon.area
