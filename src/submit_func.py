import csv
import sys

import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkt

csv.field_size_limit(sys.maxsize)


def load_from_csv(path):
    submit = dict()
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        next(reader, None)  # skip the headers

        for row in reader:
            image_id = row[0]
            class_id = int(row[1])
            multipolygon_string = row[2]

            multipolygon = shapely.wkt.loads(multipolygon_string)
            if not multipolygon.is_valid:
                multipolygon = _fix_multipolygon(multipolygon)

            submit[(image_id, class_id)] = multipolygon
    return submit


def set_prediction_polygon(submit, multipolygon, image_id, class_id):
    submit[(image_id, class_id)] = multipolygon


def _multipolygon_to_submit_string(multipolygon):
    multipolygon_string = shapely.wkt.dumps(multipolygon, rounding_precision=7)

    loaded_from_string_to_make_sure_its_valid = shapely.wkt.loads(multipolygon_string)
    while not loaded_from_string_to_make_sure_its_valid.is_valid:
        print 'Reloaded multipolygon from string is still invalid. Applying buffer fix again...'
        fixed_again = _fix_multipolygon(loaded_from_string_to_make_sure_its_valid)
        multipolygon_string = shapely.wkt.dumps(fixed_again, rounding_precision=7)
        loaded_from_string_to_make_sure_its_valid = shapely.wkt.loads(multipolygon_string)

    if multipolygon_string == 'GEOMETRYCOLLECTION EMPTY': multipolygon_string = 'MULTIPOLYGON EMPTY'
    return multipolygon_string


def _fix_multipolygon(multipolygon):
    "Fixes overlapping polygons and linear ring crossings"
    multipolygon = shapely.ops.cascaded_union(multipolygon)
    multipolygon = multipolygon.buffer(0)
    # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    # need to keep it a Multi throughout
    if multipolygon.type == 'Polygon':
        multipolygon = shapely.geometry.MultiPolygon([multipolygon])
    return multipolygon


def _image_id_order_from_sample_submission_file():
    with open('/mnt/ml-team/satellites/data/sample_submission.csv') as file:
        reader = csv.reader(file, delimiter=',')

        next(reader, None)  # skip the headers

        image_id_order = []

        for row in reader:
            image_id = row[0]

            if image_id not in image_id_order:
                image_id_order.append(image_id)

    return image_id_order

# TODO Wczytac i zapisac jako jest

_header = ['ImageId', 'ClassType', 'MultipolygonWKT']


def save_submit(filename, submit):
    image_id_order = _image_id_order_from_sample_submission_file()

    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(_header)

        for image_id in image_id_order:
            for class_id in range(1, 11):
                print 'Writing polygon for image_id: {}, class_id: {}'.format(str(image_id), str(class_id))

                key = (image_id, class_id)
                if key in submit:
                    multipolygon = submit[key]
                    multipolygon_string = _multipolygon_to_submit_string(multipolygon)
                    row = [image_id, class_id, multipolygon_string]
                    writer.writerow(row)
                else:
                    multipolygon_string = 'MULTIPOLYGON EMPTY'
                    row = [image_id, class_id, multipolygon_string]
                    writer.writerow(row)

# submit = read_submit('/mnt/storage_codi/dstl/data/sample_submission.csv')
# submit = make_predictions('predictions/testclassid1scale1024wclass1.h5',submit, 1)
# submit = make_predictions('predictions/testclassid2scale1024wclass3.h5',submit, 2)
# submit = make_predictions('predictions/testclassid3scale512wclass1.h5',submit, 3)
# submit = make_predictions('predictions/testclassid4scale512wclass3.h5',submit, 4)
# submit = make_predictions('predictions/testclassid5scale1024wclass3.h5',submit, 5)
# submit = make_predictions('predictions/testclassid6scale512wclass1.h5',submit, 6)
# submit = make_predictions('predictions/testclassid7scale512wclass3.h5',submit, 7)
# submit = make_predictions('predictions/testclassid8scale512wclass3.h5',submit, 8)
# submit = make_predictions('predictions/testclassid9scale512wclass10.h5',submit, 9)
# submit = make_predictions('predictions/testclassid9scale512wclass10.h5',submit, 10)
# save_submit('AN_ScaledUnet.csv', submit)
