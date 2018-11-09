import cv2
import os
import re
import numpy as np
import math

import tensorflow as tf
from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

# Rootdir of the dataset
#rootdir = "/media/alejandro/DATA/datasets/simdronet"
ROOTDIR = "/media/alejandro/DATA/datasets/simple_uav_dataset/simulation_divided/val"
OUTPATH = "/media/alejandro/DATA/datasets/simple_uav_dataset/simulation_divided/uav_val.record"

SHOW_IMAGES = 0
WRITE_ANNOTATIONS = 0
GENERATE_FOR_COLAB = 0
ENABLE_SHARDING = 1

if ENABLE_SHARDING:
    num_shards = 10

index_shar = 0

if not ENABLE_SHARDING:
    # Create writer
    writer = tf.python_io.TFRecordWriter(OUTPATH)

with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, OUTPATH, num_shards)

    # Go across all files and subfolders
    for subdir, dirs, files in os.walk(ROOTDIR):
        for file in files:
            if file.endswith(".jpg") and not file.startswith("frame_seg"):
                # Print some info
                print("Raw filename: " + file)
                result = re.findall("\d+", file)
                result = result[0]
                file_seg = "frame_seg_" + result + ".jpg"
                print(file_seg)

                index_shar += 1


                # Load an color image in grayscale
                img = cv2.imread(os.path.join(subdir, file))
                img_height, img_width = img.shape[:2]
                # print(str(img_width) +" "+str(img_height))
                # input()
                # Load an color image in grayscale
                img_seg = cv2.imread(os.path.join(subdir, file_seg))
                mask = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                # mask = mask / 255
                # img = cv2.bitwise_and(img, img, mask=mask)

                x, y, w, h = cv2.boundingRect(mask)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if WRITE_ANNOTATIONS:
                    if not GENERATE_FOR_COLAB:
                        # Write info in a file
                        with open(subdir + '/../../annotations.txt', 'a') as the_file:
                            the_file.write(str(subdir + "/" + file) + "," + str(x) + "," + str(y) + "," + str(x + w) + ","
                                           + str(y + h) + "," + "drone" + '\n')
                    else:
                        # Write info in a file
                        index = subdir.find("simulation/")
                        temp = subdir[index + 11 :len(subdir)]
                        with open(subdir + '/../../annotations.txt', 'a') as the_file:
                            the_file.write(str(temp + "/" + file) + "," + str(x) + "," + str(y) + "," + str(x + w) + ","
                                           + str(y + h) + "," + "drone" + '\n')

                # print(str(x))
                # print(str(y))
                # print(str(w))
                # print(str(h))

                height = img_height # Image height
                width = img_width  # Image width
                filename = tf.compat.as_bytes(str(index_shar).zfill(6) + "_" + file)  # Filename of the image. Empty if image is not from file
                encoded_image_data = tf.compat.as_bytes(cv2.imencode('.jpg', img)[1].tostring())  # Encoded image bytes
                image_format = b'jpg'  # b'jpeg' or b'png'

                xmins = [float(x) / float(img_width)]  # List of normalized left x coordinates in bounding box (1 per box)
                xmaxs = [float(x + w) / float(img_width)]  # List of normalized right x coordinates in bounding box
                # (1 per box)
                ymins = [float(y) / float(img_height)]  # List of normalized top y coordinates in bounding box (1 per box)
                ymaxs = [float(y + h) / float(img_height)]  # List of normalized bottom y coordinates in bounding box
                # (1 per box)
                classes_text = [tf.compat.as_bytes('Drone')]  # List of string class name of bounding box (1 per box)
                classes = [1]  # List of integer class id of bounding box (1 per box)

                if (xmins[0] > 1.1 or ymins[0] > 1.1 or xmaxs[0] > 1.1 or ymaxs[0] > 1.1):
                    print("Higher " + str(xmins[0]) + " " +  str(ymins[0]) + " " +  str(xmaxs[0]) + " " + str(ymaxs[0]))
                    input()

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(filename),
                    'image/source_id': dataset_util.bytes_feature(filename),
                    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                    'image/format': dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))

                if not ENABLE_SHARDING:
                    # Write tf example
                    writer.write(tf_example.SerializeToString())
                else:
                    output_shard_index = index_shar % num_shards
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

                if SHOW_IMAGES:
                    # Show iamge
                    cv2.imshow('image', img)
                    cv2.imshow('image_seg', img_seg)
                    cv2.waitKey(0)

                #input("Press Enter to continue...")

if not ENABLE_SHARDING:
    # Close writer
    writer.close()
