# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Creates a 1D CNN model to classify input TCE lightcurves as plancet candidates or false positives
The network has two convolutional branches that combine in a fully connected block for sigmoid classification
One convlutional branch processes a local view of the curve and the other a global view
An output near 1 implies a planet candidate
"""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf
import pandas as pd
import datetime, os

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from keras.layers import Input

#from google.colab import files
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tfrecord_dir",
    type=str,
    default='C:/Users/A_J_F/Documents/TESS_ExoFinder/data/tfrecords',
    help="Directory containing the TFRecords for the model.")

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Number of records to take for each batch")

parser.add_argument(
    "--number_of_epochs",
    type=int,
    default=100,
    help="Number of epochs to run the network for")

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.00001,
    help="Learning rate for the CNN")

parser.add_argument(
    "--prediction_threshold",
    type=float,
    default=0.8,
    help="Threshold above which a prediction will be classified as a planet for metrics")



total_data = 3012
training_data_size = 2400
validation_data_size = 306
test_data_size = 306

GLOBAL_SIZE = [201]
LOCAL_SIZE = [61]

def decode_global(global_curve):
    global_curve = tf.cast(global_curve, tf.float32)
    global_curve = tf.reshape(global_curve, GLOBAL_SIZE)
    return global_curve

def decode_local(local_curve):
    local_curve = tf.cast(local_curve, tf.float32)
    local_curve = tf.reshape(local_curve, LOCAL_SIZE)
    return local_curve

def read_full_image_and_label(parsed):
    local_curve = decode_local(parsed['local_view'])
    global_curve = decode_global(parsed['global_view'])

    label = parsed['Disposition']
    ticid = parsed['tic_id']

    return {'local_view_dict': local_curve, 'global_view_dict': global_curve, 'tic_dict': ticid}, label

def _parse_fn_to_dict(example_serialized, is_training):
    feature_map = {
      'Disposition': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'global_view': tf.io.FixedLenSequenceFeature([], tf.float32 ,allow_missing=True, default_value=0.0),    
      'local_view': tf.io.FixedLenSequenceFeature([], tf.float32 ,allow_missing=True, default_value=0.0),
      'tic_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed = tf.io.parse_single_example(example_serialized, feature_map)

    #local_curve = decode_local(parsed['local_view'])
    #global_curve = decode_global(parsed['global_view'])
    #label = parsed['Disposition']

    return read_full_image_and_label(parsed)

def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=1000)
    parser = partial(
        _parse_fn_to_dict, is_training=True if subset == 'train' else False)
    dataset = dataset.map(
        map_func=parser,
        num_parallel_calls=4
    )
    dataset = dataset.batch(
        batch_size=batch_size
    )

    dataset = dataset.prefetch(batch_size)
    return dataset

# fit and evaluate a model
def create_model():

 	# Local view branch
  inputs1 = Input(shape=(61,1), name='local_view_dict')
  conv1 = layers.Conv1D(filters=16, kernel_size=5, activation='relu')(inputs1)
  drop1 = layers.Dropout(0.2)(conv1)
  pool1 = layers.MaxPooling1D(pool_size=5, strides=2)(drop1)
  conv1a = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(pool1)
  drop1a = layers.Dropout(0.2)(conv1a)
  pool1a = layers.MaxPooling1D(pool_size=5, strides=2)(drop1a)
  conv1b = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(pool1a)
  drop1b = layers.Dropout(0.2)(conv1b)
  pool1b = layers.MaxPooling1D(pool_size=3, strides=2)(drop1b)
  flat1 = layers.Flatten()(pool1b)

	# Global view branch
  inputs2 = Input(shape=(201,1), name='global_view_dict')
  conv2 = layers.Conv1D(filters=16, kernel_size=5, activation='relu')(inputs2)
  drop2 = layers.Dropout(0.2)(conv2)
  pool2 = layers.MaxPooling1D(pool_size=5, strides=2)(drop2)
  conv2a = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(pool2)
  drop2a = layers.Dropout(0.2)(conv2a)
  pool2a = layers.MaxPooling1D(pool_size=5, strides=2)(drop2a)
  conv2b = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(pool2a)
  drop2b = layers.Dropout(0.2)(conv2b)
  pool2b = layers.MaxPooling1D(pool_size=5, strides=2)(drop2b)
  conv2c = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(pool2b)
  drop2c = layers.Dropout(0.2)(conv2c)
  pool2c = layers.MaxPooling1D(pool_size=5, strides=2)(drop2c)
  flat2 = layers.Flatten()(pool2c)
	
	# Merge the branches
  merged = concatenate([flat1, flat2])
 
  # Fully connected branch
  dense1 = layers.Dense(128, activation='relu')(merged)
  dropdense1 = layers.Dropout(0.2)(dense1)
  dense2 = layers.Dense(64, activation='relu')(dropdense1)
  dropdense2 = layers.Dropout(0.2)(dense2)
  dense3 = layers.Dense(32, activation='relu')(dropdense2)
  dropdense3 = layers.Dropout(0.2)(dense3)
  outputs = layers.Dense(1, activation='sigmoid')(dropdense3)
 
  model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
  
  print(model.summary())
  #plot_model(model, show_shapes=True, to_file='multiinput.png')


  return model

def evaluate_model(model, test_set):

    n = 1
    threshold = FLAGS.prediction_threshold
    partial_test_set = test_set.take(n)

    final_result = pd.DataFrame()
    for curve, label in partial_test_set:
        predictions = model.predict(partial_test_set, verbose=1, steps=n)
        binary_prediction = np.where(predictions > threshold, 1, 0)
        label_list = label.numpy()
        tic = curve["tic_dict"]

        tic_frame = pd.DataFrame(tic)
        predictions_frame = pd.DataFrame(predictions)
        binary_prediction_frame = pd.DataFrame(binary_prediction)
        label_frame = pd.DataFrame(label)
        correct_label = np.where(label_frame == binary_prediction_frame, 1, 0)
        correct_label_frame = pd.DataFrame(correct_label)

        false_positive = np.where((label_frame == 0) & (binary_prediction_frame == 1), 1, 0)
        false_positive_frame = pd.DataFrame(false_positive)

        false_negative = np.where((label_frame == 1) & (binary_prediction_frame == 0), 1 ,0)
        false_negative_frame = pd.DataFrame(false_negative)

        true_positive = np.where((label_frame == 1) & (binary_prediction_frame == 1), 1, 0)
        true_positive_frame = pd.DataFrame(true_positive)

        true_negative = np.where((label_frame == 0) & (binary_prediction_frame == 0), 1, 0)
        true_negative_frame = pd.DataFrame(true_negative)

        result = pd.concat([tic_frame, label_frame, predictions_frame, binary_prediction_frame, correct_label_frame, false_positive_frame, false_negative_frame, true_positive_frame, true_negative_frame], axis=1, join='inner')
        final_result = final_result.append(result)
    

    final_result.columns = ["TIC", "Actual", "Prediction", "Binary Prediction", "Correct", "False Positive", "False Negative", "True Positive", "True Negative"]

    total_curves = final_result.shape[0]
    correct_total = final_result['Correct'].sum()
    false_positive_total = final_result['False Positive'].sum()
    false_negative_total = final_result['False Negative'].sum()
    true_positive_total = final_result['True Positive'].sum()
    true_negative_total = final_result['True Negative'].sum()

    sums = final_result.select_dtypes(np.number).sum().rename('Total')
    final_result = final_result.append(sums)

    print(final_result)

    print()
    print("Totals")
    print("The total of curves: " + str(total_curves))
    print("The total of corrects: " + str(correct_total))
    print("The total of false positives: " + str(false_positive_total))
    print("The total of false negatives: " + str(false_negative_total))
    print("The total of true positives: " + str(true_positive_total))
    print("The total of true negatives: " + str(true_negative_total))

    accuracy = float(true_positive_total + true_negative_total) / float(total_curves)
    precision = float(true_positive_total) / float(true_positive_total + false_positive_total)
    recall = float(true_positive_total) / float(true_positive_total + false_negative_total)
    F1_score = (2 * ( precision * recall )) / ( precision + recall )

    print()
    print("Metrics")
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(F1_score))

    return accuracy, precision, recall, F1_score

def list_average(list):
    list_sum = sum(list)
    list_length = len(list)
    average = list_sum / list_length

    return average

def main(argv):
    del argv  # Unused.
    BATCH_SIZE = FLAGS.batch_size
    TFRECORD_DIR = FLAGS.tfrecord_dir
    LEARNING_RATE = FLAGS.learning_rate

    training_set = get_dataset(TFRECORD_DIR, 'train', BATCH_SIZE)
    validation_set = get_dataset(TFRECORD_DIR, 'val', BATCH_SIZE)
    test_set = get_dataset(TFRECORD_DIR, 'test', test_data_size)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(10):

        model = create_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=0.00000001),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])


        tf.compat.v1.logging.info("Training network for %d epochs", FLAGS.number_of_epochs)

        history = model.fit(x=training_set, 
                        epochs=FLAGS.number_of_epochs,
                        verbose=1,
                        validation_data=validation_set,
                        steps_per_epoch=training_data_size // BATCH_SIZE,
                        validation_steps=validation_data_size // BATCH_SIZE)
                        #callbacks=[tensorboard_callback])

        accuracy, precision, recall, f1 = evaluate_model(model,test_set)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    accuracy_average = list_average(accuracy_list)
    precision_average = list_average(precision_list)
    recall_average = list_average(recall_list)
    f1_average = list_average(f1_list)

    print()
    print("TESS ExoFinder Metrics")
    print()

    print("Accuracy values")
    print(accuracy_list)
    print("Accuracy average: " + str(accuracy_average))

    print("Precision values")
    print(precision_list)
    print("Precision average: " + str(precision_average))

    print("Recall values")
    print(recall_list)
    print("Recall average: " + str(recall_average))

    print("F1-score values")
    print(f1_list)
    print("F1 average: " + str(f1_average))

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(argv=[sys.argv[0]] + unparsed)