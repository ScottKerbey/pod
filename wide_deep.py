# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

# _CSV_COLUMNS = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education_num',
#     'marital_status', 'occupation', 'relationship', 'race', 'gender',
#     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
#     'income_bracket'
# ]
_CSV_COLUMNS = [
    'id', 'SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
#                         [0], [0], [0], [''], ['']]
_CSV_COLUMN_DEFAULTS = [[''], [0], [0.0], [0], [0], [0.0], [0.0], [0], [0], [0], [0], [0]]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/tmp/pod_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str,
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/Users/skerb/.kaggle/competitions/GiveMeSomeCredit/cs-training.csv',
    help='Path to the training data.')

parser.add_argument(
    '--validation_data', type=str, default='/Users/skerb/.kaggle/competitions/GiveMeSomeCredit/cs-validation.csv',
    help='Path to the validation data.')

parser.add_argument(
    '--test_data', type=str, default='/Users/skerb/.kaggle/competitions/GiveMeSomeCredit/cs-test.csv',
    help='Path to the test data.')

parser.add_argument(
    '--train', default=False, action='store_const', const=True,
)

parser.add_argument(
    '--predict', default=False, action='store_const', const=True,
)

_NUM_EXAMPLES = {
    'train': 120000,
    'validation': 30000,
}

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  age = tf.feature_column.numeric_column('age', default_value=40)
  unsecured_lines = tf.feature_column.numeric_column('RevolvingUtilizationOfUnsecuredLines', default_value=0)
  times_past_due_3059 = tf.feature_column.numeric_column('NumberOfTime30-59DaysPastDueNotWorse', default_value=0)
  times_past_due_6089 = tf.feature_column.numeric_column('NumberOfTime60-89DaysPastDueNotWorse', default_value=0)
  times_90_days_late = tf.feature_column.numeric_column('NumberOfTimes90DaysLate', default_value=0)
  debt_ratio = tf.feature_column.numeric_column('DebtRatio', default_value=0.5)
  monthly_income = tf.feature_column.numeric_column('MonthlyIncome', default_value=8000)
  open_credit_lines = tf.feature_column.numeric_column('NumberOfOpenCreditLinesAndLoans', default_value=0)
  num_real_estate_loans = tf.feature_column.numeric_column('NumberRealEstateLoansOrLines', default_value=0)
  num_dependents = tf.feature_column.numeric_column('NumberOfDependents', default_value=1)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [
      unsecured_lines, times_past_due_3059, times_past_due_6089, debt_ratio, monthly_income,
      open_credit_lines, times_90_days_late, num_real_estate_loans, num_dependents, age_buckets,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['DebtRatio', 'MonthlyIncome'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          ['NumberOfDependents', 'MonthlyIncome'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'NumberOfOpenCreditLinesAndLoans'], hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      age,
      unsecured_lines,
      times_past_due_3059,
      times_past_due_6089,
      debt_ratio,
      monthly_income,
      open_credit_lines,
      times_90_days_late,
      num_real_estate_loans,
      num_dependents,
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    return load_estimator(model_dir, model_type)#, None)

def load_estimator(model_dir, model_type):#, model_file):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if FLAGS.predict:
      if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config,
            warm_start_from=model_dir)
      elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config,
            warm_start_from=model_dir)
      else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config,
            warm_start_from=model_dir)
  else:
      if model_type == 'wide':
          return tf.estimator.LinearClassifier(
              model_dir=model_dir,
              feature_columns=wide_columns,
              config=run_config)
      elif model_type == 'deep':
          return tf.estimator.DNNClassifier(
              model_dir=model_dir,
              feature_columns=deep_columns,
              hidden_units=hidden_units,
              config=run_config)
      else:
          return tf.estimator.DNNLinearCombinedClassifier(
              model_dir=model_dir,
              linear_feature_columns=wide_columns,
              dnn_feature_columns=deep_columns,
              dnn_hidden_units=hidden_units,
              config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --validation_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, na_value='NA')
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('SeriousDlqin2yrs')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)



  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def main(unused_argv):

  if not FLAGS.train and not FLAGS.predict:
      print('No training or prediction called')
      return

  if FLAGS.train:

      if not FLAGS.model_type:
          FLAGS.model_type = 'wide_deep'

      prev = open('model.txt','w')
      prev.write(FLAGS.model_type)
      prev.close()

      # Clean up the model directory if present
      shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
      model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

      # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
      for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.validation_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
          print('%s: %s' % (key, results[key]))

  else:
      if not FLAGS.model_type:
          prev = open('model.txt','r')
          FLAGS.model_type = prev.read()
          prev.close()

      model = load_estimator(FLAGS.model_dir, FLAGS.model_type)#, FLAGS.model_file)

  if FLAGS.predict:
      predictions = model.predict(input_fn=lambda: input_fn(
          FLAGS.test_data, 1, False, FLAGS.batch_size))

      print('-' * 60)
      print('Predictions')
      print('-' * 60)

      writer = open('output.csv', 'w')
      writer.write('Id,Probability\n')
      i = 1
      for value in predictions:
        print('%s,%s' % (i, str(value['logistic'])[1:-1]))
        writer.write('%s,%s\n' % (i, str(value['logistic'])[1:-1]))
        i = i + 1

      writer.close()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
