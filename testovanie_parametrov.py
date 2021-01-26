import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from tensorflow.keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.python.client import device_lib

import wandb
from wandb.keras import WandbCallback

from tensorflow.compat.v1.keras import backend as K

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)
K.set_session(sess)  

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

df = pickle.load( open( "/home/kriza/programing/clanok/data/picklnute/30min.p", "rb" ) )

TRAIN_SPLIT_BEGIN = 24
TRAIN_SPLIT_END = 15551 - TRAIN_SPLIT_BEGIN
VALID_SPLIT_BEGIN = 15601 - TRAIN_SPLIT_BEGIN
VALID_SPLIT_END = 17463 -TRAIN_SPLIT_BEGIN
tf.random.set_seed(42)

items = [
# '30m-item56',
# '30m-item57',
# '30m-item58',
# '30m-item59',
# '30m-item60',
# '30m-item62',
# '30m-item63',
'30m-item64',
# '30m-item65',
# '30m-item67',
# '30m-item68',
# '30m-item69',
# '30m-item70',
# '30m-item71',
# '30m-item72',
# '30m-item73',
# '30m-item74',
# '30m-item75',
# '30m-item76',
# '30m-item77'
]

UNIT_COUNT = [
  16, 
32, 
64, 128, 256, 512
]
HISTORY = [16, 
32, 64, 128, 256, 512
]
BATCH_SIZES = [
  # 16, 
  32, 64, 
  128, 256, 512
  ]

for item in items:

  uni_data = df[item]

  uni_data = uni_data.values
  uni_data = uni_data[TRAIN_SPLIT_BEGIN:VALID_SPLIT_END + TRAIN_SPLIT_BEGIN]

  uni_train_mean = uni_data[:TRAIN_SPLIT_END].mean()
  uni_train_std = uni_data[:TRAIN_SPLIT_END].std()
  uni_data = (uni_data-uni_train_mean)/uni_train_std

  univariate_past_history = 0
  BATCH_SIZE = 0
  UNITS = 0
  saved_models = ""

  for i in BATCH_SIZES:
    for j in HISTORY:
      for k in UNIT_COUNT:
        if i == 512 and j == 512 and k == 512:
          continue
        else:
          univariate_past_history = j
          BATCH_SIZE = i
          UNITS = k

          saved_models = "saved_models/" + item + "/hist=" + str(univariate_past_history) + "batch=" + str(BATCH_SIZE) + "units=" + str(UNITS) + "/model.h5"
          # print(saved_models)
          wandb.init( project="predikcie1krok-"+item, 
                      name = "hist=" + str(univariate_past_history) + "batch=" + str(BATCH_SIZE) + "units=" + str(UNITS), 
                      reinit = True, 
                      id = item + "hist=" + str(univariate_past_history) + "batch=" + str(BATCH_SIZE) + "units=" + str(UNITS)
                    )

          univariate_future_target = 0

          x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT_END,
                                                  univariate_past_history,
                                                  univariate_future_target)
          x_val_uni, y_val_uni = univariate_data(uni_data, VALID_SPLIT_BEGIN, None,
                                              univariate_past_history,
                                              univariate_future_target)

          BUFFER_SIZE = 10000

          train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
          train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

          val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
          val_univariate = val_univariate.batch(BATCH_SIZE)

          simple_lstm_model = tf.keras.models.Sequential([
              tf.keras.layers.LSTM(UNITS, input_shape=x_train_uni.shape[-2:]),
              tf.keras.layers.Dense(1)
          ])

          simple_lstm_model.compile(optimizer='adam', loss='mae', metrics='mae')

          EPOCHS = 50

          ckpt = ModelCheckpoint(filepath=saved_models, monitor='val_loss', mode='min', save_best_only = True, save_weights_only=True, verbose = False)

          simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                              #   steps_per_epoch=int(x_train_uni.shape[0]/BATCH_SIZE),
                              validation_data=val_univariate,
                              callbacks=[ckpt, 
                              WandbCallback()
                              ]
                              #   , validation_steps=50
                              )


