# %%
import sys
import NRVE_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import os

model = NRVE_model.myModel()


model.compile(optimizer='adam', loss='mse')
print(model.summary())
model.save('test_size_model.h5')


