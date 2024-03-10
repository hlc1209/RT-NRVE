from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Convolution2D, Bidirectional,TimeDistributed
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, ReLU, Reshape, Activation, concatenate, LSTM, GRU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import he_normal,glorot_uniform
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os
from tensorflow.keras import backend as K



def myModel():
    model_input = Input(shape=(47, 257, 2))
    # print('0:', model_input.shape)

    conv1 = Convolution2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv1')(model_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # print('1:', conv1.shape)

    conv2 = Convolution2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    # print('2:', conv2.shape)

    conv3 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    # print('3:', conv3.shape)

    conv4 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    # print('4:', conv4.shape)

    conv5 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    # print('5:', conv5.shape)

    conv6 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    # print('6:', conv6.shape)

    conv9 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv9')(conv6)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    # print('9:', conv9.shape)

    conv10 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='conv10')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    # print('10:', conv10.shape)

    conv11 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='conv11')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    # print('11:', conv11.shape)

    conv12 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='conv12')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    # print('11:', conv11.shape)

    conv13 = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='conv13')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    # print('11:', conv11.shape)

    conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv15')(conv13)
    conv15 = BatchNormalization()(conv15)
    conv = Activation('relu')(conv15)
    print('15:', conv15.shape)

    AVfusion = TimeDistributed(Flatten())(conv)
    # print('AVfusion:', AVfusion.shape)

    lstm = Bidirectional(GRU(192,input_shape=(47,8*257),return_sequences=True),merge_mode='sum')(AVfusion)
    # print('lstm:', lstm.shape)

    fc = Dense(128, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    # print('fc:', fc.shape)
    fc = Dense(128, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc)
    # print('fc:', fc.shape)

    complex_mask = Dense(257 * 2 * 1, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc)
    # print('complex_mask:', complex_mask.shape)

    complex_mask_out = Reshape((47, 257, 2, 1))(complex_mask)
    # print('complex_mask_out:', complex_mask_out.shape)

    # --------------------------- AO end ---------------------------
    model = Model(inputs=model_input, outputs=complex_mask_out)

    return model


if __name__ == '__main__':
    #############################################################
    RESTORE = True
    # If set true, continue training from last checkpoint
    # needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

    # super parameters
    people_num = 2
    epochs = 50
    initial_epoch = 0
    batch_size = 2
    #############################################################

    # audio_input = np.random.rand(5, 298, 257, 2)        # 5 audio parts, (298, 257, 2) stft feature
    # audio_label = np.random.rand(5, 298, 257, 2, people_num)     # 5 audio parts, (298, 257, 2) stft feature, people num to be defined

    # ///////////////////////////////////////////////////////// #
    # create folder to save models
    path = './saved_models_AO'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('create folder to save models')
    filepath = path + "/AOmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.10f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint2 = ModelCheckpoint(path + "/AOmodel-latest-" + str(people_num) + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # ///////////////////////////////////////////////////////// #

    #############################################################
    # automatically change lr
    def scheduler(epoch):
        ini_lr = 0.001
        lr = ini_lr
        if epoch >= 5:
            lr = ini_lr / 5
        if epoch >= 10:
            lr = ini_lr / 10
        return lr

    rlr = LearningRateScheduler(scheduler, verbose=1)
    #############################################################

    # ///////////////////////////////////////////////////////// #
    # read train and val file name
    # format: mix.npy single.npy single.npy
    trainfile = []
    valfile = []
    with open('./trainfile.txt', 'r') as t:
        trainfile = t.readlines()
    with open('./valfile.txt', 'r') as v:
        valfile = v.readlines()
    # ///////////////////////////////////////////////////////// #

    # the training steps
    def latest_file(dir):
        lists = os.listdir(dir)
        lists.sort(key=lambda fn: os.path.getmtime(dir + fn))
        file_latest = os.path.join(dir, lists[-1])
        return file_latest

    if RESTORE:
        last_file = latest_file('./saved_models_AO/')
        AO_model = load_model(last_file)
        info = last_file.strip().split('-')
        initial_epoch = int(info[-2])
        # print(initial_epoch)
    else:
        AO_model = AO_model(people_num)
        adam = optimizers.Adam()
        AO_model.compile(optimizer=adam, loss='mse')
    # AO_model.fit(audio_input, audio_label,
    #              epochs=epochs,
    #              batch_size=2,
    #              validation_data=(audio_input, audio_label),
    #              shuffle=True,
    #              callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
    #              initial_epoch=initial_epoch)

    AO_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )
