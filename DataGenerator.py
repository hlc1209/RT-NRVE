import numpy as np
import tensorflow.keras as keras


class dataGenerator(keras.utils.Sequence):
    def __init__(self, filename, database_dir_path, Xdim=(47, 257, 2), ydim=(47, 257, 2), batch_size=4, shuffle=True):
        'Initialization'
        self.filename = filename
        self.Xdim = Xdim # TODO
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.database_dir_path = database_dir_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filename) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list
        filename_temp = [self.filename[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(filename_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filename))
        if self.shuffle:
            # print("shuffled")
            np.random.shuffle(self.indexes)

    def __data_generation(self, filename_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.Xdim))
        y = np.empty((self.batch_size, *self.ydim))

        # Generate data
        for i, ID in enumerate(filename_temp):
            info = ID.strip().split('-')
            X[i,] = np.load(self.database_dir_path+'noisy/' + info[0] + '/' + ID)
            
            y[i, :, :, :] = np.load(self.database_dir_path+'crm/' + info[0] + '/' + ID)

        # assert y[:,:,:,0] != y[:,:,:,1]
        return X, y
