from keras.models import Sequential
from keras.layers import *
from keras.optimizers import sgd
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import get_imagedata
import pickle as pk



def preprocess(x):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x

def get_model():
    """ Return the Keras model of the network
        """
    T = 16
    img_col = 100
    img_row = 100
    img_chan = 3
    nb_class = 200
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, padding='same', name='conv1', subsample=(1, 1, 1),
                     input_shape=(T, img_row, img_col, img_chan)))  # TODO
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(64, 3, 3, 3, padding='same', name='conv1_1', subsample=(1, 2, 2), activation='relu'))

    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, padding='same', name='conv2', subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(128, 3, 3, 3, padding='same', name='conv2_1', subsample=(1, 2, 2), activation='relu'))

    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv3b', subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv3_1', subsample=(1, 2, 2), activation='relu'))

    # 4th layer group
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv4b', subsample=(1, 2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv4_1', subsample=(1, 2, 2), activation='relu'))

    # 5th layer group
    model.add(Flatten())
    # FC layers group
    model.add(Dense(1024, name='fc6'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(1024, name='fc7'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(nb_class, activation='softmax', name='fc8'))
    model.summary()
    return model

def iden_train():

    T = 16

    model = get_model()
    SGD = sgd(lr=0.1, momentum=0.9, decay=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

    learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 60])
    early_stop = EarlyStopping(patience=40, verbose=1, monitor='val_acc')
    save_best = ModelCheckpoint('./weights/final_14.{epoch:02d}-{val_acc:.4f}.h5', monitor='val_loss', verbose=1,
                                save_best_only=True)
    train_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess,
                                                 rotation_range=10,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 zoom_range=(0.8,1))
    val_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess)

    random_sample = np.arange(1,11)
    np.random.shuffle(random_sample)
    f1 = file('train_samples.pkl','wb')
    pk.dump(random_sample,f1)
    f1.close()
    train_generator = train_gen.flow_from_directory('../data/5sec10', sub_dir=random_sample[:2], sequence=T,
                                                    target_size=(100, 100),batch_size=24,
                                                    authentication=False,sampling=False)
    val_generator = val_gen.flow_from_directory('../data/5sec10', sub_dir=random_sample[2:3], sequence=T,
                                                target_size=(100, 100),batch_size=24,
                                                authentication=False,sampling=False)

    model.fit_generator(train_generator, steps_per_epoch=300, epochs=400,
                        callbacks=[early_stop, save_best,lr_scheduler],
                        validation_data=val_generator,validation_steps=200,
                        workers=4, use_multiprocessing=True)



if __name__ == '__main__':

    iden_train()

