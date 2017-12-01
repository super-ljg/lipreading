from keras.layers import *
from keras.optimizers import sgd, rmsprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import get_imagedata
import pickle as pk
from iden_train import get_model,preprocess

def auth_train(client,random_class,train_samples):
    """
    train lip authentication model for each speaker
    """
    T = 16
    model = get_model()
    model.load_weights('./weights/new_14.129-0.9992.h5')
    print (len(model.layers))
    model.pop()
    frozen_layers = len(model.layers)-2
    model.add(Dense(2, activation='sigmoid'))
    for layer in model.layers[:frozen_layers]:
        layer.trainable = False
    model.summary()

    random_class[random_class.index(client)], random_class[0] = random_class[0], client

    rand_train_class = random_class[:50]
    SGD = sgd(lr=0.01, momentum=0.9, decay=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=SGD, metrics=['accuracy'])
    early_stop = EarlyStopping(patience=7, verbose=1, monitor='val_loss')
    save_best = ModelCheckpoint('./weights/'+ str(client) + '.h5', monitor='val_loss', verbose=1,save_best_only=True)
    train_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess, client=client,
                                                 height_shift_range=0.1,width_shift_range=0.1,
                                                 rotation_range=10,zoom_range=(0.9,1))
    val_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess, client=client)

    model.fit_generator(
        train_gen.flow_from_directory('../data/5sec10',sub_dir=train_samples[:3], sequence=T,
                                      auth_rand_train=rand_train_class,
                                      target_size=(100, 100),batch_size=32,authentication=True)
                                 , steps_per_epoch=200, epochs=15, callbacks=[early_stop, save_best],
        validation_data=val_gen.flow_from_directory('../data/5sec10', sub_dir=train_samples[3:],
                                    sequence=T,auth_rand_train=rand_train_class,
                                    target_size=(100, 100), batch_size=32,authentication=True)
                                    , validation_steps=200, workers=4,use_multiprocessing=True)


if __name__ == '__main__':
    class_list = np.arange(1, 201)
    np.random.shuffle(class_list)
    class_list = list(class_list)
    train_samples = np.arange(1, 11)
    np.random.shuffle(train_samples)
    data_dict = {'class_list': class_list, 'train_samples': train_samples}
    f1 = file('auth_train.pkl', 'wb')
    pk.dump(data_dict, f1)
    f1.close()
    for i in range(1,201):
        auth_train(client=i,random_class=class_list,train_samples=train_samples)
