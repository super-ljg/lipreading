from iden_train import get_model,preprocess
import numpy as np
import os
import get_imagedata
from keras.regularizers import *



if __name__ == '__main__':

    image_path = "../data/"

    model = get_model()



    batch_size = 1
    sequence = 16
    img_shape = (100,100,3)
    seq_batch_x = np.zeros((batch_size, sequence) + img_shape, dtype=K.floatx())
    batch_x = np.zeros((sequence,)+img_shape,dtype=K.floatx())
    img_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess)
    i=0
    for img_name in sorted(os.listdir(image_path)):
        if img_name.endswith('.jpg'):
            img = get_imagedata.load_img(os.path.join(image_path,img_name), target_size=img_shape[:2])  # TODO
            x = get_imagedata.img_to_array(img)
            x = img_gen.img_augment(x)
            batch_x[i] = x
            i+=1
    seq_batch_x[0]=batch_x
    result = model.predict(seq_batch_x,batch_size=batch_size)

    result = [np.argmax(x) + 1 for x in result]

    from collections import Counter
    c = Counter(result)
    pred_class, right_num = c.most_common(1)[0]
    print 'Test sample is speaker{}'.format(pred_class)



