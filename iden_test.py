import os
import numpy as np
import pickle as pk
from keras.regularizers import *
import get_imagedata
from iden_train import get_model,preprocess


def iden_test(h_shift = 0,w_shift = 0, rota = 0, scal = 0, random_transform = False,classes = 200):
    T = 16  
    img_col = 100
    img_row = 100
    img_chan = 3
    img_shape = (img_row,img_col,img_chan)
    model = get_model()
    model.load_weights('./weights/t8-12.165-0.9938.h5')
   
    
    directory = '../data/5sec10'
    f2 = file('train_samples.pkl','rb')
    train_samples = pk.load(f2)
    f2.close()
    test_samples = [x for x in range(1,11) if x not in train_samples[:3]]
    print test_samples
    sequence = T
    frame_total_right = 0
    sequence_total_right = 0
    frame_total = 0
    sequence_total = 0
    temp_filenames = []


    def _recursive_list(subpath):
            return sorted(os.walk(subpath), key=lambda tpl: tpl[0])


    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    for class_num in range(1,classes+1):
        temp_path = os.path.join(directory,str(class_num))
        for sub_path in test_samples:
            abs_path = os.path.join(temp_path,str(sub_path))
            for root, _, files in _recursive_list(abs_path):
                for fname in sorted(files):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        temp_filenames.append(os.path.join(abs_path,fname))
                temp_filenames = sorted(temp_filenames,key=lambda x:int(x[-8:-4]))

            batch_size = len(temp_filenames) - sequence + 1 # TODO
            frame_result = np.zeros((batch_size,classes))
            seq_batch_x = np.zeros((batch_size, sequence) + img_shape, dtype=K.floatx())
            batch_x = np.zeros((sequence,)+img_shape,dtype=K.floatx())
            img_gen = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess)
            if random_transform:
                h_shift = np.random.uniform(-h_shift, h_shift) # video randomly transform instead of clips
                w_shift = np.random.uniform(-w_shift, w_shift)
                rotation = np.random.uniform(-rota, rota)
                scale = np.random.uniform(scal, 1)
            else:
                h_shift, w_shift, rotation, scale = h_shift, w_shift, rota, scal
            for j in range(batch_size):

                for i in range(sequence):
                    img = get_imagedata.load_img(temp_filenames[i+j],target_size = img_shape[:2]) # TODO
                    x = get_imagedata.img_to_array(img)
                    x = img_gen.img_augment(x,rotation,h_shift,w_shift,scale)
                    batch_x[i] = x
                seq_batch_x[j] = batch_x
            temp_filenames = []
            frame_result[:42] = model.predict(seq_batch_x[:42],batch_size=42)
            frame_result[42:] = model.predict(seq_batch_x[42:],batch_size = batch_size-42)
            # frame_result = model.predict(seq_batch_x,batch_size=batch_size)  # TODO
            frame_result = [np.argmax(x) + 1 for x in frame_result]

            from collections import Counter
            c = Counter(frame_result)
            pred_class, right_num = c.most_common(1)[0]
            frame_total += len(frame_result)
            sequence_total += 1

            if pred_class == class_num:
                sequence_total_right += 1
                frame_total_right += right_num
    frame_acc = frame_total_right / float(frame_total)
    sequence_acc = sequence_total_right / float(sequence_total)
    f1 = file('result.txt','a')
    print('frame_total_right:', frame_total_right, '  frame_total: ', frame_total, 'accuracy:', frame_acc)
    print('sequence_total_right:', sequence_total_right, '  sequence_total: ', sequence_total, 'accuracy:', sequence_acc)
    f1.write('frame_total_right: {}  frame_total: {}  frame_accuracy: {}'.format(frame_total_right,frame_total,frame_acc))
    f1.write('sequence_total_right: {}  sequence_total: {}  sequence_accuracy: {}'.format(sequence_total_right, sequence_total, sequence_acc))
    f1.close()
if __name__ == '__main__':
    iden_test(w_shift=0,h_shift=0,scal=1,random_transform=False,rota=0)
