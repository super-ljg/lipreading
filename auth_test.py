import os
import numpy as np
from keras.regularizers import *
import get_imagedata
from iden_train import get_model,preprocess
import pickle as pk


frame_total_right = 0.0
sequence_total_right = 0.0
frame_total = 0.0
sequence_total = 0.0
pos_frame = pos_sequence = pos_frame_right = pos_sequence_right = 0.0

def auth_test(h=0,w=0,r=0,s=1,weight_path=None):
    T = 16
    img_col = 100
    img_row = 100
    img_chan = 3
    global frame_total_right,pos_frame
    global sequence_total_right,pos_sequence_right
    global frame_total,pos_frame_right
    global sequence_total,pos_sequence
    img_shape = (img_row,img_col,img_chan)
    model = get_model()
    if weight_path:
        model.load_weights(weight_path)
    else:
        raise ValueError("missing weight_path parameter")
    directory = '../data/5sec10'
    sequence = T
    client = int(weight_path.split('.')[0])
    temp_filenames = []
    
    def _recursive_list(subpath):
            return sorted(os.walk(subpath), key=lambda tpl: tpl[0])

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    f2 = file('auth_train.pkl','rb')
    data_dict = pk.load(f2)
    test_list = data_dict['class_list']
    train_samples = data_dict['train_samples']
    test_list[test_list.index(client)], test_list[0] = test_list[0], client
    test_list = test_list[50:]
    test_list.append(client)

    f2.close()
    for class_num in test_list:
        sub_dir = range(1, 11) if not client == class_num else train_samples[6:]
        temp_path = os.path.join(directory,str(class_num))
        for sub_path in sub_dir:
            h_shift = np.random.uniform(-h, h)
            w_shift = np.random.uniform(-w, w)
            rotation = np.random.uniform(-r, r)
            scale = np.random.uniform(s, 1)
            abs_path = os.path.join(temp_path,str(sub_path))
            for root, _, files in _recursive_list(abs_path):
                for fname in sorted(files):#  may have .DS_store file
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        temp_filenames.append(os.path.join(abs_path,fname))
                temp_filenames = sorted(temp_filenames,key=lambda x:int(x[-8:-4]))

            batch_size = len(temp_filenames) - sequence + 1
            frame_result = np.zeros((batch_size,2))
            seq_batch_x = np.zeros((batch_size, sequence) + img_shape, dtype=K.floatx())
            batch_x = np.zeros((sequence,)+img_shape,dtype=K.floatx())
            for j in range(batch_size):
                for i in range(sequence):
                    img = get_imagedata.load_img(temp_filenames[i+j],target_size = img_shape[:2])
                    x = get_imagedata.img_to_array(img)
                    x = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess).\
                                                    img_augment(x=x,h_shift=h_shift,
                                                w_shift=w_shift,rotation=rotation,scale=scale)
                    batch_x[i] = x
                seq_batch_x[j] = batch_x
            temp_filenames = []

            frame_result[:42] = model.predict(seq_batch_x[:42],batch_size=42)
            frame_result[42:] = model.predict(seq_batch_x[42:],batch_size = batch_size-42)
            frame_result = [np.argmax(x)  for x in frame_result]

            from collections import Counter
            c = Counter(frame_result)
            pred_class, right_num = c.most_common(1)[0]

            frame_total += len(frame_result)
            sequence_total += 1

            if class_num == client:
                pos_frame += len(frame_result)
                pos_sequence += 1
                if pred_class == 1:
                    pos_frame_right += right_num
                    pos_sequence_right += 1
                else:
                    print ('false rejecting client: ',pred_class, class_num)
                    print (frame_result)
            else:
                if pred_class==0 :
                    sequence_total_right += 1
                    frame_total_right += right_num
                else:
                    print ('false accepting imposter: ',pred_class,class_num)
                    print (frame_result)



if __name__ == '__main__':

    for i in range(1,201):
        weight_path = '{}.h5'.format(i)
        print (weight_path)
        auth_test(weight_path=weight_path)

    frame_acc = frame_total_right / frame_total
    sequence_acc = sequence_total_right / sequence_total

    pos_frame_acc = pos_frame_right / pos_frame
    pos_seq_acc = pos_sequence_right / pos_sequence
    FAR = 1 - sequence_acc
    FRR = 1 - pos_seq_acc
    print('frame_total_right:', frame_total_right, '  frame_total: ', frame_total)
    print ('frame accuracy:', frame_acc)
    print('sequence_total_right:', sequence_total_right, '  sequence_total: ', sequence_total)
    print ('sequence accuracy:', sequence_acc)
    print('pos_frame_total_right:', pos_frame_right, '  pos_frame_total: ', pos_frame,
          'accuracy:{:.4f}'.format(pos_frame_acc))
    print('pos_sequence_total_right:', pos_sequence_right, ' pos_sequence_total: ', pos_sequence,
          'accuracy:{:.4f}'.format(pos_seq_acc))
    print ('FAR:   {:.4f}'.format(FAR), 'FRR: {:.4f}'.format(FRR), 'HTER:  {:.4f}'.format((FAR + FRR) / 2))

