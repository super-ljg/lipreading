from __future__ import print_function
from vis.losses import ActivationMaximization
from vis.optimizer import *
from vis import backend, backprop_modifiers
from keras.applications import VGG16
from vis.utils import utils
import os
from vis.visualization import overlay,saliency
import PIL.Image as pil
from matplotlib import pyplot as plt
from keras.preprocessing import image
import matplotlib.cm as cm
from keras.models import Sequential
from keras.layers import *
import get_imagedata
from iden_train import get_model,preprocess

def iden_visualization():

    model = get_model()
    model.load_weights('./weights/newnetwork_14.78-0.9933.h5')

    # Swap softmax with linear, only needed when visualing softmax layer
    layer_idx = utils.find_layer_idx(model, 'fc8')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    for filt_index in range(0,200):
        visualize_saliency_3Dcnn(model, layer_idx, filter_indices=filt_index, seed_input=get_imgsequence(pre_process=True,
            path = '/home/ljg/Desktop/lipnet/data/5sec10/{}/1/'.format(filt_index+1)),
     original_img=get_imgsequence(pre_process=False,path='/home/ljg/Desktop/lipnet/data/5sec10/{}/1/'.format(filt_index+1)),
                                 backprop_modifier= None)

def get_imgsequence(path='/home/ljg/Desktop/lipnet/data/5sec10/1/1/',start_T=2,time_window = 16,pre_process = False,sampling = False):

    batch_x = np.zeros((time_window,100,100,3), dtype=K.floatx())
    for i in range(start_T, start_T + time_window):
        if sampling:
            img_name = ('IMG_00' + str(2*i)+'.jpg') if 2*i>9 else ('IMG_000' + str(2*i) + '.jpg')
        else:

            img_name = ('IMG_00' + str(i)+'.jpg') if i>9 else ('IMG_000' + str(i) + '.jpg')
        img_path = path + img_name
        img = get_imagedata.load_img(img_path, target_size=(100,100))
        x = get_imagedata.img_to_array(img)
        if pre_process:
            x = get_imagedata.ImageDataGenerator(preprocessing_function=preprocess).img_augment(x)
        batch_x[i-10] = x
    return batch_x


def fullvideo_vis():
    T = 48
    img_col = 100
    img_row = 100
    img_chan = 3
    nb_class = 200
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, padding='same', name='conv1', subsample=(1, 1, 1),input_shape=(T, img_row, img_col, img_chan)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(64, 3, 3, 3, padding='same', name='conv1_1', subsample=(1, 2, 2),activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, padding='same', name='conv2', subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(128, 3, 3, 3, padding='same', name='conv2_1', subsample=(2, 2, 2),activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3,  padding='same', name='conv3b', subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv3_1', subsample=(2, 2, 2),activation='relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same', name='pool3'))
    # 4th layer group
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv4b', subsample=(2, 2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256, 3, 3, 3, padding='same', name='conv4_1', subsample=(2, 2, 2),activation='relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='same', name='pool4'))
    # 5th layer group
    # model.add(Conv3D(512, 2, 2, 2, padding='same', name='conv5b'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
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
    model.load_weights('./weights/fullvideo_14.139-1.0000.h5')

    # Swap softmax with linear, only needed when visualing softmax layer
    layer_idx = utils.find_layer_idx(model, 'fc8')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    for filt_index in range(0,30):
        visualize_saliency_3Dcnn(model, layer_idx, filter_indices=filt_index, seed_input=get_imgsequence(pre_process=True,
        path='/home/ljg/Desktop/lipnet/data/5sec10/{}/1/'.format(filt_index+1),sampling=True,start_T=1,time_window=48),
        original_img=get_imgsequence(pre_process=False,path='/home/ljg/Desktop/lipnet/data/5sec10/{}/1/'.format(filt_index+1),
                            sampling=True,start_T=1,time_window=48),backprop_modifier= None,save_pathname='fullvideo_vis')


def test():
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'predictions')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    plt.rcParams['figure.figsize'] = (18, 6)

    img1 = utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
    img2 = utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(img1)
    # ax[1].imshow(img2)

    f, ax = plt.subplots(1, 2)

    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        # heatmap = saliency.visualize_cam(model, layer_idx, filter_indices=20, seed_input=img,backprop_modifier='guided')
        heatmap = saliency.visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img,backprop_modifier=None)
        print (np.shape(heatmap))
        # Lets overlay the heatmap onto original image.
        ax[i].imshow(overlay(heatmap,img))

    plt.show()


def _identity(x):
    return x

def deprocess_input(input_array, input_range=(0, 255)):
    """Utility function to scale the `input_array` to `input_range` throwing away high frequency artifacts.

    Args:
        input_array: An N-dim numpy array.
        input_range: Specifies the input range as a `(min, max)` tuple to rescale the `input_array`.

    Returns:
        The rescaled `input_array`.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    input_array = input_array.copy()
    input_array -= input_array.mean()
    input_array /= (input_array.std() + K.epsilon())
    input_array *= 0.1

    # clip to [0, 1]
    input_array += 0.5
    input_array = np.clip(input_array, 0, 1)

    # Convert to `input_range`
    return (input_range[1] - input_range[0]) * input_array + input_range[0]

class Optimizer_3Dcnn(Optimizer):
    def __init__(self):
        super(Optimizer_3Dcnn,self).__init__()

    def _get_seed_input(self, seed_input):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """
        desired_shape = (1, ) + K.int_shape(self.input_tensor)[1:]
        print (desired_shape)
        if seed_input is None:
            return utils.random_array(desired_shape, mean=np.mean(self.input_range),
                                      std=0.05 * (self.input_range[1] - self.input_range[0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        # for i in range(np.shape(seed_input)[0]):
        #     x = seed_input[i,...]
        #     seed_input[i,...] = x
        return seed_input.astype(K.floatx())

    def minimize(self, seed_input=None, max_iter=200,
                 input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.

        Args:
            seed_input: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            input_modifiers: A list of [InputModifier](vis.input_modifiers#inputmodifier) instances specifying
                how to make `pre` and `post` changes to the optimized input during the optimization process.
                `pre` is applied in list order while `post` is applied in reverse order. For example,
                `input_modifiers = [f, g]` means that `pre_input = g(f(inp))` and `post_input = f(g(inp))`
            grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
                specify anything, gradients are unchanged. (Default value = None)
            callbacks: A list of [OptimizerCallback](vis.callbacks#optimizercallback) instances to trigger.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor(s). (Default value = True)

        Returns:
            The tuple of `(optimized input, grads with respect to wrt, wrt_value)` after gradient descent iterations.
        """
        seed_input = self._get_seed_input(seed_input)
        input_modifiers = input_modifiers or []
        grad_modifier = _identity if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        cache = None
        best_loss = float('inf')
        best_input = None

        grads = None
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                seed_input = modifier.pre(seed_input)

            # 0 learning phase for 'test'
            computed_values = self.compute_fn([seed_input, 0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = zip(self.loss_names, losses)
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != wrt_value.shape:
                grads = np.reshape(grads, wrt_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            # for c in callbacks:
            #     c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
            if self.wrt_tensor is self.input_tensor:
                step, cache = self._rmsprop(grads, cache)
                seed_input += step

            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                seed_input = modifier.post(seed_input)

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_input = seed_input.copy()
            print ('best_input',np.shape(best_input))
            for i in range(np.shape(best_input)[1]):
                best_input[0,i,...] = deprocess_input(best_input[0,i,...],self.input_range)

        # Trigger on_end
        # for c in callbacks:
        #     c.on_end()
        return best_input[0], grads, wrt_value
        # return deprocess_input(best_input[0], self.input_range), grads, wrt_value

def visualize_saliency_with_losses(input_tensor, losses, seed_input,original_img, grad_modifier='absolute',save_path=''):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    opt = Optimizer(input_tensor, losses, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)[1]
    # print (np.shape(grads))

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(grads, axis=channel_idx)
    # if not os.path.exists('./image'):
    #     os.mkdir('./images')

    print (np.shape(grads))
    for i in range(np.shape(grads)[1]):
        temp_grads = utils.normalize(grads[:,i,...])
        # print ('temp_grads',np.shape(temp_grads))
        heatmap = np.uint8(cm.jet(temp_grads)[..., :3] * 255)[0]
        img = original_img[i,...]

        temp = image.array_to_img(overlay(img, heatmap,alpha=0.5))
        pil.Image.save(temp,save_path+'overlay{}.jpg'.format(i))

        temp = image.array_to_img(heatmap)
        pil.Image.save(temp,save_path+'heatmap{}.jpg'.format(i))

        temp = image.array_to_img(img)
        pil.Image.save(temp,save_path+'original{}.jpg'.format(i))
    # Normalize and create heatmap.


def visualize_saliency_3Dcnn(model, layer_idx, filter_indices, seed_input,original_img,
                       backprop_modifier=None, grad_modifier='absolute',save_pathname='images'):

    if backprop_modifier is not None:
        modifier_fn = backprop_modifiers.get(backprop_modifier)
        # model = backend.modify_model_backprop(model, 'guided')
        model = modifier_fn(model)


    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    if not os.path.exists('{}/{}'.format(save_pathname,filter_indices+1)):
        os.makedirs('{}/{}'.format(save_pathname,filter_indices+1))

    visualize_saliency_with_losses(model.input, losses, seed_input,original_img, grad_modifier,save_path='{}/{}/'.format(save_pathname,filter_indices+1))


if __name__ == '__main__':

    iden_visualization()