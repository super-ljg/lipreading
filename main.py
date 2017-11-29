from iden_test import iden_test
from iden_train import get_model,preprocess
import sys


if __name__ == '__main__':
    if len(sys.argv)==2:
        image_path = sys.argv[1]

    model = get_model()
    print 'It takes long time to train the model. GPU is needed.'


