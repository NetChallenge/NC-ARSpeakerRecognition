
import sys
import tensorflow as tf
import train
import test

if __name__ == "__main__":

    if len(sys.argv) > 1:

        if sys.argv[1] == 'train':
            train.run(sys.argv)
        else :
            test.run(sys.argv)



    else :
        raise Exception('set model mode "train" or "eval"')





