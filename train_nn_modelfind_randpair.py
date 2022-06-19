"""
Author: Sebastian Burgstaller-Muehlbacher
"""

import tensorflow as tf

from tensorflow import keras
import glob
import os
import yaml


from resnet import ResnetBuilder

with open('rename_randpair.yml', 'r') as file:
    yaml_file = yaml.safe_load(file)
'''
assert os.getenv('LOG_DIR'), 'Logging directory for model saving not set!'
assert os.getenv('TRAIN_DATA_DIR'), 'TRAIN_DATA_PATH not set! This is where training data is read from.'
assert os.getenv('TEST_DATA_DIR'), 'TEST_DATA_PATH not set! This is where training data is read from.'
'''

train_data_path = yaml_file['TRAIN_DATA_DIR']
test_data_path = yaml_file['TEST_DATA_DIR']
log_dir = yaml_file['LOG_DIR']

print(train_data_path, yaml_file['TRAIN_DATA_DIR'])
print(test_data_path, yaml_file['TRAIN_DATA_DIR'])


# Training
batch_size = 40
epochs = 10000

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def decode(serialized_example):
    with tf.device('/cpu:0'):
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'train/label': tf.io.FixedLenFeature([], tf.string),
                #'train/subsets_msa': tf.io.FixedLenFeature([], tf.string),
                #'train/colwise_random': tf.io.FixedLenFeature([], tf.string),
                #'train/pairwise_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/pairwise_random': tf.io.FixedLenFeature([], tf.string),
                'train/complex_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/seed': tf.io.FixedLenFeature([], tf.int64),
                'train/sec_seed': tf.io.FixedLenFeature([], tf.int64),
                'train/alpha': tf.io.FixedLenFeature([], tf.float32),
                'train/ev_model': tf.io.FixedLenFeature([], tf.string),
            })

        '''
        subsets = tf.io.decode_raw(features['train/subsets_msa'], tf.float16)
        subsets = tf.reshape(subsets, shape=[200, 200, 16])
        
        
        colwise_random = tf.io.decode_raw(features['train/colwise_random'], tf.float16)
        colwise_random = tf.reshape(colwise_random, shape=[100, 100, 16])
        
        pairwise_subsets = tf.io.decode_raw(features['train/pairwise_subsets'], tf.float16)
        pairwise_subsets = tf.reshape(pairwise_subsets, shape=[200, 200, 26])
        '''
        pairwise_random = tf.io.decode_raw(features['train/pairwise_random'], tf.float16)
        pairwise_random = tf.reshape(pairwise_random, shape=[100, 100, 26])
        '''
        complex_subsets = tf.io.decode_raw(features['train/complex_subsets'], tf.float16)
        complex_subsets = tf.reshape(complex_subsets, shape=[100, 100, 26])
        '''
        label = tf.io.decode_raw(features['train/label'], tf.int8)
        label = tf.reshape(label, shape=[12])

        # these 4 lines allow to reduce the 12 models to the 6 actual models and thus spares us geneating training data twice. 
        l = tf.argmax(label, axis=0)
        label = tf.cond(l > 5, lambda: l - 6, lambda: l)
        label = tf.one_hot(label, 12)
        #tf.print(label, summarize=-1)

        # seed = features['train/seed']
        # sec_seed = features['train/sec_seed']
        alpha = features['train/alpha']
        
        alpha = tf.cond(alpha < 0, lambda : tf.constant(50.0), lambda: alpha)
        alpha = tf.math.multiply(alpha, 1000)

    return (pairwise_random, ),  label


def data_generator(data_files):
    with tf.device('/cpu:0'):
        print(data_files)
        tf_train_data = tf.data.TFRecordDataset(data_files)
        tf_train_data = tf_train_data.map(decode)

        tf_train_data = tf_train_data.repeat()
        tf_train_data = tf_train_data.shuffle(3500)

        # Separate training data into batches
        tf_train_data = tf_train_data.batch(batch_size)
        tf_train_data = tf_train_data.prefetch(2)

        return tf_train_data


print('Build model...')


model = ResnetBuilder.build_resnet_18((26, 100, 100), 12)
#model = tf.keras.models.load_model(log_dir + '/weights.8424-1.47_0.7856_1000pos.hdf5')

adam = keras.optimizers.Adam(lr=yaml_file['lr'], ) 
model.compile(loss={'dense': 'categorical_crossentropy', },
              optimizer=adam,
              metrics={'dense': 'accuracy',} 
)

model.summary()


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


print('Loading data...')
train_data_files = glob.glob(os.path.join(train_data_path, '*.tfrecords'))
print('train data files', train_data_files) 

test_data_files = glob.glob(os.path.join(test_data_path,  '*.tfrecords'))

print('Train...')

model.fit(data_generator(train_data_files),
          batch_size=None,
          steps_per_epoch= yaml_file['epoch_steps'] // batch_size,
          epochs=epochs,
          callbacks=[keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'weights.{epoch:02d}-{val_loss:.2f}_{accuracy:.4f}_1000pos.hdf5'),  monitor='val_loss', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1), tensorboard_callback],
          verbose=2,
          validation_data=(data_generator(test_data_files)),
          validation_steps= yaml_file['validation_steps'] // batch_size,
          initial_epoch=0
          )