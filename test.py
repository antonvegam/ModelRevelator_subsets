import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras

import numpy as np
import glob
import os
import datetime
import sys
import pandas as pd

from tensorboard.backend.event_processing.event_file_loader import _make_tf_record_iterator


train_data_path = os.getenv('TRAIN_DATA_DIR')

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(floatmode='fixed')

kernel_size = 5
filters = 256
pool_size = 4

batch_size = 30

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def decode(serialized_example):
    with tf.device('/cpu:0'):
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'train/label': tf.io.FixedLenFeature([], tf.string),
                'train/subsets_msa': tf.io.FixedLenFeature([], tf.string),
                'train/colwise_random': tf.io.FixedLenFeature([], tf.string),
                'train/pairwise_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/pairwise_random': tf.io.FixedLenFeature([], tf.string),
                'train/complex_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/seed': tf.io.FixedLenFeature([], tf.int64),
                'train/sec_seed': tf.io.FixedLenFeature([], tf.int64),
                'train/alpha': tf.io.FixedLenFeature([], tf.float32),
                'train/ev_model': tf.io.FixedLenFeature([], tf.string),
                'train/seq_lengths': tf.io.FixedLenFeature([], tf.int64),
                'train/taxa': tf.io.FixedLenFeature([], tf.int64),
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

        # these 4 lines allow to reduce the 12 models to the 6 actual models and thus spares us geneating training data twice.
        l = tf.argmax(label, axis=0)
        label = tf.cond(l > 5, lambda: l - 6, lambda: l)
        label = tf.one_hot(label, 12)
        #tf.print(label, summarize=-1)

        seed = features['train/seed']
        sec_seed = features['train/sec_seed']
        alpha = features['train/alpha']

        alpha = tf.cond(alpha < 0, lambda : tf.constant(50.0), lambda: alpha)
        alpha = tf.math.multiply(alpha, 1000)

        seq_len = features['train/seq_lengths']
        taxa = features['train/taxa']

        ev_model = tf.strings.unicode_decode(features['train/ev_model'], 'utf-8')
        ev_model = tf.strings.unicode_encode(ev_model, 'UTF-8')

    return (pairwise_random, ),  label, seed, sec_seed, alpha, ev_model, seq_len, taxa


def decode_input_only(serialized_example):
    with tf.device('/cpu:0'):
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'train/label': tf.io.FixedLenFeature([], tf.string),
                'train/subsets_msa': tf.io.FixedLenFeature([], tf.string),
                'train/colwise_random': tf.io.FixedLenFeature([], tf.string),
                'train/pairwise_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/pairwise_random': tf.io.FixedLenFeature([], tf.string),
                'train/complex_subsets': tf.io.FixedLenFeature([], tf.string),
                'train/seed': tf.io.FixedLenFeature([], tf.int64),
                'train/sec_seed': tf.io.FixedLenFeature([], tf.int64),
                'train/alpha': tf.io.FixedLenFeature([], tf.float32),
                'train/ev_model': tf.io.FixedLenFeature([], tf.string),
                'train/seq_lengths': tf.io.FixedLenFeature([], tf.int64),
                'train/taxa': tf.io.FixedLenFeature([], tf.int64),
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


        

    return pairwise_random

def data_generator(data_files, input_only):
    with tf.device('/cpu:0'):

        print(data_files)
        tf_train_data = tf.data.TFRecordDataset(data_files)

        if input_only:
            tf_train_data = tf_train_data.map(decode_input_only)
        else:
            tf_train_data = tf_train_data.map(decode)

        tf_train_data = tf_train_data.repeat()

        # Separate training data into batches
        tf_train_data = tf_train_data.batch(batch_size)
        tf_train_data = tf_train_data.prefetch(10)

        return tf_train_data


model_name = '/scratch/anton/experiments/weights.28-1.80_0.7349_1000pos.hdf5'

model = load_model(model_name)
model.summary()

np.set_printoptions(threshold=sys.maxsize)

base_test_data_path = '/scratch/anton/test_data/test_1000bp'

model_int_map = {
    'JC': 0,
    'K2P': 1,
    'K80': 1,
    'F81': 2,
    'HKY': 3,
    'TN93': 4,
    'GTR': 5,
    'JC+G': 6,
    'K2P+G': 7,
    'K80+G': 7,
    'F81+G': 8,
    'HKY+G': 9,
    'TN93+G': 10,
    'GTR+G': 11,

    'K3P': 12,
    'AK2P': 13,
    'TNEG': 14,
    'SYM': 15,
}

model_int_map = {
    'JC': 0,
    'K2P': 1,
    'F81': 2,
    'HKY': 3,
    'TN93': 4,
    'GTR': 5,
    'JC+G': 0,
    'K2P+G': 1,
    'F81+G': 2,
    'HKY+G': 3,
    'TN93+G': 4,
    'GTR+G': 5,

    'K3P': 12,
    'AK2P': 13,
    'TNEG': 14,
    'SYM': 15,
}


for o in glob.glob(base_test_data_path + '/*'):

    f = [o]# glob.glob(os.path.join(o, '*.tfrecords'))
    print(f)

    results_file_name = 'ranpair_28_1000'




    df = None
    for file in glob.glob(base_test_data_path + '/*'):
        # Count number of test examples per tfrecord file,
        # make sure the file count is at least a batch_size larger than the actual count, so all samples get tested at least once.
        test_examples_per_tfrecords_file = batch_size
        for _ in _make_tf_record_iterator(file):
            test_examples_per_tfrecords_file += 1

        print(file, 'test examples in file',  test_examples_per_tfrecords_file)
    
        x,  y, seed, sec_seed, alpha, ev_model, seq_len, taxa_iterator = iter(data_generator(file, input_only=False)).get_next()
        y_proba = model.predict(data_generator(file, input_only=True), steps=test_examples_per_tfrecords_file // batch_size)
        final_results = np.argmax(y_proba, axis=1)
        
        rev_map = dict(zip(model_int_map.values(), model_int_map.keys()))
        final_results = [rev_map[x] for x in final_results]

        true_alphas = []
        true_labels = []
        primary_seed = []
        secondary_seed = []
        evolutionary_model = []
        f_names = []
        seq_lengths = []
        taxa = []
    
        for _ in range(test_examples_per_tfrecords_file // batch_size):
            tl, a, s, ss, true_m, sl, ta = [y.numpy(), alpha.numpy(), seed.numpy(), sec_seed.numpy(), ev_model.numpy(), seq_len.numpy(), taxa_iterator.numpy()]
            a = [int(x / 1000) if x >= 1000 else x / 1000 for x in a]
            true_alphas.extend(a)
            primary_seed.extend(s)
            secondary_seed.extend(ss)
            evolutionary_model.extend([x.decode() for x in true_m])
            true_labels.extend(np.argmax(tl, axis=1))
            # I mixed up taxa and seq length, so just switching those here!!!
            seq_lengths.extend(ta)
            taxa.extend(sl)

        # reconstruct file names
        for c, true_alpha in enumerate(true_alphas):
            if evolutionary_model[c].endswith('+G'):
                ff = '{}_{}Taxa_{}__alpha{}__{}_{}.phy'.format(primary_seed[c], taxa[c], evolutionary_model[c], true_alphas[c] , seq_lengths[c],
                                                   secondary_seed[c])
            else:
                ff = '{}_{}Taxa_{}__{}_{}.phy'.format(primary_seed[c], taxa[c], evolutionary_model[c], seq_lengths[c], secondary_seed[c])

            f_names.append(ff)

        nn_model = model_name.split('full_models')[-1:] * len(true_labels)

        t_df = pd.DataFrame({'seed': primary_seed, 'sec_seed': secondary_seed, 'ev_model': evolutionary_model, 'true_model': true_labels,  'estimated_model': final_results, 'true_alpha': true_alphas,
                             'taxa': taxa, 'sequence_length': seq_lengths, 'file_name': f_names,
                             'jc_prob': [x[0] for x in y_proba],
                             'k2p_prob': [x[1] for x in y_proba],
                             'f81_prob': [x[2] for x in y_proba],
                             'hky_prob': [x[3] for x in y_proba],
                             'tn93_prob': [x[4] for x in y_proba],
                             'gtr_prob': [x[5] for x in y_proba],

        })
        if df is None:
            df = t_df
        else:
            df = df.append(t_df)
    
    print('Model name:', model_name)
    print('test dataset:', results_file_name)
    df.to_csv('/scratch/anton/test_results/' + results_file_name + '.csv')
    print(df.describe())
    break