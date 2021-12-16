# coding=utf-8
import numpy as np
from tensorflow.keras.layers import Dense, dot, MultiHeadAttention, Embedding, Input, Lambda, Add, GRU, Bidirectional, \
    BatchNormalization, Dropout, Conv1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from mylayer import AttentionPooling1D, Capsule, ONLSTM, DilatedGatedConv1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
from tensorflow.keras.initializers import *
from tensorflow.keras.constraints import *
from tensorflow.keras.regularizers import *
from bert4keras.layers import LayerNormalization
import Model4_F1_Score as f1

'''
max_len=84
bench=16
len_v=max_len+2
size=768
max_len_id=22
len_aspect=max_len_id+2
'''


def model4(target_x, sentence_x, polarity_y, test_target_x, test_sentence_x, test_polarity_y,
           max_len=96, size=768, target_length=16):

    class Evaluate(Callback):

        def __init__(self):
            self.metrics = []
            self.best_record = 0
            self.precision_record = 0
            self.recall_record = 0
            self.test_best_record = 0
            self.test_precision_record = 0
            self.test_recall_record = 0


        def on_epoch_end(self, epoch, logs=None):
            predict = model.predict([test_target_x, test_sentence_x])

            test_predict_save, test_recall, test_precision, test_f1 = f1.f1_score(predict, test_polarity_y)
            print('---------------------------------------------------------------------------------------------------')

            predict = model.predict([target_x, sentence_x])

            predict_save, recall, precision, f1_score= f1.f1_score(predict, polarity_y)
            print(np.shape(predict_save))

            if f1_score > self.best_record:
                self.best_record =f1_score
                self.precision_record = precision
                self.recall_record = recall
                self.test_best_record = test_f1
                self.test_precision_record = test_precision
                self.test_recall_record = test_recall

                np.savetxt('TBSA_model_weights/Model4/restaurant/model4_best_result.txt', predict_save)
                model.save_weights('TBSA_model_weights/Model4/restaurant/model4_best_weights')

            print('---------------------------------------------------------------------------------------------------')
            print("best test recall:\n", self.test_recall_record)
            print("best test precision:\n", self.test_precision_record)
            print("best test f1:\n", self.test_best_record)
            print('---------------------------------------------------------------------------------------------------')
            print("best recall:\n", self.recall_record)
            print("best precision:\n", self.precision_record)
            print("best f1:\n", self.best_record)
            print('---------------------------------------------------------------------------------------------------')




    target = Input(shape=(target_length, size))  # 实体 取字bert后截取
    v_in = Input(shape=(max_len, size))  # 输入语料
    # 实体
    target_attention = AttentionPooling1D()(target)

    target_attention = Dense(128, use_bias=False
                            )(target_attention)
    # v
    v_in1 = Dense(size,
                  kernel_initializer=Orthogonal()
                  )(v_in)
    v_in1 = Dropout(0.01)(v_in1)

    v_in2 = Dense(512,
                  kernel_initializer=Orthogonal()
                  )(v_in1)
    v_in2 = Dropout(0.01)(v_in2)

    # condition
    v_aspect = LayerNormalization(conditional=True
                                  )([v_in2, target_attention])

    t2 = Dense(512
               )(v_aspect)
    t2 = Dropout(0.1)(t2)

    t3 = Dense(256
               )(t2)
    t3 = Dropout(0.1)(t3)

    t5 = Dense(128
               )(t3)
    t5 = Dropout(0.1)(t5)

    t6 = Dense(64
               )(t5)
    t6 = Dropout(0.1)(t6)

    t7 = Dense(32
               )(t6)
    t7 = Dropout(0.1)(t7)

    cap_v_asp = Capsule(3, 16, share_weights=True
                        )(t7)

    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(3,))(cap_v_asp)

    print("------------------------------------------------------")
    print("[aspect_in,v_in]:\n" + str([target, v_in]))
    print("------------------------------------------------------")

    model = Model([target, v_in], output)

    model.compile(loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2 + 0.25 * (1 - y_true) * K.relu(y_pred - 0.1) ** 2,
                  optimizer=Nadam(1e-4),
                  metrics=['accuracy'])

    model.summary()

    evaluator = Evaluate()

    model.fit([target_x, sentence_x], polarity_y,
              batch_size=64,
              epochs=120,
              verbose=1,
              validation_data=([test_target_x, test_sentence_x], test_polarity_y),
              callbacks=[evaluator])

'''
def model4(target_x, sentence_x, polarity_y, test_target_x, test_sentence_x, test_polarity_y, max_len=128,
           size=768, target_length=16):
'''

if __name__ == '__main__':

    '''
    # laptop train
    laptop_train_target_x = np.load('TBSA_data_set/Model4/Train/Input/targets/laptop_train_targets.npy',
                                    allow_pickle=True)
    laptop_train_sentence_x = np.load('TBSA_data_set/Model4/Train/Input/sentences/laptop_train_sentences.npy',
                                      allow_pickle=True)
    laptop_train_polarities_y = np.load('TBSA_data_set/Model4/Train/Output/laptop_train_polarities.npy',
                                        allow_pickle=True)

    # print(np.shape(laptop_train_sentence_x))
    # print(np.shape(laptop_train_target_x))
    # print(np.shape(laptop_train_polarities_y))

    laptop_train_polarities_y = laptop_train_polarities_y.astype(float)
    # print(laptop_train_polarities_y.dtype)

    # laptop test
    laptop_test_target_x = np.load('TBSA_data_set/Model4/Test/Input/targets/laptop_test_targets.npy',
                                   allow_pickle=True)
    laptop_test_sentence_x = np.load('TBSA_data_set/Model4/Test/Input/sentences/laptop_test_sentences.npy',
                                     allow_pickle=True)
    laptop_test_polarities_y = np.load('TBSA_data_set/Model4/Test/Output/laptop_test_polarities.npy',
                                       allow_pickle=True)

    laptop_test_polarities_y = laptop_test_polarities_y.astype(float)

    model4(laptop_train_target_x, laptop_train_sentence_x, laptop_train_polarities_y,
           laptop_test_target_x, laptop_test_sentence_x, laptop_test_polarities_y)
    '''

    # res train
    res_train_target_x = np.load('TBSA_data_set/Model4/Train/Input/targets/restaurant_train_targets.npy',
                                 allow_pickle=True)
    res_train_sentence_x = np.load('TBSA_data_set/Model4/Train/Input/sentences/restaurant_train_sentences.npy',
                                   allow_pickle=True)
    res_train_polarities_y = np.load('TBSA_data_set/Model4/Train/Output/restaurant_train_polarities.npy',
                                     allow_pickle=True)

    res_train_polarities_y = res_train_polarities_y.astype(float)


    res_test_target_x = np.load('TBSA_data_set/Model4/Test/Input/targets/restaurant_test_targets.npy',
                                   allow_pickle=True)
    res_test_sentence_x = np.load('TBSA_data_set/Model4/Test/Input/sentences/restaurant_test_sentences.npy',
                                     allow_pickle=True)
    res_test_polarities_y = np.load('TBSA_data_set/Model4/Test/Output/restaurant_test_polarities.npy',
                                       allow_pickle=True)

    res_test_polarities_y = res_test_polarities_y.astype(float)

    model4(res_train_target_x, res_train_sentence_x, res_train_polarities_y,
           res_test_target_x, res_test_sentence_x, res_test_polarities_y)
