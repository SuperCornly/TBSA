import numpy as np
from tensorflow.keras.layers import Dense, dot, MultiHeadAttention, Embedding, Input, Lambda, Add, GRU, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from mylayer import AttentionPooling1D, Capsule, ONLSTM, DilatedGatedConv1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
from tensorflow.keras.initializers import *
from tensorflow.keras.constraints import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import kl_divergence
from bert4keras.layers import ConditionalRandomField
from bert4keras.layers import RelativePositionEmbeddingT5 as rt5
import Model3_F1_Score as rp


def model3(x_train, y_train, x_test, y_test, batch_size, epochs):

    class Evaluate(Callback):
        def __init__(self, x_test, y_test):
            self.x_test = x_test
            self.y_test = y_test
            self.record = 0
            self.precision = 0
            self.test_best = 0
            self.temp = []

        def on_epoch_end(self, epoch, logs=None):

            print("---------------------------------------TEST--------------------------------------------------------")
            # pad 0.0 \开头 0.1 \结尾0.2 \非实体 0.3 \实体 0.4 \实体尾巴 0.5
            # predict
            predict = train_model.predict(self.x_test)

            # print(np.shape(self.y_test))
            # print("real labels:\n" + str(self.y_test))
            # print("--------------------------------------------------------------------------")

            predict = rp.softmax_to_label(predict)
            # print("predict:\n" + str(predict))
            # print("--------------------------------------------------------------------------")

            recall, precision, test_r_total, test_p_total, test_matrix = rp.f1score(predict, y_test)

            F1 = 2 / ((1 / recall) + (1 / precision))
            print("F1-Score:" + str(F1))


            print("---------------------------------------TRAIN-------------------------------------------------------")

            predict = train_model.predict(x_train)
            predict = rp.softmax_to_label(predict)

            print("predict:\n", predict[18])
            print("labels:\n", y_train[18])

            recall, precision, r_total, p_total, matrix = rp.f1score(predict, y_train)

            F1 = 2 / ((1 / recall) + (1 / precision))
            print("F1-Score:" + str(F1))


            print("---------------------------------------BEST--------------------------------------------------------")

            if F1 > (self.record - 0.03) and p_total[4] > self.precision and p_total[4] != 1:
                self.record = F1
                self.precision = p_total[4]
                self.temp = [recall, precision, r_total, p_total]
                self.test_best = test_p_total[4]

                # save best output and weights
                np.savetxt('TBSA_model_weights/Model3/laptop/model3_best_result.txt', predict)
                train_model.save_weights('TBSA_model_weights/Model3/laptop/model3_best_weights')
                np.savetxt('TBSA_model_weights/Model3/laptop/model3_best_matrix.txt', matrix)

            print("BEST:")
            print(self.record, "\n", self.temp)


    input_x = Input(shape=(96, 768))

    # GRU extract deep information
    gru1 = Bidirectional(
        GRU(384,
            kernel_initializer=variance_scaling(seed=None),
            bias_initializer=glorot_normal(seed=None),
            bias_constraint=UnitNorm(axis=0),
            return_sequences=True,
            dropout=0.01)
    )(input_x)

    gru2 = Bidirectional(
        GRU(384,
            kernel_initializer=variance_scaling(seed=None),
            bias_initializer=glorot_normal(seed=None),
            bias_constraint=UnitNorm(axis=0),
            return_sequences=True,
            dropout=0.01
            )
    )(gru1)

    gru3 = Bidirectional(
        GRU(384,
            kernel_initializer=variance_scaling(seed=None),
            bias_initializer=glorot_normal(seed=None),
            bias_constraint=UnitNorm(axis=0),
            return_sequences=True,
            dropout=0.01
            )
    )(gru2)

    '''
    # ensemble
    # dense by order
    # ensemble DG-CNN,Attention
    '''
    # 1.just dense
    ensemble_dense_1 = Dense(512,
                             'tanh',
                             False, Orthogonal(),
                             kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.9, axis=0)
                             )(gru3)

    # 2.for DG-CNN
    dg_cnn = DilatedGatedConv1D(512,
                                rate=int(86 / 8),
                                drop_gate=0.1,
                                name='dgcnn'
                                )(gru3)

    dense_dg= Dense(512,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(dg_cnn)


    # 3. for Attention
    attention = MultiHeadAttention(3,
                                   3,
                                   dropout=0.1,
                                   use_bias=False,
                                   name='attention'
                                   )(gru3, gru3, gru3)

    attention_dense = Dense(512,
                            'tanh',
                            False, Orthogonal(),
                            kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.9, axis=0)
                            )(attention)

    on_lstm = ONLSTM(256, 4, True, 0.09, bench=1, name='onlstm')(gru3)

    '''
    # 4. for ON-LSTM, temporarily ignored
    ensemble_dense_4 = Dense(256,
                             'tanh',
                             False, Orthogonal(),
                             kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.9, axis=0)
                             )(gru3)
    '''

    ensemble_output = Add()([ensemble_dense_1, dense_dg, attention_dense])

    # final dense compress and output
    # ensemble
    dense_1 = Dense(256,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(ensemble_output)

    # not ensemble
    # dense_1 = Dense(256,
    #                 'tanh',
    #                 use_bias=False,
    #                 kernel_initializer=Orthogonal()
    #                 )(gru3)

    # just dg-cnn
    # dense_1 = Dense(256,
    #                 'tanh',
    #                 use_bias=False,
    #                 kernel_initializer=Orthogonal()
    #                 )(dg_cnn)

    dense_2 = Dense(128,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(dense_1)

    dense_3 = Dense(64,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(dense_2)

    dense_4 = Dense(32,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(dense_3)

    dense_5 = Dense(16,
                    'tanh',
                    use_bias=False,
                    kernel_initializer=Orthogonal()
                    )(dense_4)

    out_crf = Dense(6,
                    kernel_initializer=Orthogonal()
                    )(dense_5)

    # out_crf = Dense(1,
    #                 kernel_initializer=Orthogonal(),
    #                 kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.3, axis=0))(dense_4)

    CRF = ConditionalRandomField(lr_multiplier=1000)
    output = CRF(out_crf)

    train_model = Model(input_x, output)
    # train_model = Model(input_x,out_crf)

    train_model.summary()

    train_model.compile(loss=CRF.sparse_loss,
                        optimizer=Nadam(1e-5),
                        metrics=[CRF.sparse_accuracy]
                        )



    evaluator = Evaluate(x_test, y_test)

    train_model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[evaluator])

    return train_model




if __name__ == '__main__':

    # set parameters
    batch_size = 32
    epochs = 120

    # load laptop train data
    laptop_train_x = np.load('TBSA_data_set/Model2&Model3/Train/Input/laptop_train_embedding.npy', allow_pickle=True)
    laptop_train_y = np.load('TBSA_data_set/Model2&Model3/Train/Output/laptop_train_labels.npy', allow_pickle=True)

    # load laptop test data
    laptop_test_x = np.load('TBSA_data_set/Model2&Model3/Test/Input/laptop_test_embedding.npy', allow_pickle=True)
    laptop_test_y = np.load('TBSA_data_set/Model2&Model3/Test/Output/laptop_test_labels.npy', allow_pickle=True)

    # build model
    model = model3(laptop_train_x, laptop_train_y, laptop_test_x, laptop_test_y, batch_size, epochs)

