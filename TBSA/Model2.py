import numpy as np
from tensorflow.keras.layers import Dense, dot, MultiHeadAttention, Embedding, Input, Lambda, Add, GRU, Bidirectional, \
    BatchNormalization, Dropout
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
from bert4keras.layers import RelativePositionEmbeddingT5 as rt5
import Model2_F1_Score as f1

'''
training data can be changed when using this model, 
just put the new dataset when model.fit()
'''


def model2(x_train, y_train, x_test, y_test, batch_size, epochs):

    # evaluate function
    class Evaluate(Callback):

        def __init__(self, predict_input, labels):
            self.predict_input = predict_input
            self.labels = labels
            self.record = 0
            self.test_record = 0
            self.precision_record = 0
            self.test_precision_record = 0
            self.recall_record = 0
            self.temp = []

        def on_epoch_end(self, epoch, logs=None):

            F1 = 0
            final = 0

            print("---------------------------------------TEST------------------------------------------------------")
            # test data predict
            predict = create_model.predict(self.predict_input)
            # change to correct shape
            predict = f1.transpose(predict)
            # print(predict)
            predict = f1.change(predict)
            # print(predict)

            test_recall, test_precision= f1.f1score(predict, self.labels)

            if test_recall != 0:
                final = 2 / ((1 / test_recall) + (1 / test_precision))

            print("---------------------------------------TRAIN------------------------------------------------------")

            # train data predict
            predict = create_model.predict(x_train)
            # change to correct shape
            predict = f1.transpose(predict)
            print("predict shape:\n", np.shape(predict))
            # print(predict)
            predict = f1.change(predict)
            print("predict shape:\n", np.shape(predict))
            print(predict[0])
            print(y_train[0])
            # print(predict)

            recall, precision = f1.f1score(predict, y_train)
            if recall != 0:
                F1 = 2 / ((1 / recall) + (1 / precision))
            print("F1-Score:" + str(F1))

            print("----------------------------------------------------------------------------------------")

            if F1 > self.record:
                self.record = F1
                self.test_record = final

            print("the best:", self.record)
            print(self.test_record)




            # if (F1 > (self.record - 0.05) and p_total[4] > (self.precision_record - 0.01) and p_total[4] != 1
            #         and test_p_total[4] != 1 and test_p_total[4] > self.test_precision_record):
            #     self.record = F1
            #     self.precision_record = p_total[4]
            #     self.test_precision_record = test_p_total[4]
            #     self.temp = [recall, precision, r_total, p_total]
            #
            #     # save best output and weights
            #     np.savetxt('TBSA_model_weights/Model2/restaurant/model2_best_result.txt', predict)
            #     create_model.save_weights('TBSA_model_weights/Model2/restaurant/model2_best_weights')
            #
            #
            # print("In test set, best beginning of entity precision:\n"+str(self.test_precision_record))
            # print("In training set, best beginning of entity precision:\n"+str(self.precision_record))
            # print("BEST:")
            # print(self.record, self.temp)


    # input's size(shape)
    input_x = Input(shape=(86, 768))

    # GRU extract deep information
    gru1 = Bidirectional(
        GRU(384,
            kernel_initializer= variance_scaling(seed=None),
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
                             False,Orthogonal(),
                             kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.9, axis=0)
                             )(gru3)

    # 2.for DG-CNN
    dg_cnn = DilatedGatedConv1D(512,
                                rate=int(86/8),
                                drop_gate=0.1,
                                name='dgcnn'
                                )(gru3)

    # 3. for Attention
    attention = MultiHeadAttention(3,
                                   3,
                                   dropout=0.1,
                                   use_bias=False,
                                   name='attention'
                                   )(gru3, gru3, gru3)

    attention_dense = Dense(512,
                             'tanh',
                             False,Orthogonal(),
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

    ensemble_output = Add()([ensemble_dense_1, dg_cnn, attention_dense])

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

    dense_out = Dense(1,
                      'elu',
                      use_bias=False,
                      kernel_initializer=Orthogonal()
                      )(dense_3)

    

    # set input and output
    create_model = Model(inputs=input_x, outputs=dense_out)

    # set loss and optimizer
    # accuracy would be calculated by call back in main function
    create_model.compile(loss='mse',
                         optimizer=Nadam(1e-4))

    # print the info fo model
    create_model.summary()

    evaluator = Evaluate(x_test, y_test)

    create_model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test),
                     validation_freq=1,
                     callbacks=[evaluator])


    return create_model



if __name__ == '__main__':

    # set parameters
    batch_size = 32
    epochs = 120


    # laptop finished
    '''
    # load train data
    laptop_train_x = np.load("TBSA_data_set/Model2&Model3/Train/Input/laptop_train_embedding.npy", allow_pickle=True)
    laptop_train_y = np.load("TBSA_data_set/Model2&Model3/Train/Output/laptop_train_labels.npy", allow_pickle=True)

    # load test data
    laptop_test_x = np.load("TBSA_data_set/Model2&Model3/Test/Input/laptop_test_embedding.npy", allow_pickle=True)
    laptop_test_y = np.load("TBSA_data_set/Model2&Model3/Test/Output/laptop_test_labels.npy", allow_pickle=True)


    # build model
    model = model2(laptop_train_x, laptop_train_y, laptop_test_x, laptop_test_y, batch_size, epochs)
    '''

    '''
    # load train data
    restaurant_train_x = np.load("TBSA_data_set/Model2&Model3/Train/Input/restaurant_train_embedding.npy")
    restaurant_train_y = np.load("TBSA_data_set/Model2&Model3/Train/Output/restaurant_train_labels.npy")

    # load test data
    restaurant_test_x = np.load("TBSA_data_set/Model2&Model3/Test/Input/restaurant_test_embedding.npy")
    restaurant_test_y = np.load("TBSA_data_set/Model2&Model3/Test/Output/restaurant_test_labels.npy")

    model = model2(restaurant_train_x, restaurant_train_y, restaurant_test_x, restaurant_test_y, batch_size, epochs)
    '''

    # load train data
    restaurant_train_x = np.load("model2_train_v.npy")
    restaurant_train_y = np.load("model2_train_pos012.npy")
    restaurant_train_y = restaurant_train_y*10

    print(np.shape(restaurant_train_x))

    restaurant_test_x = restaurant_train_x[549: 829]
    restaurant_test_y = restaurant_train_y[549: 829]

    restaurant_train_x = restaurant_train_x[1: 548]
    restaurant_train_y = restaurant_train_y[1: 548]




    # # load test data
    # restaurant_test_x = np.load("model2_test_v.npy")
    # restaurant_test_y = np.load("model2_test_pos012.npy")
    # restaurant_test_y = restaurant_test_y*10

    model = model2(restaurant_train_x, restaurant_train_y, restaurant_test_x, restaurant_test_y, batch_size, epochs)



