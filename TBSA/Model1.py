import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Add, GRU, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from mylayer import AttentionPooling1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam
from tensorflow import keras
import tensorflow as tf
import Model1_Accuracy as ac
np.set_printoptions(suppress=True)


def model1(train_x, train_y, test_x, test_y, focal_alpha, max_len, size, target_amount):
    # 生成位置向量输入

    # 堆模型
    # #输入
    class Evaluate(Callback):
        def __init__(self, evaluate_test_x, evaluate_test_y):
            self.evaluate_x = evaluate_test_x
            self.evaluate_y = evaluate_test_y
            self.best_accuracy = 0
            self.test_best_accuracy = 0

        def on_epoch_end(self, epoch, logs=None):
            predict = train_model.predict(self.evaluate_x)

            test_accuracy, amount = ac.accuracy(predict, self.evaluate_y)
            print(np.shape(predict))
            print(test_accuracy, amount)

            predict = train_model.predict(train_x)

            accuracy, amount = ac.accuracy(predict, train_y)
            print(np.shape(predict))
            print(accuracy, amount)

            if accuracy > self.best_accuracy and test_accuracy > self.test_best_accuracy-0.05:
                self.best_accuracy = accuracy
                self.test_best_accuracy = test_accuracy

                # save best output and weights
                np.savetxt('TBSA_model_weights/Model1/restaurant/model1_best_result.txt', predict)
                train_model.save_weights('TBSA_model_weights/Model1/restaurant/model1_best_weights')

            print("------------------------------------------BEST-----------------------------------------------------")
            print("the train best:\n", self.best_accuracy)
            print("the test  best:\n", self.test_best_accuracy)

    input = Input(shape=(96, 768))

    gru1 = Bidirectional(
        GRU(256,
            kernel_initializer=keras.initializers.variance_scaling(seed=None),
            bias_initializer=keras.initializers.glorot_normal(seed=None),
            bias_constraint=keras.constraints.UnitNorm(axis=0),
            return_sequences=True)
    )(input)
    gru1 = Dropout(0.01)(gru1)

    gru2 = Bidirectional(
        GRU(256,
            kernel_initializer=keras.initializers.variance_scaling(seed=None),
            bias_initializer=keras.initializers.glorot_normal(seed=None),
            bias_constraint=keras.constraints.UnitNorm(axis=0),
            return_sequences=True)
    )(gru1)
    gru2 = Dropout(0.01)(gru2)
    gru2 = AttentionPooling1D()(gru2)

    dense1 = Dense(256, use_bias=False,
                   kernel_constraint=keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.8, axis=0)
                   )(gru2)

    dense2 = Dense(128, use_bias=False,
                   kernel_constraint=keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.8, axis=0)
                   )(dense1)

    dense3 = Dense(64, use_bias=False,
                   kernel_constraint=keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.8, axis=0)
                   )(dense2)

    dense4 = Dense(32, use_bias=False,
                   kernel_constraint=keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=0.9, axis=0)
                   )(dense3)

    output = Dense(16, use_bias=False, activation='softmax',
                   kernel_constraint=keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0))(
        dense4)


    def multi_category_focal_loss1(alpha, gamma=2.0):
        """
        focal loss for multi category of multi label problem
        适用于多分类或多标签问题的focal loss
        alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
        当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
        Usage:
         model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        epsilon = 1.e-7
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = float(gamma)

        def multi_category_focal_loss1_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
            ce = -tf.math.log(y_t)
            weight = tf.math.pow(tf.subtract(1., y_t), gamma)
            fl = tf.linalg.matmul(tf.multiply(weight, ce), alpha)
            loss = tf.math.reduce_mean(fl)
            return loss

        return multi_category_focal_loss1_fixed

    train_model = Model(input, output)

    train_model.compile(optimizer=Nadam(1e-4),
                        metrics=['accuracy'],
                        loss=multi_category_focal_loss1(focal_alpha)
                        )
    train_model.summary()

    # 利用输入和真实的比较的
    # train_D=data_generator(,,,bench)

    evaluator = Evaluate(test_x, test_y)
    train_model.fit(x=train_x,
                    y=train_y,
                    batch_size=64,
                    epochs=120,
                    callbacks=[evaluator]
                    )


if __name__ == '__main__':

    """
    simple_num=8
    bench=16
    len_v=128
    size=768
    v=np.ones(shape=(bench*3,len_v,size))
    true_simple=np.ones(shape=(bench*3,simple_num))
    focal_alpha=np.ones(shape=(16,1))
    """

    # initial some hyper parameter
    # focal_alpha = np.ones(shape=(16, 1))
    target_amount = 16
    max_len = 96
    size = 768

    # the focal alpha is determined by the frequency of the amount of targets
    laptop_focal_alpha = np.load('TBSA_data_set/Model1/laptop_focal_alpha.npy')
    restaurant_focal_alpha = np.load('TBSA_data_set/Model1/restaurant_focal_alpha.npy')

    laptop_focal_alpha = [[1],
                          [0.375],
                          [0.77],
                          [0.91],
                          [0.97],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99],
                          [0.99]]

    print(np.shape(laptop_focal_alpha))

    restaurant_focal_alpha = [[1],
                              [0.50],
                              [0.71],
                              [0.87],
                              [0.95],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99],
                              [0.99]]




    # # laptop train
    # laptop_train_x = np.load('TBSA_data_set/Model1/Train/Input/laptop_train_embedding.npy', allow_pickle=True)
    # laptop_train_y = np.load('TBSA_data_set/Model1/Train/Output/laptop_model1_train_result.npy', allow_pickle=True)
    #
    # laptop_test_x = np.load('TBSA_data_set/Model1/Test/Input/laptop_test_embedding.npy', allow_pickle=True)
    # laptop_test_y = np.load('TBSA_data_set/Model1/Test/Output/laptop_model1_test_result.npy', allow_pickle=True)
    #
    # model1(laptop_train_x, laptop_train_y, laptop_test_x, laptop_test_y, laptop_focal_alpha,
    #        max_len, size, target_amount)

    # restaurant train
    restaurant_train_x = np.load('TBSA_data_set/Model1/Train/Input/restaurant_train_embedding.npy', allow_pickle=True)
    restaurant_train_y = np.load('TBSA_data_set/Model1/Train/Output/restaurant_model1_train_result.npy', allow_pickle=True)

    restaurant_test_x = np.load('TBSA_data_set/Model1/Test/Input/restaurant_test_embedding.npy', allow_pickle=True)
    restaurant_test_y = np.load('TBSA_data_set/Model1/Test/Output/restaurant_model1_test_result.npy', allow_pickle=True)

    model1(restaurant_train_x, restaurant_train_y, restaurant_test_x, restaurant_test_y, restaurant_focal_alpha,
           max_len, size,target_amount)




