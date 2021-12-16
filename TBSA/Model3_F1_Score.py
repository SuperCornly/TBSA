import numpy as np

# (33, 86, 6)

test = [[[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8]],
        [[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8]],
        [[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8],[1.2, 1.6, 1.8, 1.2, 1.6, 1.8]]]

test1 = [[1,2,3],
         [4,2,0],
         [4,2,5]]

test_predict = [[1,1,22,35,14,2,2,1,4,5,0],
                [1,1,2,3,1,2,2,1,4,5,0]]

test_real    = [[1,1,3,2,3,2,2,1,4,5,0],
                [1,1,3,2,3,2,2,1,4,5,0]]


def softmax_to_label(predict):

    sample_amount = len(predict)

    save_predict = []

    for i in range(sample_amount):

        save_predict.append([])

        for j in range(len(predict[i])):

            max_softmax = -999
            label = -1

            for k in range(len(predict[i][j])):

                if predict[i][j][k] > max_softmax:
                    max_softmax = predict[i][j][k]
                    label = k

            save_predict[i].append(label)

    return save_predict

# F1 sore
# actual matrix of predict and real are two dimensions
def f1score(predict, real):

    # Confusion matrix
    matrix = np.zeros([6, 6])

    # print matrix
    print("initialize the f1-calculating matrix:")
    print(matrix)

    for i in range(len(predict)):
        # length_j = len(predict[i])
        for j in range(len(predict[i])):
            # pad 0.0 \开头 0.1 \结尾0.2 \非实体 0.3 \实体 0.4 \实体尾巴 0.5
            if predict[i][j] == real[i][j]:
                temp = real[i][j]
                matrix[temp][temp] = matrix[temp][temp] + 1
            else:
                temp = real[i][j]
                temp_wrong = predict[i][j]
                matrix[temp][temp_wrong] = matrix[temp][temp_wrong] + 1

    # print matrix
    print("each label's tp and fp in this matrix:")
    print(matrix)

    # calculate the precision of every kind
    # name = ['pad', 'start', 'end', 'non-entity', 'entity', 'entity-end']
    # all precision and recall, saved by list
    r_total = []
    p_total = []

    for i in range(len(matrix)):

        # two calculating param of micro F1
        micro_tp = 0
        micro_fp = 0

        for j in range(len(matrix[i])):
            # calculate precision
            micro_tp = micro_tp + matrix[j][i]
            # calculate recall
            micro_fp = micro_fp + matrix[i][j]

        # RECALL
        if micro_tp == 0:
            P = 1
        else:
            P = matrix[i][i]/micro_tp

        # precision
        if micro_fp == 0:
            R = 1
        else:
            R = matrix[i][i]/micro_fp

        r_total.append(R)
        p_total.append(P)

    # print each label's recall and precision
    print("each label's recall:")
    print(r_total)
    print("each label's precision:")
    print(p_total)

    mean_r = np.mean(r_total)
    mean_p = np.mean(p_total)

    print("Recall:" + str(mean_r))
    print("Precision:" + str(mean_p))

    F1 = 2 / ((1 / mean_r) + (1 / mean_p))
    print("F1-Score:" + str(F1))

    return mean_r, mean_p, r_total, p_total, matrix

def element_to_integer(data_need):

    data = np.zeros(shape=np.shape(data_need))

    # every element multiply 10 and change to integer
    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         for k in range(len(data[i][j])):
    #             data[i][j][k] = int(data[i][j][k]*10)

    for i in range(len(data)):
        for j in range(len(data[i])):
                data[i][j] = int(data_need[i][j]*10)

    return data




if __name__ == '__main__':
    print(np.shape(test))
    print(np.shape(softmax_to_label(test)))
    print(f1score(softmax_to_label(test),test1))

    # load train data
    x_train = np.load("Old_data_set/model3_train_v.npy", allow_pickle=True)
    y_train = np.load("Old_data_set/model2_train_pos012.npy", allow_pickle=True)

    # load test data
    x_test = np.load("Old_data_set/model3_test_v.npy", allow_pickle=True)
    y_test = np.load("Old_data_set/model2_test_pos012.npy", allow_pickle=True)

    #
    print(np.shape(y_train))
    print(y_train)
    y_train = element_to_integer(y_train)
    y_train = y_train.astype(int)
    print("*****************************")
    print(y_train)
    print("-----------------------------")
    print(np.shape(y_test))
    print(y_test)
    y_test = element_to_integer(y_test)
    y_test = y_test.astype(int)
    print(y_test)

    np.save('model3_train_pos012.npy', y_train)
    np.save('model3_test_pos012.npy', y_test)