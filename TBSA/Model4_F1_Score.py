import numpy as np

# def predict_to_labels(predict):
#
#     predict_save = []
#
#     for i in range(len(predict)):
#         max = -999
#         position_predict = -1
#         position_real = -1
#         for j in range(len(predict[i])):
#             if predict[i][j] > max:
#                 max = predict[i][j]
#                 position_predict = j
#
#         temp = np.zeros(shape=np.shape(predict[i]))
#         temp[position] = 1
#
#         predict_save.append(temp)
#
#     return predict_save

def f1_score(predict, real):

    # initialize the f1 matrix
    matrix = np.zeros(shape=(3, 3))

    # save the result
    predict_save = []

    for i in range(len(predict)):
        max = -999
        position_predict = -1
        position_real = -1
        for j in range(len(predict[i])):
            if predict[i][j] > max:
                max = predict[i][j]
                position_predict = j

            if real[i][j] == 1:
                position_real = j

        matrix[position_real][position_predict] = matrix[position_real][position_predict] + 1

        # save the result into predict_save
        temp = np.zeros(shape=(1, 3))
        temp[0][position_predict] = 1
        predict_save.append(temp)

    predict_save = np.array(predict_save)
    predict_save = predict_save.reshape([np.shape(predict_save)[0], np.shape(predict_save)[2]])

    # calculate the precision and recall
    # col
    precision = []
    # row
    recall = []

    for i in range(len(matrix)):

        row = 0
        col = 0

        for j in range(len(matrix[i])):

            row = matrix[i][j] + row
            col = matrix[j][i] + col

        p_temp = matrix[i][i]/col
        r_temp = matrix[i][i]/row

        precision.append(p_temp)
        recall.append(r_temp)

    precision = np.array(precision)
    recall = np.array(recall)

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)

    f1 = 2/((1/mean_precision)+(1/mean_recall))

    print("\n")
    print("predict's shape:\n", np.shape(predict))
    print("predict's f1 matrix:\n", matrix)
    print("predict's recall:\n", recall)
    print("predict's precision:\n", precision)
    print("predict's f1:\n", f1)

    return predict_save, recall, precision, f1



