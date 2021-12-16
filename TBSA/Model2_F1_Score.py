import numpy as np
# np.set_printoptions(threshold=np.inf)

test_predict = [[0.1,0.1,0.22,0.35,0.14,0.2,0.2,0.1,0.4,0.5,0.0],
                [0.1,0.1,0.2,0.3,0.1,0.2,0.2,0.1,0.4,0.5,0.0]]

test_real    = [[0.1,0.1,0.3,0.2,0.3,0.2,0.2,0.1,0.4,0.5,0.0],
                [0.1,0.1,0.3,0.2,0.3,0.2,0.2,0.1,0.4,0.5,0.0]]


# because the result of model2 is not normalized
def transpose(predict):
    # transpose the predict data
    save_predict = []

    for i in range(len(predict)):
        # save
        temp = predict[i].T
        temp = np.array(temp)
        temp = temp.reshape(1, -1)
        save_predict.append(temp[0])

    return save_predict

# # F1 sore
# # actual matrix of predict and real are two dimensions
# def f1score(predict, real):
#
#     # Confusion matrix
#     matrix = np.zeros([6,6])
#
#     # print matrix
#     # print("initialize the f1-calculating matrix:")
#     # print(matrix)
#
#     for i in range(len(predict)):
#         # length_j = len(predict[i])
#         for j in range(len(predict[i])):
#             # pad 0.0 \开头 0.1 \结尾0.2 \非实体 0.3 \实体 0.4 \实体尾巴 0.5
#             if predict[i][j] == real[i][j]:
#                 temp = int(real[i][j])
#                 matrix[temp][temp] = matrix[temp][temp] + 1
#             else:
#                 temp = int(real[i][j])
#                 temp_wrong = int(predict[i][j])
#                 matrix[temp][temp_wrong] = matrix[temp][temp_wrong] + 1
#
#     # print matrix
#     print("each label's tp and fp in this matrix:")
#     print(matrix)
#
#     # calculate the precision of every kind
#     # name = ['pad', 'start', 'end', 'non-entity', 'entity', 'entity-end']
#     # all precision and recall, saved by list
#     r_total = []
#     p_total = []
#
#     for i in range(len(matrix)):
#
#         # two calculating param of micro F1
#         micro_tp = 0
#         micro_fp = 0
#
#         for j in range(len(matrix[i])):
#             # calculate recall
#             micro_tp = micro_tp + matrix[j][i]
#             # calculate precision
#             micro_fp = micro_fp + matrix[i][j]
#
#         # RECALL
#         if micro_tp == 0:
#             P = 1
#         else:
#             P = matrix[i][i]/micro_tp
#
#         # precision
#         if micro_fp == 0:
#             R = 1
#         else:
#             R = matrix[i][i]/micro_fp
#
#         r_total.append(R)
#         p_total.append(P)
#
#     # print each label's recall and precision
#     print("each label's recall:")
#     print(r_total)
#     print("each label's precision:")
#     print(p_total)
#
#     mean_r = np.mean(r_total)
#     mean_p = np.mean(p_total)
#
#     print("Recall:" + str(mean_r))
#     print("Precision:" + str(mean_p))
#
#     F1 = 2 / ((1 / mean_r) + (1 / mean_p))
#     print("F1-Score:" + str(F1))
#
#     return mean_r, mean_p, r_total, p_total


# F1 sore
# actual matrix of predict and real are two dimensions
def f1score(predict, real):

    # Confusion matrix
    matrix = np.zeros([6,6])

    tag_amount = 0
    predict_amount = 0
    correct_amount = 0

    # print matrix
    # print("initialize the f1-calculating matrix:")
    # print(matrix)
    for i in range(len(predict)):
        # length_j = len(predict[i])
        for j in range(len(predict[i])):
            if real[i][j] == 4:
                tag_amount = tag_amount + 1

    for i in range(len(predict)):
        # length_j = len(predict[i])
        for j in range(len(predict[i])):
            if predict[i][j] == 4:
                predict_amount = predict_amount + 1


    flag = 0

    for i in range(len(predict)):
        # length_j = len(predict[i])
        for j in range(len(predict[i])):
            # pad 0.0 \开头 0.1 \结尾0.2 \非实体 0.3 \实体 0.4 \实体尾巴 0.5
            if real[i][j] == 4 and predict[i][j] == 4:
                flag = 1
            if flag == 1:
                if real[i][j] == predict[i][j]:
                    flag = 1
                else:
                    flag = 0
                    continue
                if real[i][j] == predict[i][j] and real[i][j] != 4 and real[i][j] != 5:
                    correct_amount = correct_amount + 1
                    flag = 0

    print(tag_amount)
    print(predict_amount)
    print(correct_amount)

    recall =0
    precision = 0

    if tag_amount != 0:
        recall = correct_amount / tag_amount
    if predict_amount != 0:
        precision = correct_amount / predict_amount


    if  recall != 0 and precision !=0:
        f1 = 2*recall*precision/(precision+recall)


    # # print matrix
    # print("each label's tp and fp in this matrix:")
    # print(matrix)
    #
    # # calculate the precision of every kind
    # # name = ['pad', 'start', 'end', 'non-entity', 'entity', 'entity-end']
    # # all precision and recall, saved by list
    # r_total = []
    # p_total = []
    #
    # for i in range(len(matrix)):
    #
    #     # two calculating param of micro F1
    #     micro_tp = 0
    #     micro_fp = 0
    #
    #     for j in range(len(matrix[i])):
    #         # calculate recall
    #         micro_tp = micro_tp + matrix[j][i]
    #         # calculate precision
    #         micro_fp = micro_fp + matrix[i][j]
    #
    #     # RECALL
    #     if micro_tp == 0:
    #         P = 1
    #     else:
    #         P = matrix[i][i]/micro_tp
    #
    #     # precision
    #     if micro_fp == 0:
    #         R = 1
    #     else:
    #         R = matrix[i][i]/micro_fp
    #
    #     r_total.append(R)
    #     p_total.append(P)
    #
    # # print each label's recall and precision
    # print("each label's recall:")
    # print(r_total)
    # print("each label's precision:")
    # print(p_total)
    #
    # mean_r = np.mean(r_total)
    # mean_p = np.mean(p_total)
    #
    # print("Recall:" + str(mean_r))
    # print("Precision:" + str(mean_p))
    #
    # F1 = 2 / ((1 / mean_r) + (1 / mean_p))
    # print("F1-Score:" + str(F1))
    #
    # return mean_r, mean_p, r_total, p_total

    return recall, precision


# # F1 sore
# # actual matrix of predict and real are two dimensions
# def f1score(predict, real):
#
#     # Confusion matrix
#     matrix = np.zeros([2, 2])
#
#     # print matrix
#     # print("initialize the f1-calculating matrix:")
#     # print(matrix)
#
#     print(predict[0])
#
#     for i in range(len(predict)):
#         for j in range(len(predict[i])):
#             if predict[i][j] == 4 or predict[i][j] == 5:
#                 predict[i][j] = 1
#             else:
#                 predict[i][j] = 0
#
#     print(predict[0])
#
#     print(real[0])
#
#     for i in range(len(real)):
#         for j in range(len(real[i])):
#             if real[i][j] == 4 or real[i][j] == 5:
#                 real[i][j] = 1
#             else:
#                 real[i][j] = 0
#
#     print(real[0])
#
#
#
#     for i in range(len(predict)):
#         for j in range(len(predict[i])):
#             pos1 = int(real[i][j])
#             pos2 = int(predict[i][j])
#             matrix[pos1][pos2] = matrix[pos1][pos2] + 1
#
#     print(matrix)
#
#     precision_entity = matrix[1][1] / (matrix[0][1] + matrix[1][1])
#     precision_non = matrix[0][0] / (matrix[0][0] + matrix[1][0])
#     recall_entity = matrix[1][1] / (matrix[1][0]+matrix[1][1])
#     recall_non = matrix[0][0] / (matrix[0][0]+matrix[0][1])
#
#     recall = (recall_non + recall_entity)/2
#     precision = (precision_non + precision_entity)/2
#
#     f1 = 2/((1/recall)+(1/precision))
#
#
#     return recall, precision, f1


# make the predict's label into int
def change(predict):
    # num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    num = [0, 1, 2, 3, 4, 5]
    for i in range(len(predict)):
        for j in range(len(predict[i])):

            # pad 0.0 \开头 0.1 \结尾0.2 \非实体 0.3 \实体 0.4 \实体尾巴 0.5
            if predict[i][j]<0:
                label = 0
            else:
                label = 0
                temp = 9999

                for count in range(0, 6):
                    if abs((num[count] - predict[i][j])) < temp:
                        temp = abs(num[count] - predict[i][j])
                        label = count

            predict[i][j] = label

    return predict




if __name__ == '__main__':
    test_predict = np.load("Saved_Model_data/Result_model2.npy", allow_pickle=True)
    test_real = np.load("Model2/model2_test_pos012.npy", allow_pickle=True)

    test_v = np.load("Model2/model2_test_v.npy", allow_pickle=True)
    print("test_v:\n"+str(test_v))


    print("test_predict:\n" + str(test_predict))
    print("test_real:\n"+str(test_real))

    test_predict = change(test_predict)
    print("test_predict:\n"+str(test_predict))
    print(f1score(test_predict,test_real))
















