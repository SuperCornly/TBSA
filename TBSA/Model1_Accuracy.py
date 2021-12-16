import numpy
import numpy as np


def accuracy(predict, real_result):

    correct = 0
    wrong = 0

    # predict is two dimensions
    for i in range(len(predict)):
        max = -999
        position = -1
        for j in range(len(predict[i])):
            if predict[i][j] > max:
                max = predict[i][j]
                position = j

        if real_result[i][position] == 1:
            correct = correct + 1
        else:
            wrong = wrong + 1

    accuracy = correct/(correct + wrong)

    return accuracy, (correct + wrong)
