from matplotlib import pyplot as plt
from utils import *
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    for k in range(len(user_id)):
        theta_k = theta[user_id[k]]
        beta_k = beta[question_id[k]]
        c_k = is_correct[k]
        log_lklihood += c_k * (theta_k - beta_k) - np.logaddexp(0, theta_k - beta_k)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    for k in range(len(user_id)):
        theta_k = theta[user_id[k]]
        beta_k = beta[question_id[k]]
        c_k = is_correct[k]
        theta[user_id[k]] += lr * (c_k - sigmoid(theta_k - beta_k))
        beta[question_id[k]] += lr * (sigmoid(theta_k - beta_k) - c_k)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(data["user_id"]) + 1)
    beta = np.zeros(len(data["question_id"]) + 1)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_lld_lst.append(-neg_lld_train)
        val_lld_lst.append(-neg_lld_val)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def get_age():
    return


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # part (b)
    theta1, beta1, val_acc_lst1, train_lld_lst1, val_lld_lst1 = irt(train_data, val_data, 0.01, 50)
    theta2, beta2, val_acc_lst2, train_lld_lst2, val_lld_lst2 = irt(train_data, val_data, 0.02, 25)
    theta3, beta3, val_acc_lst3, train_lld_lst3, val_lld_lst3 = irt(train_data, val_data, 0.05, 10)
    theta4, beta4, val_acc_lst4, train_lld_lst4, val_lld_lst4 = irt(train_data, val_data, 0.001, 80)

    print("validation rate accuracy 1:", val_acc_lst1)
    print("validation rate accuracy 2:", val_acc_lst2)
    print("validation rate accuracy 3:", val_acc_lst3)
    print("validation rate accuracy 4:", val_acc_lst4)

    plt.figure()
    plt.plot(train_lld_lst4, label="training")
    plt.plot(val_lld_lst4, label="validation")
    plt.xlabel("number of iterations")
    plt.ylabel("log-likelihood")
    plt.title("Training and Validation Log-likelihoods as a Function of Iterations")
    plt.legend()
    plt.show()

    # part (c)
    val_acc = evaluate(val_data, theta4, beta4)
    print("Validation Accuracy:", val_acc)
    test_acc = evaluate(test_data, theta4, beta4)
    print("Test Accuracy:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1 = sigmoid(np.sort(theta4) - beta4[0])
    j2 = sigmoid(np.sort(theta4) - beta4[1])
    j3 = sigmoid(np.sort(theta4) - beta4[2])

    plt.figure()
    plt.plot(np.sort(theta4), j1, label="Question 1")
    plt.plot(np.sort(theta4), j2, label="Question 2")
    plt.plot(np.sort(theta4), j3, label="Question 3")
    plt.title("Probability of Correct Response as a function of Theta given a Question j")
    plt.ylabel("probability of the correct response")
    plt.xlabel("theta given a question j")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
