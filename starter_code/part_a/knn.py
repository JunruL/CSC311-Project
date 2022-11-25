from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # TODO: double check if we can change the output format
    print("Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_list = [1, 6, 11, 16, 21, 26]

    valid_acc_by_user = []
    max_acc_user = 0
    k_user = 0
    print("kNN imputed by user:")
    for k in k_list:
        print(f"When k is {k}, Validation ", end='')
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_acc_by_user.append(accuracy)
        if accuracy > max_acc_user:
            k_user = k
            max_acc_user = accuracy
    print(f"k* = {k_user} has the highest performance on validation data, with Test ", end='')
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_user)
    print("\n")

    valid_acc_by_item = []
    max_acc_item = 0
    k_item = 0
    print("kNN imputed by item:")
    for k in k_list:
        print(f"When k is {k}, Validation ", end='')
        accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc_by_item.append(accuracy)
        if accuracy > max_acc_item:
            k_item = k
            max_acc_item = accuracy
    print(f"k* = {k_item} has the highest performance on validation data, with Test ", end='')
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_item)

    plt.figure(1)
    plt.plot(k_list, valid_acc_by_user)
    plt.xlabel("k")
    plt.ylabel("validation accuracy")
    plt.title("Validation Accuracy of kNN Imputed by User")
    plt.show()

    plt.figure(2)
    plt.plot(k_list, valid_acc_by_item)
    plt.xlabel("k")
    plt.ylabel("validation accuracy")
    plt.title("Validation Accuracy of kNN Imputed by Item")
    plt.show()


if __name__ == "__main__":
    main()
