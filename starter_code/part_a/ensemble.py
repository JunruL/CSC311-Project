"""Implement bagging ensemble to improve the stability and accuracy of the base models."""

from neural_network import *


def load_and_bootstrap(base_path="../data", m=3):
    """
    Load the data in PyTorch Tensor.
    Bootstrap <m> new datasets from the loaded training data with the same length.

    :param base_path: the base path to load the data
    :param m: number of bootstrapped datasets

    :return: (bootstrapped_zero_matrix_lst, bootstrapped_train_matrix_lst, valid_data, test_data)
        WHERE:
        bootstrapped_zero_matrix_lst: a list of 2D sparse matrix where missing entries are
                                      filled with 0.
        bootstrapped_train_matrix_lst: a list of 2D sparse matrix
        valid_data: A dictionary {user_id: list, user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list, user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    bootstrapped_train_matrix_lst = []
    bootstrapped_zero_matrix_lst = []
    n = len(train_matrix)

    for _ in range(m):
        # randomly sample n indices in range(0, n)
        ids = torch.randint(0, n, (n,))
        # get the bootstrapped_train_matrix using sampled indices
        bootstrapped_train_matrix = train_matrix[ids]

        bootstrapped_zero_matrix = bootstrapped_train_matrix.copy()
        # Fill in the missing entries to 0.
        bootstrapped_zero_matrix[np.isnan(bootstrapped_train_matrix)] = 0
        # Change to Float Tensor for PyTorch.
        bootstrapped_zero_matrix = torch.FloatTensor(bootstrapped_zero_matrix)
        bootstrapped_train_matrix = torch.FloatTensor(bootstrapped_train_matrix)

        bootstrapped_train_matrix_lst.append(bootstrapped_train_matrix)
        bootstrapped_zero_matrix_lst.append(bootstrapped_zero_matrix)

    return bootstrapped_zero_matrix_lst, bootstrapped_train_matrix_lst, valid_data, test_data


def evaluate_ensemble(model_lst, train_data, test_data):
    """
    Evaluate the models in <model_lst> using the ensemble method, by taking
    the mean probability of correctness given by the models in <model_lst>.

    Note: Code adapted from neural_network.evaluate

    :param model_lst: a list of models
    :param train_data: the original training data that was used for bootstrapping
    :param test_data: the test dataset

    :return: the test accuracy
    """
    # Tell PyTorch you are evaluating the model.
    for model in model_lst:
        model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output_lst = [model(inputs) for model in model_lst]

        guess_lst = [output[0][test_data["question_id"][i]].item() for output in output_lst]

        guess = (sum(guess_lst) / len(guess_lst)) >= 0.5
        if guess == test_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    # set the random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # load the data
    zero_train_matrix_lst, train_matrix_lst, valid_data, test_data = load_and_bootstrap()
    zero_train_matrix, _, _, _ = load_data()

    # set up the model and hyper-parameters
    num_question = train_matrix_lst[0].shape[1]
    k = 50
    lr = 0.008  # regularization penalty
    lamb = 0.01
    num_epoch = 56

    # train the models
    models = [AutoEncoder(num_question, k) for _ in range(3)]
    for i in range(3):
        train(models[i], lr, lamb, train_matrix_lst[i], zero_train_matrix_lst[i], valid_data,
              num_epoch)

    # evaluate on the test set using the ensemble method
    test_acc = evaluate_ensemble(models, zero_train_matrix, test_data)
    print("test_acc: ", test_acc)


if __name__ == "__main__":
    main()
