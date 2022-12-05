"""Implement bagging ensemble to improve the stability and accuracy of the base models."""

from neural_network import *
import argparse


def bootstrap(zero_train_matrix, train_matrix, m=3):
    """
    Load the data in PyTorch Tensor.
    Bootstrap <m> new datasets from the loaded training data with the same length.
    """
    bootstrapped_train_matrix_lst = []
    bootstrapped_zero_matrix_lst = []
    n = len(train_matrix)

    for _ in range(m):
        # randomly sample n indices in range(0, n)
        ids = np.random.randint(low=0, high=n, size=n)

        # bootstrap train_matrix and zero_train_matrix using sampled indices
        bootstrapped_train_matrix_lst.append(train_matrix[ids])
        bootstrapped_zero_matrix_lst.append(zero_train_matrix[ids])

    return bootstrapped_zero_matrix_lst, bootstrapped_train_matrix_lst


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
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # bootstrap
    zero_train_matrix_lst, train_matrix_lst = bootstrap(zero_train_matrix, train_matrix)

    # set up the model and hyper-parameters
    num_question = train_matrix.shape[1]

    k_lst = [50, 50, 50]
    lr_lst = [0.008, 0.008, 0.008]
    lamb_lst = [0.001, 0.001, 0.001]
    num_epoch_lst = [56, 56, 56]

    # train the models
    model_lst = [AutoEncoder(num_question, k) for k in k_lst]
    for i in range(3):
        print('-------------------------------------')
        print(f'Model[{i}]')
        train(model_lst[i], lr_lst[i], lamb_lst[i], train_matrix_lst[i], 
              zero_train_matrix_lst[i], valid_data, num_epoch_lst[i])

    # evaluate on the test set using the ensemble method
    val_acc = evaluate_ensemble(model_lst, zero_train_matrix, valid_data)
    test_acc = evaluate_ensemble(model_lst, zero_train_matrix, test_data)
    print(f'val_acc={val_acc}, test_acc={test_acc}')
    

if __name__ == "__main__":
    main()
