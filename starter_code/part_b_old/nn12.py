from utils import *
from torch.autograd import Variable
import argparse
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocess import get_student_vectors


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_student):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        self.g1 = nn.Linear(num_student, 256)
        self.g2 = nn.Linear(256, 128)
        self.g3 = nn.Linear(128, 32)

        self.s1 = nn.Linear(num_student, 256)
        self.s2 = nn.Linear(256, 128)
        self.s3 = nn.Linear(128, 32)

        self.t1 = nn.Linear(num_student, 256)
        self.t2 = nn.Linear(256, 128)
        self.t3 = nn.Linear(128, 32)

        self.h3 = nn.Linear(32+32+32, 128)
        self.h2 = nn.Linear(128, 256)
        self.h1 = nn.Linear(256, num_student)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g1.weight, 2) ** 2 +\
                   torch.norm(self.g2.weight, 2) ** 2 +\
                   torch.norm(self.g3.weight, 2) ** 2
        s_w_norm = torch.norm(self.s1.weight, 2) ** 2 +\
                   torch.norm(self.s2.weight, 2) ** 2 +\
                   torch.norm(self.s3.weight, 2) ** 2
        t_w_norm = torch.norm(self.t1.weight, 2) ** 2 +\
                   torch.norm(self.t2.weight, 2) ** 2 +\
                   torch.norm(self.t3.weight, 2) ** 2
        h_w_norm = torch.norm(self.h1.weight, 2) ** 2 +\
                   torch.norm(self.h2.weight, 2) ** 2 +\
                   torch.norm(self.h3.weight, 2) ** 2
        return g_w_norm + s_w_norm + h_w_norm

    def forward(self, inputs, gender, age):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        x1 = self.g1(inputs)
        x1 = F.relu(x1)
        x1 = self.g2(x1)
        x1 = self.g3(x1)
        x1 = nn.Sigmoid()(x1)

        x2 = self.s1(gender)
        x2 = F.relu(x2)
        x2 = self.s2(x2)
        x2 = self.s3(x2)
        x2 = nn.Sigmoid()(x2)

        x3 = self.t1(age)
        x3 = F.relu(x3)
        x3 = self.t2(x3)
        x3 = self.t3(x3)
        x3 = nn.Sigmoid()(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        y = self.h3(x)
        y = F.relu(y)
        y = self.h2(y)
        y = self.h1(y)
        out = torch.sigmoid(y)

        return out


def train(model, lr, lamb, gender_vector, age_vector, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device=device)

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    num_question = train_data.shape[1]

    train_lst = []
    valid_lst = []
    best_valid_acc = 0
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for q_id in range(num_question):
            inputs = Variable(zero_train_data[:, q_id]).unsqueeze(0).to(device=device)
            # student_info = gender_vector.reshape(1, num_student).to(device=device)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs, gender_vector, age_vector)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[:, q_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # loss = torch.sum((output - target) ** 2.)
            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * lamb * 0.5
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, gender_vector, age_vector, zero_train_data, valid_data)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        train_lst.append(train_loss)
        valid_lst.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \tBest Valid Acc: {:.6f}".format(epoch + 1, train_loss, valid_acc,
                                                              best_valid_acc))
        sys.stdout.flush()
    return best_valid_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, gender_vector, age_vector, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device=device)
    # Tell PyTorch you are evaluating the model.
    model.eval()

    num_student = train_data.shape[0]

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[:, u]).unsqueeze(0).to(device=device)
        # student_info = gender_vector.reshape(1, num_student).to(device=device)
        output = model(inputs, gender_vector, age_vector)

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    print(f'nn12.py, same as nn10.py, but remove a activation layer')

    # read the beta value from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    args = parser.parse_args()
    lr = args.lr
    print(f'Tuning process for lr={lr}')

    np.random.seed(0)
    torch.manual_seed(0)

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    gender_vector, age_vector = get_student_vectors()
    gender_vector = torch.FloatTensor(gender_vector)
    age_vector = torch.FloatTensor(age_vector)
    num_student = len(age_vector)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    gender_vector = gender_vector.reshape(1, num_student).to(device=device)
    age_vector = age_vector.reshape(1, num_student).to(device=device)

    num_student = train_matrix.shape[0]

    # lr_lst = [1e-5, 1e-4, 1e-3]
    lamb_lst = [0, 0.00001, 0.0001, 0.001]
    num_epoch = 30
    result_lst = []

    for lamb in lamb_lst:
        print('----------------------------------------------------------')
        print(f'lr={lr}, lamb={lamb}, num_epoch={num_epoch}')
        model = AutoEncoder(num_student)
        best_valid_acc = train(model, lr, lamb, gender_vector, age_vector, train_matrix, zero_train_matrix, valid_data, num_epoch)
        test_acc = evaluate(model, gender_vector, age_vector, zero_train_matrix, test_data)
        print(f'lr={lr}, lamb={lamb}, num_epoch={num_epoch}, test_acc={test_acc}')
        print('----------------------------------------------------------')
        result_lst.append(f'lr={lr}, lamb={lamb}, best_valid_acc={best_valid_acc}')
    
    for result in result_lst:
        print(result)
           

if __name__ == "__main__":
    main()
