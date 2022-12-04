from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocess import get_student_meta

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
    def __init__(self, num_question):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.

        # ---------- structure 1 --------------------
        self.g1 = nn.Linear(num_question, 512)
        self.g2 = nn.Linear(512, 128)
        self.g3 = nn.Linear(128, 32)
        self.h3 = nn.Linear(32 + 2, 128)
        self.h2 = nn.Linear(128, 512)
        self.h1 = nn.Linear(512, num_question)
        

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g1.weight, 2) ** 2 +\
                   torch.norm(self.g2.weight, 2) ** 2 +\
                   torch.norm(self.g3.weight, 2) ** 2
        h_w_norm = torch.norm(self.h1.weight, 2) ** 2 +\
                   torch.norm(self.h2.weight, 2) ** 2 +\
                   torch.norm(self.h3.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs, user_info):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # ---------- structure 1 --------------------
        # x = self.g1(inputs)
        # x = F.relu(x)
        # x = self.g2(x)
        # x = F.relu(x)
        # x = self.g3(x)
        # x = torch.sigmoid(x)
        #
        # x = torch.cat((x, user_info), dim=1)
        #
        # x = self.h3(x)
        # x = F.relu(x)
        # x = self.h2(x)
        # x = F.relu(x)
        # x = self.h1(x)
        # out = torch.sigmoid(x)

        x = self.g1(inputs)
        x = F.relu(x)
        x = self.g2(x)
        x = nn.Tanh()(x)
        x = self.g3(x)
        x = nn.Sigmoid()(x)

        z = torch.cat((x, user_info), dim=1)

        y = self.h3(z)
        y = nn.Tanh()(y)
        y = self.h2(y)
        y = F.relu(y)
        y = self.h1(y)
        out = torch.sigmoid(y)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, student_meta, train_data, zero_train_data, valid_data, num_epoch):
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
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_lst = []
    valid_lst = []
    best_valid_acc = 0
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            user_info = torch.tensor([student_meta[user_id][0], student_meta[user_id][1]]).reshape(1, 2)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs, user_info)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # loss = torch.sum((output - target) ** 2.)
            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * lamb * 0.5
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, student_meta, zero_train_data, valid_data)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        train_lst.append(train_loss)
        valid_lst.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {} \tBest Valid Acc: {:.6f}".format(epoch + 1, train_loss, valid_acc,
                                                              best_valid_acc))
    return best_valid_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, student_meta, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        user_info = torch.tensor([student_meta[u][0], student_meta[u][1]]).reshape(1, 2)
        output = model(inputs, user_info)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    np.random.seed(3)
    torch.manual_seed(3)

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    student_meta = get_student_meta()

    num_question = train_matrix.shape[1]

    model = AutoEncoder(num_question)
    lr = 0.00001
    num_epoch = 169

    lamb = 0.01
    train(model, lr, lamb, student_meta, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc = evaluate(model, student_meta, zero_train_matrix, test_data)
    print("test_acc: ", test_acc)


if __name__ == "__main__":
    main()
