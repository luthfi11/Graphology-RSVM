import numpy as np
from numpy import linalg
import math

from svm import SupportVectorMachine
from utils import get_svm_inputs, print_performance

DATA_DIRECTORY = '../data/'
NUMBER_OF_INDEPENDENT_TESTS = 5


class SmoothSupportVectorMachine(SupportVectorMachine):

    def __init__(self, c):
        super(SmoothSupportVectorMachine, self).__init__(c)
        self.alpha = 5  # not sure what to set this value as? Could not find a good specification other than > 0

        # instance variables to be calculated during training:
        self._a = None
        self._d = None
        self._e = None
        self._gamma = None
        self._hessian = None
        self._num_features = None
        self._w = None
        self._z_gradient = None

    def _smoothing_function(self, x):
        return x + 1.0/self.alpha * np.log(1 + np.exp(-self.alpha * x))  # eq. (8) in paper

    def _get_next_w(self, current_w, step, step_direction):
        return current_w + step * step_direction[0:self._num_features]

    def _get_next_gamma(self, current_gamma, step, step_direction):
        return current_gamma + step * step_direction[self._num_features, 0]

    def _objective_function(self, gamma, w):
        """
        Equation 9 in paper, the phi function
        :param gamma: distance from decision boundary
        :type gamma: float
        :param w: weight vector
        :type w: matrix
        :return:
        """
        x = self._e - self._d * (self._a * w - gamma * self._e)
        p = self._smoothing_function(x)
        return 0.5 * (self.c * math.pow(linalg.norm(p), 2) + (np.transpose(w) * w)[0, 0] + math.pow(gamma, 2))

    def _calculate_hessian(self, plus_function_input):
        """
        Updates the hessian instance variable with the new plus_function input.
        Reference: Equation (19) in the paper. Algorithm 3.1
        :type plus_function_input: matrix
        :return:
        """
        h = 0.5 * (self._e + np.sign(plus_function_input))
        t = np.identity(h.shape[0])
        separating_hyperplane = np.transpose((self._d * self._a)) * t
        p = separating_hyperplane * (self._d * self._a)
        q = separating_hyperplane * (self._d * self._e)

        tmp1 = np.identity(self._w.shape[0] + 1)
        tmp2 = self.c * np.vstack((np.hstack((p, -q)), np.hstack((np.transpose(-q), np.mat([linalg.norm(h)])))))
        self._hessian = tmp1 + tmp2

    def _calculate_gradient(self, gamma, w):
        """
        Recalculates the hessian matrix and the z gradient.
        Side effects: modifies the hessian and z_gradient instance variables
        :param gamma: distance from boundary
        :type gamma: float
        :param w: weight vector
        :type w: matrix
        :return:
        """
        plus_function_input = self._e - (((self._d * self._a) * w) - ((self._d * self._e) * gamma))
        self._calculate_hessian(plus_function_input)
        plus_function = (plus_function_input < 0).choose(plus_function_input, 0)  # eq. (7) in the paper

        z_gradient = np.vstack((
            (w - self.c * np.transpose((self._d * self._a)) * plus_function),
            gamma + self.c * np.transpose((self._d * self._e)) * plus_function
        ))

        self._z_gradient = np.transpose(np.mat(z_gradient))

    def _get_next_armijo_step(self, w, gamma, step_direction, step_gap):
        """
        Equation (20) and (21) in the paper
        :param w: weight vector
        :type w: matrix
        :param gamma: distance to decision boundary
        :type gamma: float
        :param step_direction: a m x 1 matrix of displacements
        :type step_direction: matrix
        :type step_gap: matrix
        :return: the next armijo step
        """
        step = 1 #λ
        objective = self._objective_function(gamma, w)
        #step direc = di
        next_w = self._get_next_w(w, step, step_direction)
        next_gamma = self._get_next_gamma(gamma, step, step_direction)
        next_objective = self._objective_function(next_gamma, next_w)
        objective_difference = objective - next_objective

        while objective_difference < -0.05 * step * step_gap:
            step *= 0.5
            next_w = self._get_next_w(w, step, step_direction)
            next_gamma = self._get_next_gamma(gamma, step, step_direction)
            next_objective = self._objective_function(next_gamma, next_w)
            objective_difference = objective - next_objective

        return step

    def train(self, data, class_labels):
        """
        Trains on the data set and class labels, updating the essential instance variables for classification.
        :param data: training data, excluding the index and class label columns
        :type data: ndarray
        :param class_labels: flat array of associated true class labels
        :type class_labels: ndarray
        :return:
        """

        num_examples, self._num_features = data.shape

        self._a = np.mat(data)
        self._e = np.mat(np.ones((num_examples, 1)))
        self._d = np.mat(np.array(np.identity(num_examples)) * class_labels)
        self._w = np.mat(np.zeros((self._num_features, 1)))
        self._gamma = 0.0

        self._calculate_gradient(self._gamma, self._w)

        step_direction = linalg.inv(self._hessian) * -1 * np.transpose(self._z_gradient) #di
        step_gap = np.transpose(step_direction) * np.transpose(self._z_gradient)
        step = self._get_next_armijo_step(self._w, self._gamma, step_direction, step_gap)

        distance_to_solution = step * (self._z_gradient * np.transpose(self._z_gradient))[0, 0]
        convergence = 0.01
        while distance_to_solution >= convergence:
            self._w = self._get_next_w(self._w, step, step_direction)
            self._gamma = self._get_next_gamma(self._gamma, step, step_direction)
            self._calculate_gradient(self._gamma, self._w)
            step_direction = linalg.inv(self._hessian) * -1 * np.transpose(self._z_gradient)
            step_gap = np.transpose(step_direction) * np.transpose(self._z_gradient)
            step = self._get_next_armijo_step(self._w, self._gamma, step_direction, step_gap)
            distance_to_solution = step * (self._z_gradient * np.transpose(self._z_gradient))[0, 0]

    def classify(self, data):
        """
        Will classify a data set, assumes index columns and class label column are not present.
        :param data: data to be classified
        :type data: ndarray
        :return: a flat ndarray of predicted class labels
        """
        return np.sign(np.array([self.predict(example) for example in data]))

    def predict(self, example):
        """
        Predicts the class label of a single example.
        :param example: single example from dataset
        :return:
        """
        return (np.dot(np.transpose(self._w), example) - self._gamma)[0, 0]

    def get_accuracy(self, num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):
        return (num_true_positives + num_true_negatives) / (num_true_positives + num_true_negatives + num_false_positives + num_false_negatives)

    def print_performance(self, results):
        num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = 0.0, 0.0, 0.0, 0.0

        accuracies = []
        for result in results:
            predictions, class_labels = result['predictions'], result['class_labels']
            for prediction, class_label in zip(predictions, class_labels):
                if prediction > 0:
                    if class_label > 0:
                        num_true_positives += 1
                    else:
                        num_false_positives += 1
                else:
                    if class_label <= 0:
                        num_true_negatives += 1
                    else:
                        num_false_negatives += 1

                accuracies.append(get_accuracy(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives))

        print("Accuracy: {:0.3f} {:0.3f}".format(np.mean(accuracies), np.std(accuracies)))

if __name__ == '__main__':
    results = SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
    for result in results:
        num_correct = np.sum(result['predictions'] == result['class_labels'])
        print("{}/{} correct predictions".format(num_correct, len(result['predictions'])))

    print_performance(results)