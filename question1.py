import numpy as np

from question1Neuron import neuron

class NeuralNetwork():
    
    def __init__(self):

        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def learn(self, x):

        return 1 / (1 + np.exp(-x+2))

    def sigmoid_derivative(self, x):

        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        
        inputs = inputs.astype(float)
        output = self.learn(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 2 input values and 1 output value
    training_inputs = np.array([[0,0],
                                [1,0],
                                [0,1],
                                [1,1]])

    training_outputs = np.array([[0,0,0,1]]).T

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 1000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = int(input("Input current of first neuron ('30' represents 0 and '300' represents 1) -> "))
    B = int(input("Input current of first neuron ('30' represents 0 and '300' represents 1) -> "))


    neuron1= neuron(A)
    firingRate1 = (neuron1.firingRate())

    neuron2 = neuron(B)
    firingRate2 = (neuron2.firingRate())

    neuron1.showPlot('Neuron 1')
    neuron2.showPlot('Neuron 2')
    
    if( ((neural_network.synaptic_weights[0] * firingRate1) + (neural_network.synaptic_weights[1] *  firingRate2)) > 8 ):

        print("Output neuron 1 (which represents the number 1) has higher number of spikes because it has a higher firing rate (" 
        + str(firingRate1) + "), which means the output is 1/True" )

    else:

        print("Output neuron 0 (which represents the number 0) has higher number of spikes because it has a higher firing rate (" 
        + str(firingRate2) + "), which means the output is 0/False" )





