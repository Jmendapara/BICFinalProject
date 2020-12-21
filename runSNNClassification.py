import numpy as np
from neuron import neuron
import random
from convolve import rf
from variables import param as par
import imageio
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt; plt.rcdefaults()
from numpy import interp
import math
import pandas as pd
import cv2

def learned_weights_synapse(id):
    ans = []
    with open('weights_training.txt', 'r') as weight_file:
        lines = weight_file.readlines()
        if (len(lines) <= id):
            return ans
        for i in lines[id].split('\t'):
            ans.append(float(i))

    return ans
    
def encode(pot):

    #initializing spike train
    train = []

    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):

            temp = np.zeros([(par.T+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [-1.069,2.781], [1,20])
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pot[l][m]>0):
                while k<(par.T+1):
                    temp[int(k)] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)

    return train

def dataframe_to_array(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1.iloc[:, 1:].to_numpy()
    targets_array = dataframe1['label'].to_numpy()
    return inputs_array, targets_array
    

if __name__ == "__main__":


    #Parameters
    global time, T, dt, t_back, t_fore, w_min
    time  = np.arange(1, par.T+1, 1)

    layer2 = []

    # creating the hidden layer of neurons
    for i in range(par.n):
        a = neuron()
        layer2.append(a)

    #synapse matrix
    synapse = np.zeros((par.n,par.m))

    digits = load_digits()

    test = pd.read_csv('sign_mnist_test.csv')
    test_set = np.array(test, dtype = 'float32')

    #inputs_array, targets_array = dataframe_to_array(train)
    testinputs_array, testtargets_array = dataframe_to_array(test)

    alpha = 2 # Contrast control (1.0-3.0)
    beta = -99 # Brightness control (0-100)

    xValues = ('Neuron 0', 'Neuron 1','Neuron 2', 'Neuron 3', 'Neuron 4', 'Neuron 5', 'Neuron 6', 'Neuron 7', 'Neuron 8', 'Neuron 9')

    #random initialization for rest of the synapses
    for i in range(par.n):
        synapse[i] = learned_weights_synapse(i)


    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

    correct = 0

    total = 0

    print(len(testinputs_array))
    print(testinputs_array.size)

    totalLetters = 0
    letterQ = 0
    letterC = 0

    for i in range(len(testtargets_array)):
         if(testtargets_array[i] == 2 or testtargets_array[i] == 16):

            totalLetters += 1

            if(testtargets_array[i] == 2):
                letterC += 1

            else:
                letterQ += 1

    print(letterC)
    print(letterQ)
    print(totalLetters)

    for k in range(1):

        #for e in range(par.n):
        
            for i in range(len(testinputs_array)):

                if(testtargets_array[i] == 2 or testtargets_array[i] == 16):

                    final_spike_count = [0,0,0,0,0,0,0,0,0,0]


                    #img = imageio.imread('training/' + str(e) + "/" + str(i) + ".jpg")
                    test = testinputs_array[i].reshape((28,28))

                    img = cv2.convertScaleAbs(test, alpha=alpha, beta=beta)

                    spike_count = [0,0,0,0,0,0,0,0,0,0]

                    #read the image to be classified

                    #initialize the potentials of output neurons
                    for x in layer2:
                        x.initial(par.Pth)

                    #calculate teh membrane potentials of input neurons
                    pot = rf(img)

                    #generate spike trains
                    train = np.array(encode(pot))

                    #flag for lateral inhibition
                    f_spike = 0

                    active_pot = [0,0,0,0,0,0,0,0,0,0]

                    for t in time:
                        for j, x in enumerate(layer2):
                            active = []

                    #update potential if not in refractory period
                            if(x.t_rest<t):
                                x.P = x.P + np.dot(synapse[j], train[:,t])
                                if(x.P>par.Prest):
                                    x.P -= par.D
                                active_pot[j] = x.P

                        # Lateral Inhibition
                        if(f_spike==0):
                            high_pot = max(active_pot)
                            if(high_pot>par.Pth):
                                f_spike = 1
                                winner = np.argmax(active_pot)
                                for s in range(par.n):
                                    if(s!=winner):
                                        layer2[s].P = par.Prest

                        #Check for spikes
                        for j,x in enumerate(layer2):
                            s = x.check()
                            if(s==1):
                                spike_count[j] += 1
                                x.t_rest = t + x.t_ref
                    #print(spike_count)
                    final_spike_count = np.add(final_spike_count, spike_count)
                    maxIndex = np.argmax(spike_count)
                    #print("e =" + str(e) + "and index =" + str(maxIndex))


                    y_pos = np.arange(len(xValues))
                    plt.bar(y_pos, final_spike_count, align='center', alpha=0.5)
                    plt.xticks(y_pos, xValues)
                    plt.ylabel('Total Spikes')
                    plt.title('Total Spikes for Neuron')
                    #plt.show()

                    total += 1

                    print("Testing Letter -> " + str(class_names[testtargets_array[i]]))

                    if(spike_count[0] < spike_count[1]):
                        print("Output Letter -> C")
                        if(testtargets_array[i] == 2):
                            correct+=1

                    

                    if(spike_count[0] >= spike_count[1]):
                        print("Output Letter -> Q")
                        if(testtargets_array[i] == 16):
                            correct+=1

                    print("--------------------------------------------")
                            

                    
                    

    print("Total number of test images: " + str(total))
    print("Total correct predictions: " + str(correct))


            
