# This is the main file you can run to train the algorithm. 
# Results of what the neurons have learned can be found in the neuronX.png file after running this file.

# By: Shreya, Tilak, Jay, and Raina 

import numpy as np
import os
import time as timing 
from sklearn.datasets import load_digits
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from convolve import rf
import imageio
from neuron import neuron
import random
from variables import param as par
from numpy import interp
import math
import pandas as pd
import cv2



def reconst_weights(weights, num):
	weights = np.array(weights)
	weights = np.reshape(weights, (par.pixel_x,par.pixel_x))
	img = np.random.randint(0, 255, (par.pixel_x, par.pixel_x, 3)).astype(np.uint8)
	for i in range(par.pixel_x):
		for j in range(par.pixel_x):
			img[i][j] = int(interp(weights[i][j], [par.w_min,par.w_max], [0,255]))

	imageio.imwrite('neuron' + str(num) + '.png', img)
	return img

#STDP reinforcement learning curve
def rl(t):

	if t>0:
		return -par.A_plus*np.exp(-float(t)/par.tau_plus)
	if t<=0:
		return par.A_minus*np.exp(float(t)/par.tau_minus)


#STDP weight update rule
def update(w, del_w):
	if del_w<0:
		return w + par.sigma*del_w*(w-abs(par.w_min))*par.scale
	elif del_w>0:
		return w + par.sigma*del_w*(par.w_max-w)*par.scale


def threshold(train):

    tu = np.shape(train[0])[0]
    thresh = 0
    for i in range(tu):
        simul_active = sum(train[:,i])
        if simul_active>thresh:
            thresh = simul_active

    return (thresh/3)*par.scale

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

    #potentials of output neurons
    pot_arrays = []
    for i in range(par.n):
        pot_arrays.append([])

    #time series
    time  = np.arange(1, par.T+1, 1)

    layer2 = []

    # creating the hidden layer of neurons
    for i in range(par.n):
        a = neuron()
        layer2.append(a)

    #synapse matrix initialization
    synapse = np.zeros((par.n,par.m))

    for i in range(par.n):
        for j in range(par.m):
            synapse[i][j] = random.uniform(0,par.w_max*0.5)

    spike_probe = []
    for i in range(par.n):
        spike_probe.append([(0, 0)])

    digits = load_digits()
    train = pd.read_csv('sign_mnist_train.csv')
    #train_set = np.array(train, dtype = 'float32')

    inputs_array, targets_array = dataframe_to_array(train)
    #testinputs_array, testtargets_array = dataframe_to_array(testdataset)

    alpha = 2 # Contrast control (1.0-3.0)
    beta = -99 # Brightness control (0-100)



    for k in range(par.epoch):

        #Traverse through the files 
        for i in range(300):

                if(targets_array[i] == 2 or targets_array[i] == 16):

                        print(k)
                        

                        #print("epoch: "+str(k))
                
                        #For each digit we are trying to train
                        #for e in range(par.n):

                        #print("Looking at digit "+str(e))
                        #img = imageio.imread('training/' + str(e) + "/" + str(i) + ".jpg")

                        test = inputs_array[i].reshape((28,28))

                        img = cv2.convertScaleAbs(test, alpha=alpha, beta=beta)


                        #plt.imshow(img)
                        #plt.show()

                        pot = rf(img)
                        #Generating spike train
                        train = np.array(encode(pot))

                        if k == 0:
                            print(train.shape)

                        var_threshold = threshold(train)

                        var_D = par.D

                        for x in layer2:
                            x.initial(par.Pth)

                        #flag for lateral inhibition
                        f_spike = 0

                        img_win = 100

                        active_pot = []
                        for index1 in range(par.n):
                            active_pot.append(0)

                        #Leaky integrate and fire neuron dynamics
                        for t in time:
                            for j, x in enumerate(layer2):
                                active = []
                                if(x.t_rest<t):
                                    x.P = x.P + np.dot(synapse[j], train[:,t])
                                    if(x.P>par.Prest):
                                        x.P -= var_D
                                    active_pot[j] = x.P

                                pot_arrays[j].append(x.P)

                            # Lateral Inhibition
                            if(f_spike==0):
                                high_pot = max(active_pot)
                                if(high_pot>par.Pth):
                                    f_spike = 1
                                    winner = np.argmax(active_pot)
                                    img_win = winner
                                    for s in range(par.n):
                                        if(s!=winner):
                                            layer2[s].P = -500

                            #Check for spikes and update weights
                            for j,x in enumerate(layer2):
                                s = x.check()
                                if(s==1):
                                    spike_probe[j].append((len(pot_arrays[j]), 1))
                                    x.t_rest = t + x.t_ref
                                    x.P = par.Prest
                                    for h in range(par.m):
                                        for t1 in range(-2,par.t_back-1, -1):
                                            if 0<=t+t1<par.T+1:
                                                if train[h][t+t1] == 1:
                                                    synapse[j][h] = update(synapse[j][h], rl(t1))




                                        for t1 in range(2,par.t_fore+1, 1):
                                            if 0<=t+t1<par.T+1:
                                                if train[h][t+t1] == 1:
                                                    synapse[j][h] = update(synapse[j][h], rl(t1))

                        if(img_win!=100):
                            for p in range(par.m):
                                if sum(train[p])==0:
                                    synapse[img_win][p] -= 0.06*par.scale
                                    if(synapse[img_win][p]<par.w_min):
                                        synapse[img_win][p] = par.w_min

        plt.figure(2)
        for i in range(par.n):
            plt.subplot(par.n/2, par.n/2, 1)
            plt.gca().grid(False)
            weights = np.array(synapse[i])
            weights = np.reshape(weights, (par.pixel_x,par.pixel_x))
            img = np.zeros((par.pixel_x,par.pixel_x))
            for i in range(par.pixel_x):
                for j in range(par.pixel_x):
                    img[i][j] = np.interp(weights[i][j], [par.w_min,par.w_max], [-1.0,1.0])
            plt.imshow(img)
            plt.colorbar()
        plt.show()

  



        
    ttt = np.arange(0,len(pot_arrays[0]),1)
    Pth = []
    for i in range(len(ttt)):
        Pth.append(layer2[0].Pth)

    #plotting
    plt.figure(0)
    for i in range(par.n):
        plt.subplot(par.n, 1, i+1)
        axes = plt.gca()
        axes.set_ylim([-20,60])
        plt.plot(ttt,Pth, 'r')
        plt.plot(ttt,pot_arrays[i])

    plt.figure(1)

    for i in range(par.n):
        plt.subplot(par.n, 1, i+1)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        vals = np.array(spike_probe[i])
        plt.stem(vals[:,0],vals[:,1])
    plt.show()

    plt.figure(2)
    for i in range(par.n):
        plt.subplot(par.n/2, par.n/2, 1)
        plt.gca().grid(False)
        weights = np.array(synapse[i])
        weights = np.reshape(weights, (par.pixel_x,par.pixel_x))
        img = np.zeros((par.pixel_x,par.pixel_x))
        for i in range(par.pixel_x):
            for j in range(par.pixel_x):
                img[i][j] = np.interp(weights[i][j], [par.w_min,par.w_max], [-1.0,1.0])
        plt.imshow(img)
        plt.colorbar()
    plt.show()


    #with open('weights_training.txt', 'w') as weight_file:
    #    for i in range(len(synapse)):
    #        weights = []
    #        for j in synapse[i]:
    #            weights.append(str(j))
    #        convert = '\t'.join(weights)
    #        weight_file.write("%s\n" % convert)

    #Reconstructing weights to analyse training
    for i in range(par.n):
        reconst_weights(synapse[i],i)


