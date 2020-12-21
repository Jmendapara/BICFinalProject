from numpy import *
from pylab import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


class neuron:

    def __init__(self, input_current):
        self.T = 100
        self.dt = dt = 0.1 # simulation time step (msec)
        self.time = arange(0, self.T+ self.dt, self.dt) # time array
        self.t_rest = 0
        self.Vm = zeros(len(self.time))
        self.Rm = 1 # resistance in kOhm
        self.Cm = 10 # capacitance in uF
        self.spikeDelta = 0.5 #V
        self.I = input_current # input current in A
        self.Vth = 1 # spike threshold in V

    def firingRate(self):

        v_th = 8
        return 1.0 / (0 + (self.Rm*self.Cm) * log(self.I/(self.I - v_th)))

    def showPlot(self, name):
        ## simulate neuron
        for i, t in enumerate(self.time):
            if t > self.t_rest:
                self.Vm[i] = self.Vm[i-1] + (-self.Vm[i-1] + self.I*self.Rm) / (self.Rm*self.Cm) * self.dt
                if self.Vth <= self.Vm[i]:
                    self.Vm[i] += self.spikeDelta
                    self.t_rest = t + 6 ## refractory period
    
        figure(1)
        plot(self.time, self.Vm)
        title('Spikes for ' + str(name))
        xlabel('Time (msec)')
        ylabel('Membrane Potential (V)')
        ylim([0,10])
        show()


if __name__ == "__main__":

    neuron1 = neuron(30)
    print(neuron1.firingRate())

