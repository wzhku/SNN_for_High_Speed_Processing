# coding: utf-8

import numpy as np
from scipy import io
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import slayerSNN as snn
from slayerSNN.learningStats import learningStats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile):        
        self.path = datasetPath        
        self.samples = np.loadtxt(sampleFile).astype('int')
            
    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        
        filename_data = self.path + str(inputIndex.item()).zfill(5) + '.mat'
        
        inputSpikes = torch.tensor(io.loadmat(filename_data)['Voxel'], dtype=torch.float)
        desiredClass = torch.zeros((11, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]

class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        slayer.srmKernel = self.Kernel(tau=netParams['neuron']['tauSr'], simulation=netParams['simulation'], mult=1, EPSILON = 1e-5)
        slayer.refKernel = self.Kernel(tau=netParams['neuron']['tauRef'], simulation=netParams['simulation'], mult=-10)
        
        self.slayer = slayer
        
        self.pad = lambda input: F.pad(input, (0, 0, 6, 6, 6, 6))
        self.pool1 = slayer.pool(7)
        
        self.fc1   = slayer.dense(800, 480, preHookFx=lambda input: torch.clamp(input, -5, 5))
        self.fc2   = slayer.dense(480, 120, preHookFx=lambda input: torch.clamp(input, -5, 5))        
        self.fc    = slayer.dense(120, 11,  preHookFx=lambda input: torch.clamp(input, -3, 3))
        # Apply weight clamping to mitigate the impact of RRAM conductivity fluctuations when performing inference on hardware
        
        self.clamp = lambda input: torch.clamp(input, 0, 5)
        # Apply activation clamping to 
        # (1) introduce nonlinearity by saturating extreme values for numerical stability and 
        # (2) prevent the output from exceeding limits during hardware inference
    
    def Kernel(self, tau, simulation, mult = 1, EPSILON = 0.01):
        # The kernel design here allows the SNN system to have time-scaling capabilities
        
        kernel = []
        for t in np.arange(0, simulation['tSample'], simulation['Ts']):
            kVal = mult / tau * math.exp(- t / tau)
            if abs(kVal) < EPSILON and t > tau:
                break
            kernel.append(kVal)
        return torch.tensor(kernel, dtype=torch.float)
 
    def forward(self, spikeInput):        
        Input = self.pool1(self.pad(spikeInput))
        
        netLayer1 = self.fc1(Input.reshape((spikeInput.shape[0], -1, 1, 1, spikeInput.shape[-1])))        
        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.clamp(netLayer1)))
        
        netLayer2 = self.fc2(spikeLayer1)            
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.clamp(netLayer2)))
        
        netLayer3 = self.fc(spikeLayer2)
        spikeLayer3 = self.slayer.spike(self.slayer.psp(self.clamp(netLayer3)))
        
        return spikeLayer3
 
def plotConfusion(outputs, labels, output_size):    
    confusion = torch.zeros(output_size, output_size)
    for o, t in zip(outputs.int(), labels.int()):
        confusion[o][t] += 1    
    
    plt.figure(figsize=(14, 14))
    ax = plt.subplot(1, 1, 1)
    cax = ax.matshow(confusion)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=30)
        
    for i, j in [[i, j] for i in range(confusion.shape[0]) for j in range(confusion.shape[1])]:
        plt.text(j, i, f"{confusion[i, j].int()}", family='Arial', fontsize=25, color='w', weight='bold', va='center', ha='center')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('True', fontdict={'family': 'Arial', 'size': 30})
    plt.ylabel('Predicted', fontdict={'family': 'Arial', 'size': 30})
    plt.xticks(fontproperties='Arial', fontsize=30)
    plt.yticks(fontproperties='Arial', fontsize=30)    
    plt.title(f"Accuracy = {torch.trace(confusion) / confusion.sum() * 100:.2f}%", fontdict={'family': 'Arial', 'size': 30})
    plt.show()
    

if __name__ == '__main__':    
    device = torch.device('cuda')
    
    netParams = snn.params('network.yaml')
    net = Network(netParams).to(device)
    error = snn.loss(netParams).to(device)
        
    model_data = torch.load('model.tar', map_location=device, weights_only=True)
    net.load_state_dict(model_data['Model']['model_state_dict'])
        
    dataset = nmnistDataset(datasetPath = netParams['training']['path']['inTest'],
                            sampleFile  = netParams['training']['path']['test'])
    dataloader = DataLoader(dataset=dataset, batch_size=10)
    
    stats = learningStats()
    
    outputTest = torch.tensor([])
    labelTest  = torch.tensor([])
            
    for i, (input, target, label) in enumerate(dataloader, 0):                
        input  = input.to(device)
        target = target.to(device)
        
        output = net.forward(input)
        outputTest_i = snn.predict.getClass(output)
        outputTest = torch.cat((outputTest, outputTest_i), dim=0)                
        labelTest = torch.cat((labelTest, label), dim=0)
                
        stats.testing.correctSamples += torch.sum( outputTest_i == label ).data.item()
        stats.testing.numSamples     += len(label)
    
        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        if i % 5 == 0:            
            stats.print(1, i, header=None, footer=None)
    
    plotConfusion(outputs=outputTest, labels=labelTest, output_size=11)
    