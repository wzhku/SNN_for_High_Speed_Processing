# coding: utf-8
"""Training of the SNN on the DVS128 Gesture dataset.

This is the training script of `motion_recognition.py`, which runs inference with a
trained checkpoint. Both scripts share the same network definition, the same neuron kernels
and the same `network.yaml`.

The three-layer fully connected network is trained end-to-end on the native event streams
by backpropagation through time, with a surrogate gradient for the spiking
non-differentiability. Weight clamping and activation clamping enforce the constraints
representative of the hardware implementation; L2 regularization and data augmentation
enhance training effectiveness:

  * weight clamping     : [-5, +5] for the two hidden layers, [-3, +3] for the output layer
  * activation clamping : [0, 5] at the input of every neuron layer
  * L2 regularization   : lam / (2 N) * sum(W^2), N being the number of training samples
  * data augmentation   : random resized crop, perspective transform and rotation

Weight-noise injection in conductance-equivalent units is available as an option. It is off
by default, which is the setting used for the DVS128 Gesture results. The same mechanism, at
a standard deviation of 10 uS in conductance-equivalent units, was used for the SHD task.

Usage:
    python train.py                  # settings used for the DVS128 Gesture results
    python train.py --sigma-g 10     # with weight-noise injection enabled
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, InterpolationMode, RandomPerspective,
                                    RandomResizedCrop, RandomRotation)

import slayerSNN as snn
from slayerSNN.learningStats import learningStats

from motion_recognition import DvsGestureDataset, Network

SENSOR_SIZE = (128, 128)   # spatial resolution of the DVS128 event camera


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='network.yaml', help='network and dataset configuration')
    parser.add_argument('--out', default='runs/dvsgesture', help='directory for checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005, help='Adam learning rate')
    parser.add_argument('--l2-lambda', type=float, default=5.0, help='L2 regularization strength')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0, help='data loading worker processes')
    parser.add_argument('--no-augment', action='store_true', help='disable data augmentation')
    parser.add_argument('--sigma-g', type=float, default=None,
                        help='standard deviation of the injected weight noise, in uS of '
                             'conductance; omit to train without noise injection')
    parser.add_argument('--g-max', type=float, default=150.0,
                        help='largest conductance programmed on the array, in uS')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_augmentation():
    """Spatial augmentation applied to the training samples only."""
    return Compose([
        RandomResizedCrop(SENSOR_SIZE, scale=(0.6, 1.0), interpolation=InterpolationMode.NEAREST),
        RandomPerspective(),
        RandomRotation(25),
    ])


class GaussianWeightNoise(torch.nn.Module):
    """Gaussian weight-noise injection, parameterised in conductance-equivalent units."""

    def __init__(self, sigma_g, g_max):
        super(GaussianWeightNoise, self).__init__()
        self.sigma_g = sigma_g
        self.g_max = g_max

    def forward(self, weight):
        if not self.training:
            return weight
        return weight + torch.randn_like(weight) * self.sigma_g / self.g_max * weight.abs().max().detach()


def l2_penalty(net, lam, numTrainSamples):
    """L2 regularization over the synaptic weights, lam / (2 N) * sum(W^2)."""
    if lam == 0:
        return 0.0
    reg = sum((layer.weight ** 2).sum() for layer in (net.fc1, net.fc2, net.fc))
    return reg * lam / (2 * numTrainSamples)


def train_one_epoch(net, dataloader, error, optimizer, device, stats, lam, numTrainSamples):
    net.train()
    for input, target, label in dataloader:
        input  = input.to(device)
        target = target.to(device)

        output = net.forward(input)
        loss = error.numSpikes(output, target) + l2_penalty(net, lam, numTrainSamples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.training.numSamples     += len(label)
        stats.training.lossSum        += loss.cpu().data.item()


@torch.no_grad()
def evaluate(net, dataloader, error, device, stats):
    net.eval()
    for input, target, label in dataloader:
        input  = input.to(device)
        target = target.to(device)

        output = net.forward(input)

        stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.testing.numSamples     += len(label)
        stats.testing.lossSum        += error.numSpikes(output, target).cpu().data.item()


def save_checkpoint(path, net, optimizer):
    torch.save({'Model': {'model_state_dict': net.state_dict(),
                          'opt_state_dict': optimizer.state_dict()}}, path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)
    netParams = snn.params(args.config)

    weightNoise = None if args.sigma_g is None else GaussianWeightNoise(args.sigma_g, args.g_max)
    net = Network(netParams, weightNoise=weightNoise).to(device)
    error = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

    trainingSet = DvsGestureDataset(datasetPath = netParams['training']['path']['inTrain'],
                                    sampleFile  = netParams['training']['path']['train'],
                                    transform   = None if args.no_augment else build_augmentation())
    testingSet  = DvsGestureDataset(datasetPath = netParams['training']['path']['inTest'],
                                    sampleFile  = netParams['training']['path']['test'])

    trainLoader = DataLoader(dataset=trainingSet, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)
    testLoader  = DataLoader(dataset=testingSet, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)

    print(f"training samples {len(trainingSet)}, testing samples {len(testingSet)}, "
          f"time steps {int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])}")
    print(f"lr {args.lr}, batch size {args.batch_size}, L2 lam {args.l2_lambda}, "
          f"augmentation {not args.no_augment}, "
          f"weight noise {'off' if args.sigma_g is None else f'{args.sigma_g} uS'}")

    stats = learningStats()
    bestAccuracy = 0.0

    for epoch in range(args.epochs):
        tSt = time.time()

        train_one_epoch(net, trainLoader, error, optimizer, device, stats,
                        args.l2_lambda, len(trainingSet))
        evaluate(net, testLoader, error, device, stats)
        stats.update()

        testAccuracy = stats.testing.accuracyLog[-1]
        if testAccuracy > bestAccuracy:
            bestAccuracy = testAccuracy
            save_checkpoint(os.path.join(args.out, 'model.tar'), net, optimizer)
        save_checkpoint(os.path.join(args.out, 'checkpoint.tar'), net, optimizer)

        print(f"epoch {epoch + 1:4d}/{args.epochs} | "
              f"train {stats.training.accuracyLog[-1]:.4f} (loss {stats.training.lossLog[-1]:.4e}) | "
              f"test {testAccuracy:.4f} (loss {stats.testing.lossLog[-1]:.4e}) | "
              f"best {bestAccuracy:.4f} | {time.time() - tSt:.1f} s")


if __name__ == '__main__':
    main()
