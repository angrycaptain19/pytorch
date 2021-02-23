from statistics import mean
import matplotlib.pyplot as plt
import argparse

ROLL_LEN = 5

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=True,
    help="Filename of text file with epoch data in it.")
args = vars(ap.parse_args())

def parseFile (filename):
    with open(filename, 'r') as inputFile:
        epochLosses = []
        trainingAccuracies = []
        validationAccuracies = []
        for line in inputFile:
            if 'Epoch:' in line:
                epoch = int(line.split()[1][:-1])
                loss = float(line.split()[-1])
                if len(epochLosses) < epoch:
                    epochLosses.append([loss])
                else:
                    epochLosses[epoch - 1].append(loss)
            if 'TRAINING' in line:
                correct = float(line.split()[1])
                outOf = float(line.split()[4])
                trainingAccuracies.append(correct/outOf)
            if 'VALIDATION' in line:
                correct = float(line.split()[1])
                outOf = float(line.split()[4])
                validationAccuracies.append(correct/outOf)
    epochTmp = []
    for lossList in epochLosses:
        epochTmp.append(mean(lossList))
    epochLosses = epochTmp
    return epochLosses, trainingAccuracies, validationAccuracies



losses, train, validation = parseFile (args['file'])
epochs = len(losses)
assert len(losses) == len(train)
assert len(losses) == len(validation)
x = range(epochs - ROLL_LEN)
for i in range(epochs):
    if i > ROLL_LEN - 1:
        losses.append(sum(losses[i - ROLL_LEN:i]) / ROLL_LEN)
        train.append(sum(train[i - ROLL_LEN:i]) / ROLL_LEN)
        validation.append(sum(validation[i - ROLL_LEN:i]) / ROLL_LEN)
losses = losses[epochs:]
train = train[epochs:]
validation = validation[epochs:]
plt.plot(x, [loss for loss in losses])
plt.plot(x, [t for t in train])
plt.plot(x, [v for v in validation])
plt.show()
