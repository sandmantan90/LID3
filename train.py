'''
Training Program
Call different models and use dataLoaders
and train them

Pass on trainLoader with fileNos and labels,
model, optimizer, learningRate, iterations,location of Train/Test folders,
batchSize, inputShape, output CLasses, rnnLayers and Size

Output returns a trained model and lossHistory over epochs
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import dataLoader
import CRNN
import preAug


def trainModel(config,model,trainLoader):
    if config['cuda'] == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    loss = nn.NLLLoss()
    learningRate = config['learningRate']
    opti = config['optimizer']
    if opti == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=learningRate) #Need to Tune
    elif opti == 'Momentum':
        optimizer = optim.Adam(model.parameters(),lr=learningRate,momentum = 0.9)
    elif opti == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(),lr=learningRate)
    elif opti == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(),lr=learningRate,momentum = 0.9)
        
    lossHistory = []
    runningLoss = 0
    print('\nTraining Begins\n')
    for epoch in range(1,config['iterations']+1):
        model.train(True)
        for data in trainLoader:
            fileNos, labels = data
            labelsList = labels.tolist()
            fileNosList = fileNos.tolist()
            
            imagesList, labelsList = preAug.preAugmentation(config,
                                                            fileNosList,
                                                            labelsList, True) 
            
            images = torch.FloatTensor(len(imagesList), config['inShape'][0],
                                       config['inShape'][1], config['inShape'][2])
            for i in range(len(imagesList)):
                images[i] = torch.from_numpy(np.array(imagesList[i]))
            images = images.to(device)
            
            labels = torch.LongTensor(len(labelsList))
            i = -1
            for l in labelsList:
                i += 1
                labels[i] = torch.from_numpy(np.array(l))
            labels = labels.to(device)
            
            optimizer.zero_grad()
            probabilities = model(images)
            losses = loss(probabilities,labels)
            losses.backward()
            optimizer.step()
            
            runningLoss += losses.item()
        lossHistory.append(runningLoss)
        print('EpochNum: '+str(epoch)+' RunningLoss: '+str(runningLoss))
        runningLoss = 0
            
    print('\nTraining Ends\n')
    return model, lossHistory