'''
Testing/Validation Script

Will take in the Model location and testdirectory, testLoader as input
and find recall, precision and F1 score by inferences.
Output: Recall, Precision, F1Score Report and Confusion Matrix
'''

import torch
import CRNN
import dataLoader
import numpy as np
import preAug
from sklearn import metrics

def testModel(config,model,testLoader):
    if config['cuda'] == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    with torch.no_grad():
        predictions = []
        groundTruths = []
        for data in testLoader:
            
            fileNos,labels = data
            labelsList = labels.tolist()
            fileNosList = fileNos.tolist()
            
            imagesList, labelsList = preAug.preAugmentation(config,
                                                            fileNosList,
                                                            labelsList, False) 
            
            images = torch.FloatTensor(len(imagesList), config['inShape'][0],
                                       config['inShape'][1], config['inShape'][2])
            for i in range(len(imagesList)):
                images[i] = torch.from_numpy(np.array(imagesList[i]))
            images = images.to(device)
            '''
            labels = torch.LongTensor(len(labelsList))
            i = -1
            for l in labelsList:
                i += 1
                labels[i] = torch.from_numpy(np.array(l))
            labels = labels.to(device)
            '''
            probabilities = model(images)
            pred = np.array((torch.argmax(probabilities,-1)).cpu())
            for i in range(len(labels)):
                predictions.append(pred[i])
                groundTruths.append(labelsList[i])
            
        confMatrix = metrics.confusion_matrix(groundTruths, predictions)
        perf = metrics.classification_report(groundTruths, predictions,
                                             digits = config['outClasses'])
        print('\nTest Performance\n')
        print(perf)
        print('\nConfusion Matrix\n')
        print(confMatrix)
        return confMatrix, perf
    