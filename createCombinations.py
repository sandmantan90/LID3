import random
import pickle

def createCombos():
    epochs = [40, 60, 80, 100, 120]
    optimizers = ['Adam', 'Momentum', 'Adagrad', 'RMSProp']
    learningRates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    batchSizes = [64, 128, 256, 512, 1024]
    rnnHiddenSizes = [128, 256, 512]
    rnnLayers = [1,2]
    noFCLayers = [1,2]
    fcSizes = [128,256]
    cnnActivation = ['relu','elu','leaky_relu','tanh']
    linearActivation = ['relu','elu','leaky_relu','tanh']
    cnnDropRates =  [0, 0.1, 0.2]
    linearDropRates = [0, 0.1, 0.25, 0.4]
    lstmDropRates = [0, 0.1, 0.25, 0.4]
    
    lists = []
    for epo in epochs:
        for opt in optimizers:
            for lr in learningRates:
                for bs in batchSizes:
                    for rnnHS in rnnHiddenSizes:
                        for rnnL in rnnLayers:
                            for noF in noFCLayers:
                                for fcSize in fcSizes:
                                    for cnnAct in cnnActivation:
                                        for lnAct in linearActivation:
                                            for cnnDrop in cnnDropRates:
                                                for linearDrop in linearDropRates:
                                                    for lstmDrop in lstmDropRates:
                                                        lists.append({
                                                                'iterations':epo,
                                                                'optimizer': opt,
                                                                'learningRate':lr,
                                                                'batchSize': bs,
                                                                'rnnHiddenSize': rnnHS,
                                                                'rnnLayers': rnnL,
                                                                'fcHiddenSize':fcSize,
                                                                'fcLayers': noF,
                                                                'cnnActivation': cnnAct,
                                                                'linearActivation': lnAct,
                                                                'cnnDrop': cnnDrop,
                                                                'lstmDrop': lstmDrop,
                                                                'linearDrop': linearDrop
                                                                })
    random.shuffle(lists)
    fileName = 'combos.pkl'
    file = open(fileName,'wb+')
    pickle.dump(lists[:60],file)
    file.close()
    return fileName