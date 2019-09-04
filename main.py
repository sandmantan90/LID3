'''
Main Wrapper to be called from the bash script
It takes in all arguements from the config.yaml file and runs
Train or Test or Predict with necessary preProcessing steps

Saving only weights after training
Testing saves the various metrics as np array

Need to write new code for prediction later on
'''
import argparse
import yaml
import CRNN
import train
import test
import torch
import numpy as np
import pickle
import dataLoader
from createCombinations import createCombos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',dest="config", default="config.yaml")
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fileName = createCombos()
    file = open(fileName,'rb')
    combinations = pickle.load(file)
    file.close()
    
    trainLoader = dataLoader.dataLoaders(config,True)
    testLoader = dataLoader.dataLoaders(config,False)
    
    for combo in combinations:
        print('\n',combinations.index(combo) + 1)
        print('\n',combo)
        for x,y in combo.items():
            config[x] = y
        config['runName'] = str(combinations.index(combo))
        model = CRNN.CRNN(config)
        if config['cuda'] == 1:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
    
        if config['train']:
            model, lossHist = train.trainModel(config,model,trainLoader)
            torch.save(model.state_dict(), 
                       config['modelLoc'] + config['runName'] + "Weights.pth")
            
            if config['test']:
                confMatrix, metrics = test.testModel(config,model,testLoader)
                np.save(config['modelLoc'] + config['runName'] + 
                        + str(config['recTest']) + "Confusion.npy",confMatrix)
                f = open(config['modelLoc'] + config['runName'] +
                         str(config['recTest']) +"Metrics.txt",'w+')
                f.write(metrics)
                f.close()
                
        elif config['test']:
            model.load_state_dict(torch.load(
                    config['modelLoc'] + config['runName'] + "Weights.pth"))
            confMatrix, metrics = test.testModel(config,model,testLoader)
            np.save(config['modelLoc'] + config['runName'] + 
                        + str(config['recTest']) + "Confusion.npy",confMatrix)
            f = open(config['modelLoc'] + config['runName'] +
                         str(config['recTest']) +"Metrics.txt",'w+')
            f.write(metrics)
            f.close()
        del model
        