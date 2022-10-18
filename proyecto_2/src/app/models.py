from .base import Process_Data
import pandas as pd
import numpy as np




class dataframe(Process_Data):

    def __init__(self, filepath, features = None, labels = None):
        self.filepath = filepath
        self.features = features 
        self.labels = labels


        self.df = pd.read_csv(self.filepath)

    def list_features(self):
        #define list features 
        features = self.features
        labels = self.labels

        #remove features if features != None
        if (features == None):
            features_name = self.df.columns

        else: 
            features_name = features
            
        #remove labels into dataset
        if (labels == None):
            pass
        else: 
            for i in labels:
                features_name = features_name.drop(i)

        return list(features_name)

    def list_labels(self):

        label = self.labels
        if (label == None):
            list_labels = []
        else:
            list_labels = label
        return list_labels

    def new_dataset(self):
        #new dataset pandas 
        df_features =  self.df[self.list_features()]
        df_label = self.df[self.list_labels()]

        # change dataset pandas to numpy
        features_tensor = df_features.to_numpy()
        labels_tensor = df_label.to_numpy()

        #return data in matriz
        return features_tensor, labels_tensor


def perceptron_train(name_csv, labels):

    #--------- PERCEPTRON TRAIN---------#

    #import dataframe
    data = dataframe(name_csv, features = None, labels = labels)
    #remove columns of some features
    features, labels_data = data.new_dataset()
    #reshape labels column
    labels = labels_data.flatten()
    #weight + bias in position 0
    weight =  np.zeros(len(features[0]) + 1)


    while(True):
        m = 0

        for count, label in enumerate(labels):

            #compute function of perceptron
            func = np.matmul(features[count], weight[1:]) + weight[0]

            #missclasified
            if ((label * func) <= 0):
                m += 1
                #update weight vector
                weight[1:] += features[count] * label
                #update bias
                weight[0] += label
                
        #if missclasified is = 0 
        if(m == 0):
            break
    return weight.tolist()


def pocket_train(name_csv, labels, max_iters = 100):

    

   #--------- POCKET TRAIN---------#

    #import dataframe
    data = dataframe(name_csv, features = None, labels = labels)
    #remove columns of some features
    features, labels_data = data.new_dataset()
    #reshape labels column
    labels = labels_data.flatten()
    #weight + bias in position 0
    weight =  np.zeros(len(features[0]) + 1)
    max_bad = len(features)

    for i in range(max_iters):
        m = 0

        for count, label in enumerate(labels):

            #compute function of perceptron
            func = np.matmul(features[count], weight[1:]) + weight[0]

            #missclasified
            if ((label * func) <= 0):
                m += 1
                #update weight vector
                weight[1:] += features[count] * label
                #update bias
                weight[0] += label
                
        if(m < max_bad):
            max_bad = m
            best_weight = weight

    return best_weight.tolist()


def perceptron_and_pocket_test(features, weight):


     #--------- PERCEPTRON TEST---------#
    #list of outputs
    out = []
    
    #compute functions of perceptron and pocket
    for feature in features:
        func_test = (np.matmul(feature, weight[1:])) + weight[0]
        value = np.sign(func_test)
        out.append(int(value))

    return out