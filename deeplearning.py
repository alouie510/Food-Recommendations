import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd


def deeplearning(calories, overUnder, restrictions, allergies):
    # LOGISTIC REGRESSION CLASSIFICATION VARIABLES
    testDataSize = 30 # DO NOT TOUCH: amount of recommended food items outputted (keep under 10% of total)
    inputSize = 4 # DO NOT TOUCH: num of food item categories for classification
    numClasses = 2 # DO NOT TOUCH: num of food classification classes (like/dislike)
    numEpochs = 30
    trainBatchSize = 20
    testBatchSize = 15
    learningRate = 0.0001
    likeWeight = 1.05  # added weight to mean rating value for cutoff point

    # LINEAR REGRESSION VARIABLES
    linDataSize = 3000 # DO NOT TOUCH: amount of data used for lin reg training (prevents overfitting)
    linNumEpochs = 250
    linLearningRate = 0.01


    # LAMBDA FUNCTIONS
    # Function Description: reduces data from raw data to data amount per calories ratio (protein, fat, sodium)
    def quantityToRatio(row, option):
        calories = row['calories']
        if calories > 0:
            return row[option] / calories
        else:
            return row[option]

    # Function Description: turn food ratings into percentages (better interpretation)
    def normalizeRatings(row, maxi):
        if maxi < 5:
            maxi = (maxi + 10) / 3
        if maxi > 0:
            if row['rating'] == maxi:
                return 99
            else:
                return (row['rating'] / maxi) * 100
        else:
            return row['rating']

    # Function Description: deciphers between 'liked' and 'disliked' classes
    def labelLike(row, mean, likeWeight):
        if row['rating'] > likeWeight*mean:
            return 1 # GOOD
        else:
            return 0 # BAD

    # Function Description: provides much needed variety within rating category - preprocessing
    def adjustRating(row, calories, overUnder, allergies):
        rating = row['rating']
        if overUnder == "Gain":
            if row['calories'] >= calories:
                rating = rating * 1.05
            else:
                rating = rating * .95
        else:
            if row['calories'] <= calories:
                rating = rating * 1.05
            else:
                rating = rating * .95

        if ("dairy" in allergies) and (row['dairy free'] == 1):
            rating = rating * 1.1
        if ("egg" in allergies) and (row['egg'] == 0):
            rating = rating * 1.1
        if ("tree" in allergies) and (row['tree nut free'] == 1):
            rating = rating * 1.1
        if ("peanuts" in allergies) and (row['peanut free'] == 0):
            rating = rating * 1.1
        if ("shell" in allergies) and (row['shellfish'] == 0):
            rating = rating * 1.1
        if ("soy" in allergies) and (row['soy free'] == 1):
            rating = rating * 1.1
        if ("fish" in allergies) and (row['fish'] == 0):
            rating = rating * 1.1
        if ("gluten" in allergies) and (row['wheat/gluten-free'] == 1):
            rating = rating * 1.1

        return rating


    # LINEAR REGRESSION CLASS
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(4, 1)
        def forward(self, point):
            output = self.linear(point)
            return output

    # LOGISTIC REGRESSION CLASSIFICATION CLASS
    class LogisticRegression(nn.Module):
        def __init__(self, inputSize, numClasses):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(inputSize, numClasses)
        def forward(self, point):
            output = self.linear(point)
            return output


    # READ DATA + PREPROCESS
    data = pd.read_csv("epi_r.csv", encoding = 'utf8')
    df = data[['title','rating','calories','protein','fat','sodium','kosher','pescatarian','vegan','vegetarian', 'dairy free','egg','tree nut free','peanut free','shellfish', 'soy free','fish','wheat/gluten-free','sugar conscious']]
    df.replace(r'\s+', np.nan, regex=True)
    df.replace('', np.nan, regex=True)
    df.reset_index(level=0, inplace=True)
    df = df.dropna() # rids of all NAN rows - unable to be used

    df['protein'] = df.apply(lambda row: quantityToRatio (row, "protein"),axis=1)
    df['fat'] = df.apply(lambda row: quantityToRatio (row, "fat"),axis=1)
    df['sodium'] = df.apply(lambda row: quantityToRatio (row, "sodium"),axis=1)


    # FIND MATCHING FOOD ITEMS IN DATASET
    filterDF = df
    trainDF = df
    # calories
    if overUnder == "Gain": # gain weight
        filterDF = df[df['calories'] >= calories].sort_values(by=['calories', 'rating'], ascending=True, na_position='last').drop_duplicates(subset=['title'])
    else: # lose weight - default
        filterDF = df[df['calories'] <= calories].sort_values(by=['calories', 'rating'], ascending=False, na_position='last').drop_duplicates(subset=['title'])
    # restrictions
    if restrictions == "vegetarian":
        filterDF = filterDF[filterDF['vegetarian'] == 1]
    elif restrictions == "vegan":
        filterDF = filterDF[filterDF['vegan'] == 1]
    elif restrictions == "pescatarian":
        veg = filterDF[filterDF['vegan'] == 1]
        pesc = filterDF[filterDF['pescatarian'] == 1]
        filterDF = pd.concat([veg,pesc]).drop_duplicates().reset_index(drop=True)
    elif restrictions == "kosher":
        filterDF = filterDF[filterDF['kosher'] == 1]
    elif restrictions == "diabetic":
        filterDF = filterDF[filterDF['sugar conscious'] == 1]
    # allergies
    if "dairy" in allergies:
        filterDF = filterDF[filterDF['dairy free'] == 1]
    if "egg" in allergies:
        filterDF = filterDF[filterDF['egg'] == 0]
    if "tree" in allergies:
        filterDF = filterDF[filterDF['tree nut free'] == 1]
    if "peanuts" in allergies:
        filterDF = filterDF[filterDF['peanut free'] == 0]
    if "shell" in allergies:
        filterDF = filterDF[filterDF['shellfish'] == 0]
    if "soy" in allergies:
        filterDF = filterDF[filterDF['soy free'] == 1]
    if "fish" in allergies:
        filterDF = filterDF[filterDF['fish'] == 0]
    if "gluten" in allergies:
        filterDF = filterDF[filterDF['wheat/gluten-free'] == 1]

    filterDF = filterDF[:testDataSize]
    trainDF = df.drop(filterDF['index']).drop_duplicates(subset=['title'])
    trainDF['rating'] = trainDF.apply(lambda row: adjustRating(row, calories, overUnder, allergies),axis=1)
    trainDF['like'] = trainDF.apply(lambda row: labelLike (row, trainDF['rating'].mean(), likeWeight),axis=1)
    trainDF = trainDF[['like','rating','calories','protein','fat','sodium','sugar conscious','title']].sort_values(by=['calories'], ascending=False, na_position='last').drop_duplicates(subset=['title'])


    # LINEAR REGRESSION TRAINING + RESULTS
    linTrainDF = trainDF.sample(frac=1).reset_index(drop=True).sort_values(by=['calories'], ascending=False, na_position='last')

    linTrainLabelDF = linTrainDF['rating']
    linTrainLabelDF = linTrainLabelDF[:linDataSize]

    linTrainDataDF = linTrainDF[['protein', 'fat','sodium','sugar conscious']]
    linTrainDataDF = linTrainDataDF[:linDataSize]

    linTrainLabelTorch = torch.Tensor(np.asarray(linTrainLabelDF).reshape(-1,1))
    linTrainDataTorch = torch.Tensor(np.asarray(linTrainDataDF).reshape(-1,4))

    filterDFLin = filterDF[['protein','fat','sodium','sugar conscious']].dropna()
    filterLinTorch = torch.from_numpy(np.array(filterDFLin)).float()

    linModel = LinearRegressionModel()

    linCriterion = nn.MSELoss()
    linOptimiser = torch.optim.SGD(linModel.parameters(), lr = linLearningRate)

    for epoch in range(linNumEpochs):
        prediction = linModel(linTrainDataTorch)

        loss = linCriterion(prediction, linTrainLabelTorch)
        linOptimiser.zero_grad()
        loss.backward()
        linOptimiser.step()

    prediction = linModel(filterLinTorch).data.numpy()
    prediction = pd.DataFrame.from_records(prediction)
    prediction.columns = ['rating']
    prediction = prediction.reset_index(drop=True)
    prediction = prediction.apply(lambda row: normalizeRatings (row, prediction['rating'].max()),axis=1)


    # LOGISTIC REGRESSION TRAINING + RESULTS
    filterTitles = filterDF['title']
    filterDF = filterDF[['protein','fat','sodium','sugar conscious']].dropna()
    trainDF = trainDF[['like','protein','fat','sodium','sugar conscious']].dropna()
    trainDF = trainDF[:2250]

    filterTorch = torch.from_numpy(np.array(filterDF))
    trainTorch = torch.from_numpy(np.array(trainDF))

    testLoader = torch.utils.data.DataLoader(dataset = filterTorch, batch_size = testBatchSize, shuffle = False)
    trainLoader = torch.utils.data.DataLoader(dataset = trainTorch, batch_size = trainBatchSize, shuffle = True)

    model = LogisticRegression(inputSize, numClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

    for epoch in range(numEpochs):
        for i, block in enumerate(trainLoader):
            label = Variable(block[:,0].long())
            features = Variable(block[:,range(1,5)].float())

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()


    # APPEND CLASSIFICATION TO DATAFRAME
    labels = np.empty([testDataSize, 1], dtype=int)
    li = 0

    for block in testLoader:
        features = block.float()
        output = model.forward(features)
        addOutput = output.data.numpy()

        for rowI in range(len(addOutput)):
            row = addOutput[rowI]
            if row[0] > row[1]:
                labels[li] = 0
            else:
                labels[li] = 1

            li = li + 1

    labelsFrame = pd.DataFrame.from_records(labels)
    labelsFrame = labelsFrame.reset_index(drop=True)
    filterDF = filterDF.reset_index(drop=True)
    filterTitles = filterTitles.reset_index(drop=True)

    filterDF = pd.concat([labelsFrame,filterDF],axis=1)
    filterDF = pd.concat([filterTitles, filterDF],axis=1)
    filterDF.columns = ['title', 'score', 'protein', 'fat', 'sodium', 'sugar conscious']

    filterDF = filterDF[filterDF['score'] != 0]
    filterDF = filterDF['title']
    filterDF = pd.concat([filterDF, prediction],axis=1).dropna()
    filterDF.columns = ['title', 'rating']
    filterDF = filterDF.sort_values(by=['rating'], ascending=False, na_position='last')

    if filterDF['title'].empty or filterDF['rating'].empty:
        col_names =  ['A', 'B']
        empty  = pd.DataFrame(columns = col_names)
        return empty, empty
    else:
        #for index, row in filterDF.iterrows():
        #    row['title'] = row['title'] + ": " + str(row['rating'])
        #filterDF = filterDF['title']
        return filterDF['title'], filterDF['rating']
