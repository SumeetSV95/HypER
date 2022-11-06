import os
from torch import nn, tensor
import numpy
import pickle
import dill
from collections import defaultdict


class Data:
    def __init__(self):
        self.entityToId = {}
        self.relationToId = {}
        self.entityEmbeddings = None
        self.relationEmbeddings = None
        self.trainingData = []
        self.testData = []
        self.validationData = []
        self.trainingIdsPairs = defaultdict(list)
        self.validationIdsPairs = defaultdict(list)
        self.testIdsPairs = defaultdict(list)

    def generateEntityRelation(self, file):
        entitySet = set()
        relationSet = set()
        with open(file) as f:
            data = f.read()
            data = data.split('\n')
            for d in data:
                if d:
                    e1, r, e2 = d.split('\t')
                    entitySet.add(e1)
                    entitySet.add(e2)
                    relationSet.add(r)
        return entitySet, relationSet

    def generate_data(self, file, lis, name):
        with open(file) as f:
            data = f.read()
            data = data.split('\n')
            if name == 'train':
                temp = self.trainingIdsPairs
            elif name == 'test':
                temp = self.testIdsPairs
            else:
                temp = self.validationIdsPairs
            for i, d in enumerate(data):
                if d:
                    e1, r, e2 = d.split('\t')
                    temp[(self.entityToId[e1], self.relationToId[r])].append(self.entityToId[e2])
                    lis.append(
                        [self.entityEmbeddings(tensor(self.entityToId[e1])),
                         self.relationEmbeddings(tensor(self.relationToId[r])),
                         self.entityEmbeddings(tensor(self.entityToId[e2]))])

        return lis

    def load_data(self, filepath, d1=5, d2=5):
        print(filepath)
        testFile = os.path.join(filepath, "test.txt")
        trainFile = os.path.join(filepath, "train.txt")
        validationFile = os.path.join(filepath, "valid.txt")
        print(trainFile)
        entitySet = set()
        relationSet = set()
        entitySetTrain, relationSetTrain = self.generateEntityRelation(trainFile)
        entitySetTest, relationSetTest = self.generateEntityRelation(testFile)
        entitySetVal, relationSetVal = self.generateEntityRelation(validationFile)
        entities, relations = list(entitySetTrain.union(entitySetTest).union(entitySetVal)), list(
            relationSetTrain.union(relationSetVal).union(relationSetTest))
        self.entityEmbeddings = nn.Embedding(len(entities), d1)

        self.relationEmbeddings = nn.Embedding(len(relations), d2)
        # print(self.relationEmbeddings, self.entityEmbeddings)
        for i in range(len(entities)):
            self.entityToId[entities[i]] = i
        for i in range(len(relations)):
            self.relationToId[relations[i]] = i
        self.trainingData = self.generate_data(trainFile, self.trainingData, "train")
        self.testData = self.generate_data(testFile, self.testData, 'test')
        self.validationData = self.generate_data(validationFile, self.validationData, 'validation')


# convert a list of tensors to tensors https://stackoverflow.com/questions/61359162/convert-a-list-of-tensors-to-tensors-of-tensors-pytorch

if __name__ == '__main__':
    data = Data()
    data.load_data(os.path.curdir)
    if not os.path.exists('data_obj.pickle'):
        open('data_obj.pickle', 'x')
        with open('data_obj.pickle', 'wb') as f:
            dill.dump(data, f)
    with open('data_obj.pickle', 'rb') as f:
        data = dill.load(f)
    print(data.trainingData.shape)
