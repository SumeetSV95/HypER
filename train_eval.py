import dill
import numpy as np
from load_data import Data
import os
from torch import device, nn, tensor, optim
from hyper import HypER
import pickle
import torch


class Experiment:
    def __init__(self, learning_rate=0.001, cuda=False, batch_size=128, nf=32, lf=3, d_e=200, d_r=200,
                 num_of_iterations=100, load_saved_Model=True, save_data=True, load_data=True, train=True):
        self.learning_rate = learning_rate
        self.cuda = cuda
        self.batch_size = batch_size
        self.nf = nf
        self.lf = lf
        self.d_e = d_e
        self.d_r = d_r
        self.num_of_iterations = num_of_iterations
        self.load_saved_Model = load_saved_Model
        self.path = os.path.join(os.path.curdir, "model.model")
        self.load_data = load_data
        self.save_data = save_data
        self.train = train

    def eval(self, model: HypER, data: Data, isTest=False):
        if not isTest:
            lis = list(data.validationIdsPairs.keys())
        else:
            lis = list(data.testIdsPairs.keys())
        # print(lis)
        hits = [[], [], []]
        for i in range(0, len(lis), exp.batch_size):
            if not isTest:
                entities, relations, targets = self.getBatch(lis, data.validationIdsPairs, data, i, exp.batch_size,
                                                             len(data.entityToId.keys()))
            else:
                entities, relations, targets = self.getBatch(lis, data.testIdsPairs, data, i, exp.batch_size,
                                                             len(data.entityToId.keys()))
            if self.cuda:
                entities = entities.cuda()
                relations = relations.cuda()
                # targets = torch.from_numpy(targets).cuda()
            # print(entities, relations, targets)
            # predictions = np.random.random((len(lis), len(data.entityToId.keys())))  # todo get the prediction from the
            # model

            # predictions = torch.from_numpy(predictions)
            predictions = model.forward(entities, relations)
            # print(predictions)
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu()
            rows, columns = np.where(targets == 1.0)
            # print(rows, columns)
            hit_levels = [1, 5, 10]

            for i, row in enumerate(rows):
                arr = np.where(sort_idxs[row] == columns[i])
                for val in arr:
                    rank = val + 1
                    for i, hit_level in enumerate(hit_levels):
                        if rank <= hit_level:
                            hits[i].append(1)
                        else:
                            hits[i].append(0)
        print(np.mean(hits[0]))
        print(np.mean(hits[1]))
        print(np.mean(hits[2]))
        return np.mean(hits[0]), np.mean(hits[1]), np.mean(hits[2])

    def getBatch(self, lis, dictMapping, dataObj, i, batchSize, entitySize):
        end = min(len(lis), i + batchSize)
        targets = np.zeros((end - i, entitySize))
        entities = []
        relations = []
        iterator = 0
        for index in range(i, end):
            ele = lis[index]
            eid, r = ele
            entities.append(eid)
            relations.append(r)
            e2_ids = dictMapping[ele]
            for id in e2_ids:
                targets[iterator][id] = 1
            iterator += 1

        if self.cuda:
            entities = torch.from_numpy(np.array(entities)).cuda()
            relations = torch.from_numpy(np.array(relations)).cuda()
            entities = dataObj.entityEmbeddings(tensor(entities)).cuda()
            relations = dataObj.relationEmbeddings(tensor(relations)).cuda()
        else:
            entities = dataObj.entityEmbeddings(tensor(entities))
            relations = dataObj.relationEmbeddings(tensor(relations))
        return entities, relations, targets

    def train_and_eval(self, data):
        model = HypER(data, self.nf, self.lf, d_r=self.d_r, d_e=self.d_e)
        if self.load_saved_Model:
            model.load_state_dict(torch.load(self.path))
        if self.cuda:
            model.cuda()
        model.init()
        opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        maxEval = -float('inf')
        for iter in range(self.num_of_iterations):
            model.train()
            losses = []
            lis = list(data.trainingIdsPairs.keys())
            # print(len(lis))
            # print(len(data.trainingData))
            np.random.shuffle(lis)
            for params in model.parameters():
                params.requires_grad = True
            for i in range(0, len(lis), exp.batch_size):
                entities, relations, targets = self.getBatch(lis, data.trainingIdsPairs, data, i, exp.batch_size,
                                                             len(data.entityToId.keys()))

                targets = torch.from_numpy(targets)
                opt.zero_grad()
                if self.cuda:
                    entities = entities.cuda()
                    relations = relations.cuda()
                    targets = targets.cuda()
                predictions = model.forward(entities, relations)
                predictions = predictions.to(torch.float64)

                # print(predictions.shape)
                # predictions = torch.from_numpy(predictions)  # comment this out later
                # predictions.requires_grad = True  # comment this out later

                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
            losses.append(loss.item())
            print(losses)
            print(iter)
            print(np.mean(losses))
            with torch.no_grad():
                print("validation")
                if iter % 2 == 0:
                    h1, h5, h10 = self.eval(model, data)
                    if h10 > maxEval:
                        print("saving model")
                        maxEval = h10
                        torch.save(model.state_dict(), self.path)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("running using cuda")
        exp = Experiment(cuda=True)
    else:
        exp = Experiment()
    if exp.load_data:
        with open("data_obj.pickle", 'rb') as data_file:
            print("loading data...")
            data = pickle.load(data_file)
    else:
        data = Data()
        filepath = os.path.join(os.path.curdir, "WN18")
        data.load_data(filepath, exp.d_e, exp.d_r)
    print(data.trainingIdsPairs)
    print(data.validationIdsPairs)
    print(data.testIdsPairs)
    if torch.cuda.is_available():
        data.entityEmbeddings = data.entityEmbeddings.cuda()
        data.relationEmbeddings = data.relationEmbeddings.cuda()
    # model = HypER(data, exp.nf, exp.lf, d_r=exp.d_r, d_e=exp.d_e)
    # exp.eval(model, data)
    if exp.train:
        exp.train_and_eval(data)
    else:
        model = model = HypER(data, exp.nf, exp.lf, d_r=exp.d_r, d_e=exp.d_e)
        model.load_state_dict(torch.load(exp.path))
        if exp.cuda:
            model.cuda()
        with torch.no_grad():
            print("test scores with nn..")
            exp.eval(model, data, True)
            print("test score without nn")
            model.fc.weight.data.fill_(1.0)
            model.fc.bias.data.fill_(1.0)
            model.fc1.weight.data.fill_(1.0)
            model.fc1.bias.data.fill_(1.0)
            exp.eval(model, data, True)

    if exp.save_data:
        with open("data_obj.pickle", "wb") as data_file:
            print("saving data..")
            pickle.dump(data, data_file)

