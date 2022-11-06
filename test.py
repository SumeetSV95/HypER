import random


def createDictionary(dictionary, f):
    data = f.read()
    data = data.split("\n")
    for line in data:
        data = line.split("\t")
        if len(data) < 2:
            continue
        dictionary[data[1]] = data[0]


def main():
    dir = "/Users/sumeet95/Downloads/GraphDBCourse/KGC_datasets/Example/"
    idToentity = {}
    idToRelation = {}
    lis = dir.strip().split("/")
    print(lis)
    dirname = lis[-2]
    with open(dir + "entity2id.txt") as f:
        createDictionary(idToentity, f)
    print(idToentity)

    with open(dir + "relation2id.txt") as f:
        createDictionary(idToRelation, f)
    print(idToRelation)
    with open(dir + "dataset.txt") as f:
        lines = []
        print(dirname+"/dataset.txt")
        with open(dirname+"/dataset.txt", "w") as file:
            data = f.read()
            data = data.split("\n")
            print(data)
            for line in data:
                data = line.split(" ")
                if len(data) < 3:
                    continue
                e1, e2, r = data[0], data[1], data[2]
                lines.append(idToentity[e1] + "\t" + idToRelation[r] + "\t" + idToentity[e2] + "\n")
            print(lines)
            file.writelines(lines)

    with open(dirname+"/dataset.txt") as file:
        data = file.read()
        data = data.split("\n")
        random.shuffle(data)
        size = len(data)
        print(size)
        train_size = int(size * 0.8)
        print(train_size)
        remainingData = size - train_size

        validation_end = train_size + int(remainingData * 0.5)
        print(validation_end)
        with open(dirname+"/train.txt", 'w') as f:
            f.writelines(f"{l}\n" for l in data[:train_size])

        with open(dirname+"/valid.txt", 'w') as f:
            f.writelines(f"{l}\n" for l in data[train_size:validation_end])

        with open(dirname+"/test.txt", "w") as f:
            f.writelines(f"{l}\n" for l in data[validation_end:])


if __name__ == '__main__':
    main()
