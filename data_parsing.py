import os

def ontologyFromFolder(dataset, folder):



    with open(os.path.join(folder, dataset,".data.mme")) as f:
        for line in f:
            pred, rest = tuple(line.split("("))
            args = rest.split(")")[0].split()

