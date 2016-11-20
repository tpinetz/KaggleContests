import numpy as np

class Dataset(object):
    def __init__(self, data, labels, ids):
        self.data = data
        self.labels = labels
        self.ids = ids
    
    def size(self):
        return len(self.data)
    
    def nclasses(self):
        return len(np.unique(self.labels))

    def sample(self, sid):
        return (self.data[sid], self.labels[sid], self.ids)
