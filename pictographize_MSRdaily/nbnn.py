import numpy as np

# from sklearn.neighbors import KDTree
from preproc import preproc_train_data, prepare_test_data


# train_sample = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
# tree = KDTree(train_sample, 1)
#
# test_data = [(2.1, 3.1), (2, 4.5)]
# print(tree.query(test_data, 2, return_distance=True))

class NBNN:
    def __init__(self, trainning_data, label):
        # self.kdtree = KDTree(trainning_data, leaf_size=10)
        self.label = label
        self.trainning_data = trainning_data
        # a map from label to the array index
        self.mLabel2arr = {}
        # a map from array index to the label
        self.marr2Label = {}
        ind = 0
        for k in label:
            if k not in self.mLabel2arr:
                self.mLabel2arr[k] = ind
                self.marr2Label[ind] = k
                ind += 1
        self.totol_types_num = ind # = len(self.mLabel2arr)

    def query(self, test_data):
        distances = np.zeros((test_data.shape[0], self.totol_types_num))
        for s in range(len(test_data)):
            for t in range(len(self.trainning_data)):
                try:
                    distances[s, self.mLabel2arr[self.label[t]]] += np.linalg.norm(test_data[s] - self.trainning_data[t])
                except:
                    print(test_data[s].shape)
                    print(self.trainning_data[t].shape)
                    exit(-1)
        # TODO 详细查查argmin
        mdis = np.argmin(distances, axis=1)
        result = [self.marr2Label[mdis[k]] for k in range(test_data.shape[0])]
        return result

def test(as_id):
    trainning_data, label, min_frames_num = preproc_train_data(as_id)
    test_data, groundtruth = prepare_test_data(as_id, min_frames_num)
    nbnn = NBNN(trainning_data, label)
    predict_result = nbnn.query(test_data)
    print(predict_result)
    print(groundtruth)
    diff = np.array(predict_result)-np.array(groundtruth)
    print(diff)
    print((len(diff)-np.count_nonzero(diff))/len(diff))

test(2)
