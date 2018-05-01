
filename = 'Datasets/mnist_train.csv'
trainDataset = pd.read_csv(filename,skipinitialspace=True)

filename = 'Datasets/mnist_test.csv'
testDataset = pd.read_csv(filename,skipinitialspace=True)

array = trainDataset.values
XTrain = array[:,1:785]
YTrain = array[:,0]

array = testDataset.values
XTest = array[:,1:785]
YTest = array[:,0]