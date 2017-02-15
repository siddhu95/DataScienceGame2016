npy4th = require 'npy4th'
dir = '/home/siddhu95/dsg/roof_imagesaug_32/test_train_dataset/'
Xtrain = npy4th.loadnpy(dir..'train_X.npy')
Xtest = npy4th.loadnpy(dir..'test_X.npy')
Ytrain = npy4th.loadnpy(dir..'train_Y.npy')
Ytest = npy4th.loadnpy(dir..'test_Y.npy')
dataset_32_aug_fresh = {
	train_X = Xtrain,
	train_Y = Ytrain,
	test_X = Xtest,
	test_Y = Ytest
}
torch.save(dir..'dataset_32_aug_fresh.dat',dataset_32_aug_fresh,'ascii')
