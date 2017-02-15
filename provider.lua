require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 34744
  local tesize = 1500
  -- load dataset
  A = torch.load('/home/siddhu95/dsg/roof_imagesaug_32/test_train_dataset/dataset_32_aug_fresh.dat','ascii')
  --A.train_X = torch.cat(A.train_X,A.test_X[{{1,1500}}],1)
  --A.train_Y = torch.cat(A.train_Y,A.test_Y[{{1,1500}}],1)
  --A.test_X = A.test_X[{{1,1500}}]
  --A.test_Y = A.test_Y[{{1,1500}}]


  self.trainData = {
     data = A.train_X:double(), 
     labels = A.train_Y:double(),
     size = function() return trsize end
  }
  local trainData = self.trainData

  self.testData = {
     data = A.test_X:double(),
     labels = A.test_Y:double(),
     size = function() return tesize end
  }
  local testData = self.testData
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     print(i)
     print(trainData:size())
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size() do
    print(i)
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
