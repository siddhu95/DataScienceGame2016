-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
local utils = paths.dofile'models/utils.lua'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'
dofile('../cifar.torch/provider.lua')

opt = {
  dataset = './norm_unlabelled_all.t7',
  save = 'logs',
  batchSize = 16,
  learningRate = 0.01,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.7,
  epoch_step = "5",
  max_epoch = 300,
  model = 'wide-resnet',
  optimMethod = 'sgd',
  init_value = 10,
  depth = 28,
  shortcutType = 'A',
  nesterov = true,
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',
  cudnn_fastest = true,
  cudnn_deterministic = false,
  optnet_optimize = true,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 10,
  nGPU = 1,
}
opt = xlua.envparams(opt)

confusionT = optim.ConfusionMatrix(4)
opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset, 'ascii')
opt.num_classes = provider.labels:max()

local provider_val = torch.load('../cifar.torch/provider.t7')
-- provider_val.trainData = nil
--provider.trainData.data = provider.trainData.data:narrow(1,1,1000)
--provider.trainData.labels = provider.trainData.labels:narrow(1,1,1000)


print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = torch.load('logs/run1_retrain/model_1.t7'):cuda()
do
   function nn.Copy.updateGradInput() end
   local function add(flag, module) if flag then model:add(module) end end
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())

   cudnn.convert(net, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end

   print(net)
   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end

   model:add(utils.makeDataParallelTable(net, opt.nGPU))
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

local parameters,gradParameters = model:getParameters()

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()

-- a-la autograd
local f = function(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   confusionT:batchAdd(model.output, targets)
   --return table.unpack({loss,output})
   return loss
end

print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)


function train()
  model:training()

  local targets = torch.CudaTensor(math.floor(0.7*opt.batchSize) + math.floor(0.3*opt.batchSize))
  local indices = torch.randperm(provider.data:size(1)):long():split(math.floor(0.3*opt.batchSize))
  local indices1 = torch.randperm(provider_val.trainData.data:size(1)):long():split(math.floor(0.7*opt.batchSize))
  -- remove last element so that all minibatches have equal size
  indices[#indices] = nil

  local loss = 0
  local ite = 0
  local tic = torch.tic()
  for t,v in ipairs(indices) do
    ite=ite+1
    if ite>250 then
      break
    end
    xlua.progress(t,#indices)
    local inputs = provider.data:index(1,v):cat(provider_val.trainData.data:index(1,indices1[t]),1)
    local indt = provider.labels
    indt = indt:view(indt:size(1),1):index(1,v)
    local indt2 = provider_val.trainData.labels
    indt2 = indt2:view(indt2:size(1),1):index(1,indices1[t]):long()
    targets:copy(indt:cat(indt2,1))

    optim[opt.optimMethod](function(x)
      if x ~= parameters then parameters:copy(x) end
      model:zeroGradParameters()
      loss = loss + f(inputs, targets)
      return f,gradParameters
    end, parameters, optimState)
  end

  confusionT:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(confusionT.totalValid * 100, torch.toc(tic)))
  print(confusionT)
  confusionT:zero()
  return loss / #indices
end


function test()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider_val.testData.data:split(opt.batchSize,1)
  local labels_split = provider_val.testData.labels:split(opt.batchSize,1)

  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v), labels_split[i])
  end

  confusion:updateValids()
  print(confusion)
  return confusion.totalValid * 100
end


for epoch=1,opt.max_epoch do
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  -- drop learning rate and reset momentum vector
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
     torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
    opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
  end

  local function t(f) local s = torch.Timer(); return f(), s:time().real end

  local loss, train_time = t(train)
  local test_acc, test_time = t(test)

  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }
  torch.save(opt.save..'/run3/model_unlabelled_'..epoch..'.t7', net:clearState())
end

torch.save(opt.save..'/model.t7', net:clearState())