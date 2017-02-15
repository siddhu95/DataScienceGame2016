require 'image'
require 'nn'
require 'cudnn'
require 'cunn'
local c = require 'trepl.colorize'
local json = require 'cjson'
local utils = paths.dofile'models/utils.lua'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

if #arg < 1 then
  io.stderr:write('Usage: th example_classify.lua [MODEL] [CSVFILE] Plz use absolute path!\n')
  os.exit(1)
end

if not paths.filep(arg[1]) then
  io.stderr:write('Model file not found: ' .. f .. '\n')
  os.exit(1)
end

local model_path = arg[1]
local file_path = arg[2]

-- ###########################################################CSV READING UTILITIES###############################

---------------------------------------------------------------------
local function split(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end
---------------------------------------------------------------------
function csvfile_read(path, sep, tonum)
    tonum = tonum or true
    sep = sep or ','
    local csvFile = {}
    local file = assert(io.open(path, "r"))
    for line in file:lines() do
        fields = split(line, sep)
        if tonum then -- convert numeric fields to numbers
            for i=1,#fields do
                fields[i] = tonumber(fields[i]) or fields[i]
            end
        end
        table.insert(csvFile, fields)
    end
    file:close()
    return csvFile
end

---------------------------------------------------------------------
function csvfile_write(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    for i=1,#data do
        for j=1,#data[i] do
            if j>1 then file:write(sep) end
            file:write(data[i][j])
        end
        file:write('\n')
    end
    file:close()
end
--#######################################################################################


-- loads the normalization parameters


local function dirLookup(dir)
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   for file in p:lines() do                         --Loop through all files
       print(file)       
   end
end

-- require 'provider'
-- local provider = torch.load 'provider.t7'

-- local function normalize(final_testdata)

--   local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
--   for i = 1,final_testdata:size(1) do
--     --print(i)
--     xlua.progress(i, final_testdata:size(1))
--      -- rgb -> yuv
--      local rgb = final_testdata[i]
--      local yuv = image.rgb2yuv(rgb)
--      -- normalize y locally:
--      yuv[{1}] = normalization(yuv[{{1}}])
--      final_testdata[i] = yuv
--   end

--   local mean_u = provider.trainData.mean_u
--   local std_u = provider.trainData.std_u

--   local mean_v = provider.trainData.mean_v
--   local std_v = provider.trainData.std_v

--   -- normalize u globally:
--   final_testdata:select(2,2):add(-mean_u)
--   final_testdata:select(2,2):div(std_u)
--   -- normalize v globally:
--   final_testdata:select(2,3):add(-mean_v)
--   final_testdata:select(2,3):div(std_v)

--   return final_testdata
-- end

--###########################################

opt = {
  dataset = '../cifar.torch/provider.t7',
  save = 'logs',
  batchSize = 16,
  learningRate = 0.1,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",
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

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = torch.load(model_path):cuda()
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

--model = model:double()
model:add(nn.SoftMax():cuda())
model:evaluate()
print('Model done!',model)

-- model definition should set numInputDims
-- hacking around it for the moment
--local view = model:findModules('nn.View')
--if #view > 0 then
--  view[1].numInputDims = 3
--end

local cls = {'1', '2', '3', '4'}
if file_exists('norm_unlabel_data.t7') then
  print("normed read")
  D = torch.load('norm_unlabel_data.t7','ascii')
  D.testData = D.testData:cuda()
else
  D = torch.load('/home/siddhu/cifar.torch/test_pokharna.t7','ascii')
  D.testData = D.testData:double()
  D.testData = normalize(D.testData)
  torch.save('norm_test_pokharna.t7',D,'ascii')
  print('Successfully saved')
end

local ops = torch.Tensor(D.testData:size(1)):cuda()
local pr = torch.Tensor(D.testData:size(1),4):cuda()
print(ops)

print(c.blue '==>'.." testing begins...")
local bs = 100
local output = torch.Tensor(D.testData:size(1),4)
for i=1,D.testData:size(1),bs do
  if(i+bs-1)>D.testData:size(1) then
    be = D.testData:size(1)-i+1
  else
    be = bs
  end
  local bu = model:forward(D.testData:narrow(1,i,be))
  max,index = torch.max(bu,2)
  print(i)
  ops[{{i,i+be-1}}] = index
  pr[{{i,i+be-1}}] = bu
end

print("writing to file")
local m = csvfile_read(file_path) -- read file csv1.txt to matrix m                

for i=1,ops:size(1) do
  m[i+1][2] = ops[i]
  m[i+1][3] = pr[i][1]
  m[i+1][4] = pr[i][2]
  m[i+1][5] = pr[i][3]
  m[i+1][6] = pr[i][4]
end
csvfile_write('./unlabel_prediction_run1_retrain_model1.csv', m)       -- write matrix to file csv2.txt

