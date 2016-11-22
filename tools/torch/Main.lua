require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'

----------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'alexnet.lua',            'Model file - must return valid network.')
cmd:option('-LR',                 0.05,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              0,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')
cmd:option('-logfile',		  'torch_log.log',	   'name of log file')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-normalization',      'simple',               'simple - whole sample, channel - by image channel, image - mean and std images')
cmd:option('-format',             'rgb',                  'rgb or yuv')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')
cmd:option('-nodeIndex',          1,       		  'Index of the current node')
cmd:option('-numNodes',           1,       		  'Number of nodes')

cmd:text('===>Misc')
cmd:option('-visualize',          0,                      'visualizing results')

opt = cmd:parse(arg or {})
opt.devid = opt.devid + 1
opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.augment then
    require 'image'
end
----------------------------------------------------------------------
-- Model + Loss:
local model
if paths.filep(opt.load) then
  pcall(require , 'cunn')
  pcall(require , 'cudnn')
  model = torch.load(opt.load)
else
  model = require(opt.network)
end

local loss = nn.CrossEntropyCriterion()--nn.ClassNLLCriterion()

-- classes
local data = require 'Data'
local classes = data.Classes

----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local AllowVarBatch = not opt.constBatchSize


----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log('./' .. opt.logfile, opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local TensorType = types[opt.type] or 'torch.FloatTensor'

if opt.type == 'cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    local cudnnAvailable = pcall(require , 'cudnn')
    if cudnnAvailable then
      model = cudnn.convert(model, cudnn)
    end
elseif opt.type == 'cl' then
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.devid)
end

model:type(TensorType)
loss = loss:type(TensorType)

---Support for multiple GPUs - currently data parallel scheme
print("opt.nGPU: " .. opt.nGPU)
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)

Nesterov = false
if string.find(opt.network, "resnet") then
    print("Use Nestrerov")
    Nesterov = true
end 
------------------Optimization Configuration--------------------------
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    dampening = 0,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay,
    nesterov = Nesterov 
}
----------------------------------------------------------------------

local function SampleImages(images,labels)
    if not opt.augment then
        return images,labels
    else

        local sampled_imgs = images:clone()
        for i=1,images:size(1) do
            local sz = math.random(9) - 1
            local hflip = math.random(2)==1

            local startx = math.random(sz)
            local starty = math.random(sz)
            local img = images[i]:narrow(2,starty,32-sz):narrow(3,startx,32-sz)
            if hflip then
                img = image.hflip(img)
            end
            img = image.scale(img,32,32)
            sampled_imgs[i]:copy(img)
        end
        return sampled_imgs,labels
    end
end


------------------------------
local function Forward(Data, train)


  local MiniBatch = DataProvider.Container{
    Name = 'GPU_Batch',
    MaxNumItems = opt.batchSize,
    Source = Data,
    ExtractFunction = SampleImages,
    TensorType = TensorType
  }

  local yt = MiniBatch.Labels
  local x = MiniBatch.Data
  local SizeData = Data:size()
  if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end

  local NumSamples = 0
  local NumBatches = 0
  local lossVal = 0

  while NumSamples < SizeData do
    MiniBatch:getNextBatch()
    local y, currLoss
    NumSamples = NumSamples + x:size(1)
    NumBatches = NumBatches + 1

    y = model:forward(x)
--    print(y)
--    print(yt)
    currLoss = loss:forward(y,yt)
    if train then
      local function feval()
        model:zeroGradParameters()
        local dE_dy = loss:backward(y, yt)
        model:backward(x, dE_dy)
        return currLoss, Gradients
      end
      _G.optim[opt.optimization](feval, Weights, optimState)
      if opt.nGPU > 1 then
        model:syncParameters()
      end
    end

    lossVal = currLoss + lossVal

    if type(y) == 'table' then --table results - always take first prediction
      y = y[1]
    end

    confusion:batchAdd(y,yt)
    xlua.progress(NumSamples, SizeData)
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
  end
  return(lossVal/math.ceil(SizeData/opt.batchSize))
end

------------------------------
local function Train(Data)
  model:training()
  return Forward(Data, true)
end

local function Test(Data)
  model:evaluate()
  return Forward(Data, false)
end
------------------------------

local epoch = 0
print '\n==> Starting Training\n'

while epoch ~= opt.epoch do
    data.TrainData:shuffleItems()

    print('Epoch ' .. epoch)
    --Train
    confusion:zero()
    sys.tic()
    local LossTrain = Train(data.TrainData)
    eptime = sys.toc()
    confusion:updateValids()
--    torch.save(netFilename, model:clearState())
    local ErrTrain = (1-confusion.totalValid)
    if #classes <= 20 then
        print(confusion)
    end
    print('EpochInfo :' .. epoch .. ': Epoch time:' .. eptime)
    print('Training Error = ' .. ErrTrain)
    print('EpochInfo :' ..epoch ..':Training Loss = :' .. LossTrain)

    --Test
    confusion:zero()
    local LossTest = Test(data.TestData)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end

    accuracy = 1 - ErrTest
    print('EpochInfo :' ..epoch ..':Test accuracy = :' .. accuracy)
    print('Test Error = ' .. ErrTest)
    print('Test Loss = ' .. LossTest)

    Log:add{['Epoch time']= eptime, ['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
        Log:plot()
    end
    epoch = epoch + 1
end
