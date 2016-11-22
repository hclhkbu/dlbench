local DataProvider = require 'DataProvider'
local opt = opt or {}
local Dataset = opt.dataset or 'Cifar10'
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = os.getenv("HOME").."/data/torch/" 
local normalization = opt.normalization or 'channel'
local format = opt.format or 'rgb'
local TestData
local TrainData
local Classes

if Dataset =='Cifar100' then
    TrainData = torch.load(DataPath .. 'Cifar100/cifar100-train.t7')
    TestData = torch.load(DataPath .. 'Cifar100/cifar100-test.t7')
    TrainData.labelCoarse:add(1)
    TestData.labelCoarse:add(1)
    Classes = torch.linspace(1,100,100):storage():totable()
elseif Dataset == 'Cifar10' then
    TrainData = torch.load(DataPath .. 'cifar10/cifar10-train.t7')
    TestData = torch.load(DataPath .. 'cifar10/cifar10-test.t7')
    Classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
elseif Dataset == 'STL10' then
    TrainData = torch.load(DataPath .. 'STL10/stl10-train.t7')
    TestData = torch.load(DataPath .. 'STL10/stl10-test.t7')
    Classes = {'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'}
    TestData.label = TestData.label:add(-1):byte()
    TrainData.label = TrainData.label:add(-1):byte()
elseif Dataset == 'MNIST' then
    TrainData = torch.load(DataPath .. 'mnist/mnist-train.t7')
    TestData = torch.load(DataPath .. 'mnist/mnist-test.t7')
    Classes = {1,2,3,4,5,6,7,8,9,0}
    TestData.data = TestData.data:view(TestData.data:size(1),1,28,28)
    TrainData.data = TrainData.data:view(TrainData.data:size(1),1,28,28)
    TestData.label = TestData.labels:byte()
    TrainData.label = TrainData.labels:byte()
elseif Dataset == 'mnist' then
    local util = require 'autograd.util'
    local Dataset = require 'dataset.Dataset'
    TrainData = Dataset(DataPath .. 'mnist/train.t7', {
	    partition = opt.nodeIndex,
	    partitions = opt.numNodes,
    })
    TestData = Dataset(DataPath .. 'mnist/train.t7', {
	    partition = opt.nodeIndex,
	    partitions = opt.numNodes,
    })
    Classes = {1,2,3,4,5,6,7,8,9,0}
    
elseif Dataset == 'SVHN' then
    TrainData = torch.load(DataPath .. 'SVHN/train_32x32.t7','ascii')
    ExtraData = torch.load(DataPath .. 'SVHN/extra_32x32.t7','ascii')
    TrainData.X = torch.cat(TrainData.X, ExtraData.X,1)
    TrainData.y = torch.cat(TrainData.y[1], ExtraData.y[1],1)
    TrainData = {data = TrainData.X, label = TrainData.y}
    TrainData.label = TrainData.label:add(-1):byte()
    TrainData.X = nil
    TrainData.y = nil
    ExtraData = nil

    TestData = torch.load(DataPath .. 'SVHN/test_32x32.t7','ascii')
    TestData = {data = TestData.X, label = TestData.y[1]}
    TestData.label = TestData.label:add(-1):byte()
    Classes = {1,2,3,4,5,6,7,8,9,0}
end

TrainData.label:add(1)
TestData.label:add(1)
TrainData.data = TrainData.data:float()
TestData.data = TestData.data:float()

local TrainDataProvider = DataProvider.Container{
  Name = 'TrainingData',
  CachePrefix = nil,
  CacheFiles = false,
  Source = {TrainData.data,TrainData.label},
  MaxNumItems = 1e6,
  CopyData = false,
  TensorType = 'torch.FloatTensor',
}
local TestDataProvider = DataProvider.Container{
  Name = 'TestData',
  CachePrefix = nil,
  CacheFiles = false,
  Source = {TestData.data, TestData.label},
  MaxNumItems = 1e6,
  CopyData = false,
  TensorType = 'torch.FloatTensor',

}


--Preprocesss


if format == 'yuv' then
  require 'image'
  TrainDataProvider:apply(image.rgb2yuv)
  TestDataProvider:apply(image.rgb2yuv)
end

if Whiten then
  require 'unsup'
  local meanfile = paths.concat(PreProcDir, format .. 'imageMean.t7')
  local mean, P, invP
  local Pfile = paths.concat(PreProcDir,format .. 'P.t7')
  local invPfile = paths.concat(PreProcDir,format .. 'invP.t7')

  if (paths.filep(Pfile) and paths.filep(invPfile) and paths.filep(meanfile)) then
    P = torch.load(Pfile)
    invP = torch.load(invPfile)
    mean = torch.load(meanfile)
    TrainDataProvider.Data = unsup.zca_whiten(TrainDataProvider.Data, mean, P, invP)
  else
    TrainDataProvider.Data, mean, P, invP = unsup.zca_whiten(TrainDataProvider.Data)
    torch.save(Pfile,P)
    torch.save(invPfile,invP)
    torch.save(meanfile,mean)
  end
    TestDataProvider.Data = unsup.zca_whiten(TestDataProvider.Data, mean, P, invP)

else
  local meanfile = paths.concat(PreProcDir, format .. normalization .. 'Mean.t7')
  local stdfile = paths.concat(PreProcDir,format .. normalization .. 'Std.t7')
  local mean, std
  local loaded = false

  if paths.filep(meanfile) and paths.filep(stdfile) then
    mean = torch.load(meanfile)
    std = torch.load(stdfile)
    loaded = true
  end

  mean, std = TrainDataProvider:normalize(normalization, mean, std)
  TestDataProvider:normalize(normalization, mean, std)

  if not loaded then
    torch.save(meanfile,mean)
    torch.save(stdfile,std)
  end
end



return{
    TrainData = TrainDataProvider,
    TestData = TestDataProvider,
    Classes = Classes
}
