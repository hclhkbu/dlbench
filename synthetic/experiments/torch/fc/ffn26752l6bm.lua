require 'sys';
require 'bit';
require 'optim';
require 'cutorch'
local opts = require 'opts'
local opt = opts.parse(arg)

if opt.deviceId >= 0 then
    require 'cunn';
    require 'cudnn';
    cutorch.setDevice(opt.deviceId+1)
else
    require 'nn'
end
print 'fc'
torch.setdefaulttensortype('torch.FloatTensor')

-- local steps = 1000 -- number of runs
local steps = opt.nIterations*opt.nEpochs-- nb of steps in loop to average perf

local Linear = nn.Linear
local Transfer = nn.Sigmoid
local isize = 26752 
local hsize = 2048
local osize = 26752 

-- Network definition
local mlp = nn.Sequential()
mlp:add(Linear(isize,hsize)):add(Transfer(true)) -- hidden layer 1
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 2
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 3
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 4
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 5 
if opt.deviceId >= 0 then
    mlp:add(Linear(hsize,osize)):add(cudnn.LogSoftMax()) -- output layer
else
    mlp:add(Linear(hsize,osize)):add(nn.LogSoftMax()) -- output layer
end

-- Fake data
local bsize = opt.batchSize
local inputCPU = torch.randn(bsize,isize)
local input = nil 
local target = nil
if opt.deviceId >= 0 then
    --input = torch.CudaTensor(inputCPU:size())
    input = torch.Tensor(inputCPU:size()):float():cuda()
    target = torch.IntTensor(bsize):random(1,bsize):cuda()
else
    input = torch.Tensor(inputCPU:size()):float()
    target = torch.IntTensor(bsize):random(1,bsize)
end

-- for k=0,2 do
nGPU = 1 -- bit.lshift(1,k)

local model = nil
-- if nGPU > 1 then
--     model = nn.DataParallelTable(1)
--     for i=1,nGPU do
--         cutorch.setDevice(i)
--         model:add(mlp:clone():cuda(), i)
--     end
-- else
if opt.deviceId >= 0 then
    cutorch.setDevice(opt.deviceId+1)
    -- else
    model = mlp:cuda()
else
    model = mlp:float()
end
-- end

-- optimizer declarations
local criterion
if opt.deviceId >= 0 then
    criterion = nn.ClassNLLCriterion():cuda()
else
    criterion = nn.ClassNLLCriterion()
end
local parameters, gradParameters = model:getParameters()
local optimState = { learningRate = 0.01 }

collectgarbage()
sys.tic()
for t = 1, steps do
    input:copy(inputCPU) -- transfer data to GPU memory
    feval = function(x)
        model:zeroGradParameters()
        local output = model:forward(input)
        local err = criterion:forward(output, target)
        local gradOutput = criterion:backward(output, target)
        local gradInput = model:backward(input, gradOutput)
        return err, gradParameters
    end
    optim.sgd(feval, parameters, optimState)

    -- DataParallelTable's syncParameters
    model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
    cutorch.synchronize()
end
local elapsed = sys.toc()

    print(string.format("%d GPUs: %0.0f samples per sec", nGPU, steps * bsize / elapsed))
    print(string.format(" | Epoch: [][]    Time %10.6f", elapsed/steps))
-- end

