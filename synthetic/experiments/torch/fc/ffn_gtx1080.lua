require 'sys';
require 'bit';
require 'cunn';
require 'cudnn';
require 'optim';
torch.setdefaulttensortype('torch.FloatTensor')

local steps = 1000 -- number of runs

local Linear = nn.Linear
local Transfer = nn.Sigmoid
local isize = 512
local hsize = 2048
local osize = 1000

-- Network definition
local mlp = nn.Sequential()
mlp:add(Linear(isize,hsize)):add(Transfer(true)) -- hidden layer 1
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 2
mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 3
-- mlp:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 4
mlp:add(Linear(hsize,osize)):add(cudnn.LogSoftMax()) -- output layer

-- Fake data
local bsize = 1024 
local inputCPU = torch.randn(bsize,isize)
local input = torch.CudaTensor(inputCPU:size())
local target = torch.IntTensor(bsize):random(1,bsize):cuda()

-- for k=0,2 do
nGPU = 1 -- bit.lshift(1,k)

local model = nil
-- if nGPU > 1 then
--     model = nn.DataParallelTable(1)
--     for i=1,nGPU do
--         cutorch.setDevice(i)
--         model:add(mlp:clone():cuda(), i)
--     end
cutorch.setDevice(1)
-- else
model = mlp:cuda()
-- end

-- optimizer declarations
local criterion = nn.ClassNLLCriterion():cuda()
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
-- end

