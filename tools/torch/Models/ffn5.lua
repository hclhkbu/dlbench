require 'cunn'
require 'cudnn'
local model = nn.Sequential()

model:add(nn.View(784))
model:add(nn.Linear(784, 2048))
model:add(nn.Sigmoid())
model:add(nn.Linear(2048, 4096))
model:add(nn.Sigmoid())
model:add(nn.Linear(4096, 1024))
model:add(nn.Sigmoid())
model:add(nn.Linear(1024, 10))
model:add(nn.LogSoftMax())

local method = 'xavier'
local model_new = require('./weight-init')(model, method)

return model_new
