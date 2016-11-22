require 'cunn'
require 'cudnn'
local model = nn.Sequential() 

-- Stage 1:
model:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2 ))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialCrossMapLRN(3, 0.00005, 0.75, 1))

-- Stage 2:
model:add(cudnn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.SpatialCrossMapLRN(3, 0.00005, 0.75, 1))

-- Stage 3:
model:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.SpatialCrossMapLRN(3, 0.00005, 0.75, 1))

model:add(nn.View(64*3*3))
model:add(nn.Linear(64*3*3,10))
model:add(nn.LogSoftMax())
return model
