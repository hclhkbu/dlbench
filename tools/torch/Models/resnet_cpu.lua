--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgement in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

require './residual-layers'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
local nninit = require 'nninit'


-- Residual network.
-- Input: 3x32x32
local N = 9 --opt.Nsize
input = nn.Identity()()
------> 3, 32,32
model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
:init('weight', nninit.kaiming, {gain = 'relu'})
:init('bias', nninit.constant, 0)(input)
model = cudnn.SpatialBatchNormalization(16)(model)
model = cudnn.ReLU(true)(model)
------> 16, 32,32   First Group
for i=1,N do   model = addResidualLayer2(model, 16)   end
------> 32, 16,16   Second Group
model = addResidualLayer2(model, 16, 32, 2)
for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
------> 64, 8,8     Third Group
model = addResidualLayer2(model, 32, 64, 2)
for i=1,N-1 do   model = addResidualLayer2(model, 64)   end
------> 10, 8,8     Pooling, Linear, Softmax
model = nn.SpatialAveragePooling(8,8)(model)
model = nn.Reshape(64)(model)
model = nn.Linear(64, 10)(model)
model = nn.LogSoftMax()(model)

model = nn.gModule({input}, {model})
model:cuda()
print(model)
--print(#model:forward(torch.randn(512, 3, 32,32):cuda()))
return model
