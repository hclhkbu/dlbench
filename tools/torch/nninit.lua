local nn = require 'nn'

-- Helper functions

-- Calculates fan in and fan out of module
local function calcFan(module)
  local typename = torch.type(module)
  if typename == 'nn.Linear' or
     typename == 'nn.LinearNoBias' or
     typename == 'nn.LookupTable' then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('TemporalConvolution') then
    return module.weight:size(2), module.weight:size(1)
  elseif typename:find('SpatialConvolution') or typename:find('SpatialFullConvolution') then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  elseif typename:find('VolumetricConvolution') or typename:find('VolumetricFullConvolution') then
    return module.nInputPlane * module.kT * module.kW * module.kH, module.nOutputPlane * module.kT * module.kW * module.kH
  else
    error("Unsupported module")
  end
end

-- Returns the gain or calculates if given a gain type (with optional args)
local function calcGain(gain)
  -- Return gain if a number already
  if type(gain) == 'number' then
    return gain
  end

  -- Extract gain string if table
  local args
  if type(gain) == 'table' then
    args = gain
    gain = gain[1]
  end

  -- Process gain strings with optional args
  if gain == 'linear' or gain == 'sigmoid' then
    return 1
  elseif gain == 'relu' then
    return math.sqrt(2)
  elseif gain == 'lrelu' then
    return math.sqrt(2 / (1 + math.pow(args.leakiness, 2)))
  end

  -- Return 1 by default
  return 1
end

-- init method

-- Add init to nn.Module
nn.Module.init = function(self, accessor, initialiser, ...)
  -- Extract tensor to initialise
  local tensor
  if type(accessor) == 'string' then
    tensor = self[accessor]
  elseif type(accessor) == 'table' then
    tensor = self[accessor[1]][accessor[2]]
  elseif type(accessor) == 'function' then
    tensor = accessor(self)
  else
    error("Unsupported accessor")
  end

  -- Initialise tensor (given module and options)
  initialiser(self, tensor, ...)

  -- Return module for chaining
  return self
end

-- nninit

local nninit = {}

-- Copies another tensor to the tensor to be initialised
nninit.copy = function(module, tensor, init)
  tensor:copy(init)

  return module
end

-- Fills tensor with a constant value
nninit.constant = function(module, tensor, val)
  tensor:fill(val)

  return module
end

-- Adds to current tensor with a constant value
nninit.addConstant = function(module, tensor, val)
  tensor:add(val)

  return module
end

-- Multiplies current tensor by a constant value
nninit.mulConstant = function(module, tensor, val)
  tensor:mul(val)

  return module
end

-- Fills tensor ~ N(mean, stdv)
nninit.normal = function(module, tensor, mean, stdv)
  tensor:normal(mean, stdv)

  return module
end

-- Adds to current tensor with ~ N(mean, stdv)
nninit.addNormal = function(module, tensor, mean, stdv)
  tensor:add(torch.Tensor(tensor:size()):normal(mean, stdv))

  return module
end

-- Fills tensor ~ U(a, b)
nninit.uniform = function(module, tensor, a, b)
  tensor:uniform(a, b)

  return module
end

-- Adds to current tensor with ~ U(a, b)
nninit.addUniform = function(module, tensor, a, b)
  tensor:add(torch.Tensor(tensor:size()):uniform(a, b))

  return module
end

-- Fills weights with the identity matrix (for linear layers)
-- Fills filters with the Dirac delta function (for convolutional layers)
-- TODO: Generalise for arbitrary tensors?
nninit.eye = function(module, tensor)
  if module.weight ~= tensor then
    error("nninit.eye only supports 'weight' tensor")
  end

  local typename = torch.type(module)

  if typename == 'nn.Linear' or
     typename == 'nn.LinearNoBias' or
     typename == 'nn.LookupTable' then
    local I = torch.eye(tensor:size(1), tensor:size(2))
    tensor:copy(I)
  elseif typename:find('TemporalConvolution') then
    tensor:zero()
    for i = 1, module.inputFrameSize do
      tensor[{{}, {(i-1)*module.kW + math.ceil(module.kW/2)}}]:fill(1/module.inputFrameSize)
    end
  elseif typename:find('SpatialConvolution') or typename:find('SpatialFullConvolution') then
    tensor:zero():view(module.nInputPlane, module.nOutputPlane, module.kW, module.kH)[{{}, {}, math.ceil(module.kW/2), math.ceil(module.kH/2)}]:fill(1/module.nInputPlane)
  elseif typename:find('VolumetricConvolution') or typename:find('VolumetricFullConvolution') then
    tensor:zero():view(module.nInputPlane, module.nOutputPlane, module.kT, module.kW, module.kH)[{{}, {}, math.ceil(module.kT/2), math.ceil(module.kW/2), math.ceil(module.kH/2)}]:fill(1/module.nInputPlane)
  else
    error("Unsupported module for 'eye'")
  end

  return module
end

--[[
--  Glorot, X., & Bengio, Y. (2010)
--  Understanding the difficulty of training deep feedforward neural networks
--  In International Conference on Artificial Intelligence and Statistics
--
--  Also known as Glorot initialisation
--]]
nninit.xavier = function(module, tensor, options)
  local fanIn, fanOut = calcFan(module)
  options = options or {}
  gain = calcGain(options.gain)
  dist = options.dist or 'uniform' -- Uniform by default

  local stdv = gain * math.sqrt(2 / (fanIn + fanOut))
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    tensor:uniform(-b, b)
  elseif dist == 'normal' then
    tensor:normal(0, stdv)
  end

  return module
end

--[[
--  He, K., Zhang, X., Ren, S., & Sun, J. (2015)
--  Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification
--  arXiv preprint arXiv:1502.01852
--
--  Also known as He initialisation
--]]
nninit.kaiming = function(module, tensor, options)
  local fanIn = calcFan(module)
  options = options or {}
  gain = calcGain(options.gain)
  dist = options.dist or 'normal' -- Normal by default

  local stdv = gain * math.sqrt(1 / fanIn)
  if dist == 'uniform' then
    local b = stdv * math.sqrt(3)
    tensor:uniform(-b, b)
  elseif dist == 'normal' then
    tensor:normal(0, stdv)
  end

  return module
end

--[[
--  Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013)
--  Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
--  arXiv preprint arXiv:1312.6120
--]]
nninit.orthogonal = function(module, tensor, options)
  local sizes = tensor:size()
  if #sizes < 2 then
    error("nninit.orthogonal only supports tensors with 2 or more dimensions")
  end

  -- Calculate "fan in" and "fan out" for arbitrary tensors based on module conventions
  local fanIn = sizes[2]
  local fanOut = sizes[1]
  for d = 3, #sizes do
    fanIn = fanIn * sizes[d]
  end

  options = options or {}
  gain = calcGain(options.gain)

  -- Construct random matrix
  local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)
  local U, __, V = torch.svd(randMat, 'S')

  -- Pick out orthogonal matrix
  local W
  if fanOut > fanIn then
    W = U
  else
    W = V:narrow(1, 1, fanOut)
  end
  -- Resize
  W:resize(tensor:size())
  -- Multiply by gain
  W:mul(gain)

  tensor:copy(W)

  return module
end

--[[
-- Martens, J. (2010)
-- Deep learning via Hessian-free optimization
-- In Proceedings of the 27th International Conference on Machine Learning (ICML-10)
--]]
nninit.sparse = function(module, tensor, sparsity)
  local nElements = tensor:nElement()
  local nSparseElements = math.floor(sparsity * nElements)
  local randIndices = torch.randperm(nElements):long()
  local sparseIndices = randIndices:narrow(1, 1, nSparseElements)

  -- Zero out selected indices
  tensor:view(nElements):indexFill(1, sparseIndices, 0)

  return module
end

return nninit
