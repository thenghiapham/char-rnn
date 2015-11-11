require 'torch'
require 'nn'
require 'nngraph'
require 'model.LinearAttention'
require 'optim'

function buildModel()
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- factor
    table.insert(inputs, nn.Identity()()) -- attentee
    
    --local factor = input[1]
    --local attentee = input[2]
    -- no need to separate them
    local linearAttention = nn.LinearAttention(5,4)(inputs)
    local attention = nn.SoftMax()(linearAttention)
    local outputs = {}
    table.insert(outputs,attention)
    return  nn.gModule(inputs, outputs)
end

local attentionModel = buildModel()
local params, gradParams = attentionModel:getParameters()
params:uniform(-0.2, 0.2)

local factor = torch.Tensor(5)
local attentee = torch.Tensor(4,3)
local i = 0

factor:apply(function()
  i = i + .1
  return i
end)
print("factor: ")
print(factor)
i = 0

attentee:apply(function()
  i = i + .1
  return i
end)

print("attentee:")
print(attentee)


local input = {factor, attentee}
local output = torch.Tensor(3)
output[1] = 0.3
output[2] = 0.1
output[3] = 0.6

local criterion = nn.DistKLDivCriterion()

local function eval(x)
  if x ~= params then
    params:copy(x)
  end
  gradParams:zero()
  local loss = 0
  
  -- forward
  local prediction = attentionModel:forward(input)
  loss = criterion:forward(prediction, output)
  
  -- backward
  local gradPrediction = criterion:backward(prediction, output) 
  local _ = attentionModel:backward(input, gradPrediction)
  
  return loss, gradParams
end 

--local diff,dC,dC_est = optim.checkgrad(eval, params, 1e-7)
--eval(params)
--print(diff)
--print("realGrad")
--print(dC)
--
--print("numericGrad")
--print(dC_est)



local function eval_attentee(x)
  if x ~= attentee then
    attentee:copy(x)
  end
  local loss = 0
  
  -- forward
  local prediction = attentionModel:forward(input)
  loss = criterion:forward(prediction, output)
  
  -- backward
  local gradPrediction = criterion:backward(prediction, output) 
  local grad_input = attentionModel:backward(input, gradPrediction)
  
  return loss, grad_input[2]:resize(12)
end 

local flat_attentee = torch.Tensor(12)
attentee:copy(attentee)

local diff,dC,dC_est = optim.checkgrad(eval_attentee, flat_attentee, 1e-7)
print(diff)
print("realGrad")
print(dC)

print("numericGrad")
print(dC_est)




--local function eval_factor(x)
--  if x ~= factor then
--    factor:copy(x)
--  end
--  local loss = 0
--  
--  -- forward
--  local prediction = attentionModel:forward(input)
--  loss = criterion:forward(prediction, output)
--  
--  -- backward
--  local gradPrediction = criterion:backward(prediction, output) 
--  local grad_input = attentionModel:backward(input, gradPrediction)
--  print(loss)
--  return loss, grad_input[1]
--end 
--
--
--local diff,dC,dC_est = optim.checkgrad(eval_factor, factor, 1e-7)
----eval(params)
--print(diff)
--print("realGrad")
--print(dC)
--
--print("numericGrad")
--print(dC_est)