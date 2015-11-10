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
  print("prediction")
  print(prediction)
  loss = criterion:forward(prediction, output)
  
  -- backward
  local gradPrediction = criterion:backward(prediction, output) 
  print("gradPrediction")
  print(gradPrediction)
  local _ = attentionModel:backward(input, gradPrediction)
  
  return loss, gradParams
end 

local diff,dC,dC_est = optim.checkgrad(eval, params, 1e-7)
--eval(params)
print(diff)
print("realGrad")
print(dC)

print("numericGrad")
print(dC_est)