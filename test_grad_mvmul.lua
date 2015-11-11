require 'torch'
require 'nn'
require 'nngraph'
require 'model.MVMul'
require 'optim'

function buildModel()
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- factor
    table.insert(inputs, nn.Identity()()) -- attentee
    
    --local factor = input[1]
    --local attentee = input[2]
    -- no need to separate them
    local linearAttention = nn.MVMul()(inputs)
    local attention = nn.SoftMax()(linearAttention)
    local outputs = {}
    table.insert(outputs,attention)
    return  nn.gModule(inputs, outputs)
end

local attentionModel = buildModel()

local vector = torch.Tensor(3)
local matrix = torch.Tensor(4,3)
local i = 0

matrix:apply(function()
  i = i + .1
  return i
end)
print("matrix: ")
print(matrix)
i = 0

vector:apply(function()
  i = i + .1
  return i
end)

print("vector:")
print(vector)

local flat_matrix = torch.Tensor(12)
flat_matrix:copy(matrix) 

local input = {matrix, vector}
local output = torch.Tensor(4)
output[1] = 0.3
output[2] = 0.1
output[3] = 0.5
output[4] = 0.1

local criterion = nn.DistKLDivCriterion()
local iter = 1
local function eval_matrix(x)
  if x ~= matrix then
    matrix:copy(x)
  end
  local loss = 0
  iter = iter + 1
  -- forward
  local prediction = attentionModel:forward(input)
  loss = criterion:forward(prediction, output)
  -- backward
  local gradPrediction = criterion:backward(prediction, output) 
  local gradInput = attentionModel:backward(input, gradPrediction)
  return loss, gradInput[1]
end 

local function eval_vector(x)
  if x ~= vector then
    vector:copy(x)
  end
  local loss = 0
  
  -- forward
  local prediction = attentionModel:forward(input)
  -- print("prediction")
  -- print(prediction)
  loss = criterion:forward(prediction, output)
  
  -- backward
  local gradPrediction = criterion:backward(prediction, output) 
  -- print("gradPrediction")
  -- print(gradPrediction)
  local gradInput = attentionModel:backward(input, gradPrediction)
  
  return loss, gradInput[2]
end 

local diff,dC,dC_est = optim.checkgrad(eval_matrix, flat_matrix, 1e-7)
--eval(params)
print(diff)
print("realGrad mat")
print(dC)
local grad_matrix = torch.Tensor(4,3)
grad_matrix:copy(dC_est)
print("numericGrad mat")
print(grad_matrix)


diff,dC,dC_est = optim.checkgrad(eval_vector, vector, 1e-7)
--eval(params)
print(diff)
print("realGrad vec")

print(dC)

print("numericGrad vec")
print(dC_est)