require 'torch'
require 'nn'
require 'model.Attention'

local factor = torch.Tensor(5)
local attentee = torch.Tensor(4,3)
local i = 0

factor:apply(function()
  i = i + 1
  return i
end)
print("factor: ")
print(factor)
i = 0

attentee:apply(function()
  i = i + 1
  return i
end)

print("attentee:")
print(attentee)

local attentionLayer = nn.LinearAttention(5,4)
local input = {factor, attentee}

local output = attentionLayer:forward(input)
print("output:")
print(output)

i = -1
local gradOutput = output.new()
gradOutput:apply(function()
  i = i + 1
  return i
end)
print("gradOutput")
print(gradOutput)

local params, gradParams = attentionLayer:parameters()
local gradInput =attentionLayer:backward(input, gradOutput)
-- print("gradFactor")
-- print(gradInput[1])
-- print("attentee")
-- print(gradInput[2])

