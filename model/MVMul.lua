
local MVMul, parent = torch.class('nn.MVMul', 'nn.Module')

function MVMul:__init()
   parent.__init(self)
   self.gradInput = {}
end

function MVMul:updateOutput(input)
   self.output:resize(input[1]:size(1))
   -- mul = zeroes, addmv
   -- TODO: do the same for LinearAttention, i.e. replace 2 with 1
   self.output:mv(input[1],input[2])
--   print(self.output)
   
   return self.output
end

-- I guess this thing has divided by zero problem
--[[function MVMul:updateGradInput_efficient(input, gradOutput)
   ... copied from CMulTable
end--]] 

function MVMul:updateGradInput(input, gradOutput)
--   print(input[1])
--   print(input[2])
--   print(gradOutput)
--   error("Stop")
--   print("backward")
--   print("input")
--   print(input[1])
--   print(input[2])
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[1]:resizeAs(input[1])
   self.gradInput[2]:resizeAs(input[2])
   
   self.gradInput[1]:ger(gradOutput, input[2])
   self.gradInput[2]:mv(input[1]:t(),gradOutput)
   return self.gradInput
end

function MVMul:__tostring__()
  return torch.type(self) ..
      string.format('(m x n , n --> m)')
end