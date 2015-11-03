local Attention, parent = torch.class('nn.Attention', 'nn.Module')

----
-- Now I assume input is a table, including
--   - 1 vector (size k)
--   - 1 matrix (size m, l) where m is the size of each individual small attentee
--     vectors and l is the number of these vectors
function Attention:__init(attenteeSize, factorSize)
   parent.__init(self)
   self.gradInput = {torch.Tensor(), Torch.Tensor()}
   
   self.weightFactor = torch.Tensor(factorSize)
   self.weightAttentee = torch.Tensor(attenteeSize)
   self.bias = torch.Tensor(1)
   self.gradWeightFactor = torch.Tensor(factorSize)
   self.gradWeightAttentee = torch.Tensor(attenteeSize)
   self.gradBias = torch.Tensor(1)
   
   self:reset()
end

function Attention:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      self.weightFactor:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
      self.weightAttentee:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
      self.bias[1] = torch.uniform(-stdv, stdv)
      
   else
      self.weightFactor:uniform(-stdv, stdv)
      self.weightAttentee:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function Attention:parameters()
   return {self.weightFactor, self.weightAttentee, self.bias}, 
          {self.gradWeightFactor, self.gradWeightAttentee, self.gradBias}
end

function Attention:updateOutput(input)
   local factor = input[1]
   local attentee = input[2]
   if attentee:dim() == 2 and factor:dim() == 1 then
      local nframe = attentee:size(2)
      local nElement = self.output:nElement()
      self.output:resize(1, nframe)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      local factorShare = self.bias[1] + self.weightFactor:dot(factor)
      self.output:addmv(1, attentee.t(), self.weightAttentee) -- remove t() change dimension of attentee
      self.output:add(1, factorShare) -- add all element 
   else
      error('input must be a table of a vector and a matrix')
   end
   return self.output
end



function Linear:updateGradInput(input, gradOutput)
   local factor = input[1]
   local attentee = input[2]
   
   if self.gradInput then
      local gradFactor = self.gradInput[1]
      local gradAttentee = self.gradInput[2]
      local factorElement = gradFactor:nElement()
      local attenteeElement = gradAttentee:nElement()
      self.gradInput[1]:resizeAs(factor)
      self.gradInput[2]:resizeAs(attentee)
      if self.gradInput[2]:nElement() ~= attenteeElement then
         self.gradInput[1]:zero()
         self.gradInput[2]:zero()
      end
      self.gradAttentee:addr(1, gradOutput, self.weightAttentee) -- switch if change dimension of attentee
      self.gradFactor:add(self.weightFactor:t(), gradOutput:sum())
      
      return self.gradInput
   end
end


function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local factor = input[1]
   local attentee = input[2]
   self.gradBias:add(gradOutput:sum())
   
   self.gradWeightAttentee:addmv(scale, attentee, gradOutput)
   self.gradWeightFactor:add(scale * gradOutput:sum(), input)
end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters


function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end