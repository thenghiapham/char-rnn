local LinearAttention, parent = torch.class('nn.LinearAttention', 'nn.Module')

----
-- Now I assume input is a table, including
--   - 1 vector (size k)
--   - 1 matrix (size m, l) where m is the size of each individual small attentee
--     vectors and l is the number of these vectors
function LinearAttention:__init(factorSize, attenteeSize)
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   
   self.weightFactor = torch.Tensor(factorSize)
   self.weightAttentee = torch.Tensor(attenteeSize)
   self.bias = torch.Tensor(1)
   self.gradWeightFactor = torch.Tensor(factorSize)
   self.gradWeightAttentee = torch.Tensor(attenteeSize)
   self.gradBias = torch.Tensor(1)
   
   self:reset()
end

function LinearAttention:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weightAttentee:size(1))
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
   -- TODO: remove the fill commands
   self.weightFactor:fill(1)
   self.weightAttentee:fill(2)
   self.bias:fill(3)
   
   return self
end

function LinearAttention:parameters()
   return {self.weightFactor, self.weightAttentee, self.bias}, 
          {self.gradWeightFactor, self.gradWeightAttentee, self.gradBias}
end

function LinearAttention:updateOutput(input)
   -- TODO:
   -- understand why they don't have this guy here 
   -- self.output:zero()
   local factor = input[1]
   local attentee = input[2]
   if attentee:dim() == 2 and factor:dim() == 1 then
      local nframe = attentee:size(2)
      local nElement = self.output:nElement()
      self.output:resize(nframe)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      local factorShare = self.bias[1] + self.weightFactor:dot(factor)
      self.output:addmv(1, attentee:t(), self.weightAttentee) -- remove t() change dimension of attentee
      self.output:add(factorShare) -- add all element 
   else
      error('input must be a table of a vector and a matrix')
   end
   return self.output
end



function LinearAttention:updateGradInput(input, gradOutput)
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
      gradAttentee:addr(1, self.weightAttentee, gradOutput) -- switch if change dimension of attentee
      gradFactor:add(self.weightFactor, gradOutput:sum())
      
      return self.gradInput
   end
end


function LinearAttention:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local factor = input[1]
   local attentee = input[2]
   self.gradBias:add(gradOutput:sum())
   
   self.gradWeightAttentee:addmv(scale, attentee, gradOutput)
   self.gradWeightFactor:add(scale * gradOutput:sum(), factor)
end

-- we do not need to accumulate parameters when sharing
LinearAttention.sharedAccUpdateGradParameters = LinearAttention.accUpdateGradParameters


function LinearAttention:__tostring__()
  return torch.type(self) ..
      string.format('(%d and %d -> attention)', self.weightFactor:size(1), self.weightAttentee:size(1))
end