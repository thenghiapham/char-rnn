local FlatWeight, parent = torch.class('nn.FlatWeight', 'nn.Module')

----
-- Now I assume input is a table, including
--   - 1 vector (size k)
--   - 1 matrix (size m, l) where m is the size of each individual small attentee
--     vectors and l is the number of these vectors
function FlatWeight:__init(size)
   parent.__init(self)

   self.gradInput = torch.Tensor()
   
   self.weight = torch.Tensor(1,size)
   self.gradWeight = torch.Tensor(1,size)

   self:reset()
end

function FlatWeight:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      self.weight:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
   
   return self
end

function FlatWeight:updateOutput(input)
   -- TODO:
   -- understand why they don't have this guy here 
   -- if don't understand, uncomment the line below
   self.output:zero()
   self.output:resize(1,input:size(2))
   if input:dim() == 2 then
      self.output:mm(self.weight, input) 
   else
      error('input must be a matrix')
   end
   return self.output:resize(input:size(2))
end



function FlatWeight:updateGradInput(input, gradOutput)
   local tmp = torch.Tensor(gradOutput:size(1))
   tmp:copy(gradOutput)
   tmp:resize(1,gradOutput:size(1))
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      self.gradInput:addmm(1,self.weight:t(), tmp) -- switch if change dimension of attentee
      return self.gradInput
   end
end


function FlatWeight:accGradParameters(input, gradOutput, scale)
   local tmp = torch.Tensor(gradOutput:size(1))
   tmp:copy(gradOutput)
   tmp:resize(1,gradOutput:size(1))
   scale = scale or 1
   -- TODO: fix this, don't rotate input?
   self.gradWeight:addmm(scale, tmp, input:t())
end

-- we do not need to accumulate parameters when sharing
FlatWeight.sharedAccUpdateGradParameters = FlatWeight.accUpdateGradParameters


function FlatWeight:__tostring__()
  return torch.type(self) ..
      string.format('(%d and k -> 1 x k)', self.weight:size(2))
end