local FakeLoader = {}
FakeLoader.__index = FakeLoader

function FakeLoader:next_batch()
    x = {torch.Tensor{1,2,3,4,3},torch.Tensor{2,4}}
    --x = {torch.Tensor{1,3},torch.Tensor{2,4}}
    y = torch.Tensor{2}
    return {x,y}
end



function FakeLoader.create()
    local self = {}
    setmetatable(self, FakeLoader)
    
    return self
end




local FakeBatchLoader = {}
-- TODO:
function FakeBatchLoader:next_batch()
    x = {{1,2,3,5,6},{2,5}}
    y = {3}
    return {x,y}
end



local RealLoader = {}

return FakeLoader