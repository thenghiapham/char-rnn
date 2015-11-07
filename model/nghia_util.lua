local FakeLoader = {}

function FakeLoader:next_batch()
    x = {torch.Tensor{1,2,3,5,6},torch.Tensor{2,5}}
    y = {torch.Tensor{3}}
    return {x,y}
end

local FakeBatchLoader = {}
-- TODO:
function FakeBatchLoader:next_batch()
    x = {{1,2,3,5,6},{2,5}}
    y = {3}
    return {x,y}
end

local RealLoader = {}