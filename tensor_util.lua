require 'torch'
local tensor_utils = {}

function tensor_utils.merge(vector_list1, vector_list2)
    ---- here we assume #vector_list1 == #vector_list2
    -- we concat 
    local matrix1 = torch.cat(vector_list1,2)
    
    local matrix2 = torch.cat(vector_list2,2)
    local result = torch.cat(matrix1, matrix2, 1)
    return result
end

function tensor_utils.cut_vectors(matrix)
    
    local matrix_t = matrix:t()
    local vector_list1 = {}
    local vector_list2 = {}
    local length = matrix:size(1)
    local pos_cut = length / 2
    local vector2_length = length - pos_cut
    for i = 1,matrix:size(2) do
        local vector1 = torch.Tensor(pos_cut)
        vector1:copy(matrix:sub(1,pos_cut,i,i))
        table.insert(vector_list1, vector1)
        local vector2 = torch.Tensor(pos_cut)
        vector2:copy(matrix:sub(pos_cut + 1,length,i,i))
        table.insert(vector_list2, vector2)
    end
    return vector_list1, vector_list2
end

--local a = torch.rand(6,4)
--print(a)
--local b,c = tensor_utils.cut_vectors(a)
--local d = tensor_utils.merge(b,c)
--print(d)

return tensor_utils