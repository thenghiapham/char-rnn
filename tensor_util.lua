local tensor_utils

function tensor_utils.merge(vector_list1, vector_list2)
    ---- here we assume #vector_list1 == #vector_list2
    -- we concat 
    local matrix1 = torch.cat(vector_list1,2)
    local matrix2 = torch.cat(vector_list2,2)
    local result = torch.cat(matrix1, matrix2, 1)
    return result
end

function tensor_utils.vector_cut(matrix)
    
    local matrix_t = matrix:t()
    local vector_list1 = {}
    local vector_list2 = {}
    local length = matrix:size(1)
    local pos_cut = length / 2
    local vector2_length = length - pos_cut
    for i = 1,matrix:size(2) do
        table.insert(vector_list1, matrix:sub(1,pos_cut,i,i):resize(pos_cut))
        table.insert(vector_list2, matrix:sub(pos_cut + 1,length,i,i):resize(vector2_length))
    end
end