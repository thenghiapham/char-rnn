require 'torch'
local tensor_utils = {}

function tensor_utils.merge(vector_list1, vector_list2)
    ---- here we assume #vector_list1 == #vector_list2
    -- we concat
    if vector_list1[1]:dim() == 1 then
        -- make it seq x rnn like the 3D case?
        local matrix1 = torch.cat(vector_list1,2)
        local matrix2 = torch.cat(vector_list2,2)
        local result = torch.cat(matrix1, matrix2, 1)
        return result
    else
        local new_list = {}
        local batch_size = vector_list1[1]:size(1)
        local seq_size = #vector_list1
        local attentee_size = (vector_list1[1]:size(2)) * 2
        local list_length = #vector_list1
        for t = 1,(list_length * 2) do
            if (t % 2 == 1) then
                new_list[t] = vector_list1[(t + 1) / 2]
            else
                new_list[t] = vector_list2[t / 2]
            end
        end
        
        local result = torch.cat(new_list,2)
        return result:resize(batch_size, seq_size, attentee_size)
    end
end

function tensor_utils.cut_vectors(matrix)
    local vector_list1 = {}
    local vector_list2 = {}
    if matrix:dim() == 2 then 
        local matrix_t = matrix:t()
        
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
    else
        local batch_size = matrix:size(1)
        local seq_size = matrix:size(2)
        local length = matrix:size(3)
        local pos_cut = length / 2
        local vector2_length = length - pos_cut
        for i = 1,seq_size do
            local vector1 = torch.Tensor(batch_size, pos_cut)
            vector1:copy(matrix:sub(1,batch_size,i,i, 1, pos_cut))
            table.insert(vector_list1, vector1)
            local vector2 = torch.Tensor(batch_size, pos_cut)
            vector2:copy(matrix:sub(1,batch_size,i,i, pos_cut + 1,length))
            table.insert(vector_list2, vector2)
        end
        
    end
    return vector_list1, vector_list2
end


function tensor_utils.extract_last_index(mat, i,j)
    if mat:dim() == 1 then
        return mat:sub(i,j)
    elseif mat:dim() == 2 then
        local fist_size = mat:size(1)
        local result = mat:sub(1,fist_size, i, j)
        return result
    elseif mat:dim() == 3 then
        local fist_size = mat:size(1)
        local second_size = mat:size(2)
        return mat:sub(1,fist_size, 2, second_size, i, j)
    else
        error("Don't want to deal with this")
    end
end
--local a = torch.rand(6,4)
--print(a)
--local b,c = tensor_utils.cut_vectors(a)
--local d = tensor_utils.merge(b,c)
--print(d)

return tensor_utils