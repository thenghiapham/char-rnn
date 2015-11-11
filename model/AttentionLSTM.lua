---- to see whether we need to declare these require
-- TODO: uncomment these requires if needed
-- require 'LinearAttention'
-- require 'MVMul'

---- here everything is word embedding and not char embedding
-- therefore, the input is not a one hot vector but the word embedding

local BareLSTM = require 'model.BareLSTM'
local model_utils = require 'util.model_utils'
require 'model.LinearAttention'
require 'model.MVMul'
require 'util.misc'
local tensor_utils = require 'tensor_util'

local AttentionLSTM = {}

function AttentionLSTM.softmax_attention_layer(factor_size, attentee_size)
    -- let try to use this guy as well as a component to build another module
    local inputs={}
    local outputs={}
    table.insert(inputs, nn.Identity()()) -- factor
    table.insert(inputs, nn.Identity()()) -- attentee
    local linearAttention = nn.LinearAttention(factor_size, attentee_size)(inputs)
    local softmaxAttention = nn.SoftMax()(linearAttention)
    table.insert(outputs, softmaxAttention)
    return nn.gModule(inputs, outputs)
end

-- in this module the matrix that decide the attention weights is also the input that
-- the weight is decided upon
-- let create another module for the other guy where the two guys are different
function AttentionLSTM.simple_attention_classifier(factor_size, attentee_size, output_size)
    local inputs={}
    local outputs={}
    
    table.insert(inputs, nn.Identity()()) -- factor
    table.insert(inputs, nn.Identity()()) -- attentee
    table.insert(inputs, nn.Identity()()) -- gold
    
    
    local factor = inputs[1]
    local attentee = inputs[2]
    local gold = inputs[3]
    
    -- attempt to use the above module, if doesn't work, just copy & paste, should be short
    local linearAttention = nn.LinearAttention(factor_size, attentee_size)({factor, attentee})
    local softmaxAttention = nn.SoftMax()(linearAttention)
    -- local softmaxAttention = AttentionLSTM.softmax_attention_layer(factor_size, attentee_size)({factor, attentee})
    
    --- todo weighted application
    local weightedSumVec = nn.MVMul()({attentee, softmaxAttention})
    local proj = nn.Linear(attentee_size, output_size)(weightedSumVec)
    
    -- TODO: add log softmax here, for ClassNLL criteria
    -- if softmax, need to use CrossEntropy or KLDivergion, rumours has it that
    -- softmax + Cross Entropy is slower than log softmax + ClassNLL
    
    local logsoft = nn.LogSoftMax()(proj)
    local err = nn.ClassNLLCriterion()({logsoft, gold})
    table.insert(outputs, err)
    return nn.gModule(inputs, outputs)
end

---- So there are two inputs here
--  - the sequence for attentive LSTM (i.e. sentence -> sentence representation)
--  - the word/pair representation (question is, where is this representation)
--  TODO: fix the BareLSTM so that it doesn't encode the character, but the word
function AttentionLSTM.create_network(opt)
    local network = {}
    network.base_embedding_layer = nn.LookupTable(opt.vocab_size, opt.rnn_size)
    -- require model.BareLSTM
    network.right_rnn = BareLSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout)
    network.left_rnn = BareLSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout)
    -- require model.LinearAttention
    local factor_size = opt.rnn_size * 2
    network.classifier = AttentionLSTM.simple_attention_classifier(factor_size, opt.rnn_size * 2, opt.output_size)
    -- require util.model_utils
    -- this probably create a bigger place that set the metatable for every poor
    -- model
    
    if opt.gpuid >= 0 and opt.opencl == 0 then
        for k,v in pairs(network) do
            v:cuda()
        end
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then
        for k,v in pairs(network) do v:cl() end
    end
    
    local params, grad_params = model_utils.combine_all_parameters(network.base_embedding_layer, network.right_rnn, 
            network.left_rnn, network.classifier)
    
    -- TODO: add the number of thing here
    network.embedding_clones = model_utils.clone_many_times(network.base_embedding_layer, opt.max_seq_length + 2)
    network.right_clones = model_utils.clone_many_times(network.right_rnn, opt.max_seq_length)
    network.left_clones = model_utils.clone_many_times(network.left_rnn, opt.max_seq_length)
    
    network.params = params
    network.grad_params = grad_params
    return network
end

function AttentionLSTM.prepare_training(opt)
    
    ---- factor size should be rnn_size * (1,2)?
    AttentionLSTM.network = AttentionLSTM.create_network(opt)
    ---- TODO: create init_state_global, and init_state here
    -- transfer here?
    local init_state_global = {}
    ---- TODO: deal with batch_size later,
    -- for now, batch_size = 1
    -- to deal with batch size, need to deal with the attention
    local init_state = {}
    for L=1,opt.num_layers do
        -- local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        local h_init = torch.zeros(opt.rnn_size)
        if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end
    local init_state_global = clone_list(init_state)
    
    AttentionLSTM.network.init_state = init_state
    AttentionLSTM.network.init_state_global = init_state_global
    AttentionLSTM.network.opt = opt
end

----
-- TODO: deal with all the global varible
--    loader

function AttentionLSTM.feval(x)
    
    local network = AttentionLSTM.network
    local opt = network.opt
    local num_layers = opt.num_layers
    
    if not network then
        error("Network not initialized")
    end
    local params = AttentionLSTM.network.params
    local grad_params = AttentionLSTM.network.grad_params
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    -- TODO: change the input/output data
    ------------------ get minibatch -------------------
    -- Nghia: integers can't be cuda() but can be cl()?
    local derr = torch.ones(1.0)
    local x, y = unpack(opt.loader:next_batch())
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        derr = derr:cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
        derr = derr:cl()
    end
    local sequence = x[1]
    local pair = x[2]
    local gold = y
    
    
    local sequence_length = (#sequence)[1]
    --print(gold)
    ------------------- FORWARD PASS  -------------------
    -- TODO: set this init_state_global to contains only zeros
    -- (since we're not doing language modeling
    -- no  memory from before
    
    local right_state = {[0] = network.init_state_global}
    local left_state = {[sequence_length+1] = network.init_state_global}
    local predictions = {}           
    local loss = 0
--    error("Stop here")
    local embeddings = {}
    
    -- get embedding
    -- TODO: fix this, i.e. deal with sequence
    local attentee = {{},{}}
    for t=1,sequence_length do
        embeddings[t] = network.embedding_clones[t]:forward(sequence:sub(t,t)):resize(opt.rnn_size)
    end
    -- TODO: change if necessary
    local pair_embedding1 = network.embedding_clones[sequence_length + 1]:forward(pair:sub(1,1)):resize(opt.rnn_size)
    local pair_embedding2 = network.embedding_clones[sequence_length + 2]:forward(pair:sub(2,2)):resize(opt.rnn_size)
    
    local factor = torch.cat(pair_embedding1, pair_embedding2)
    ---- embedding returns 2 dimensional vector
    -- factor:resize(opt.rnn_size * 2)
    
    local left_t = 0
    -- forward through bidirectional lstm
    
    for t=1,sequence_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        network.embedding_clones[t]:training()
        network.right_clones[t]:training() 
        network.left_clones[t]:training()
        
        ---- Nghia: output lst will be {c1,h1,c2,h2..., ct,ht,prediction}
        -- init state is {c1,h1,c2,h2...,ct,ht}
        local rst = network.right_clones[t]:forward{embeddings[t], unpack(right_state[t-1])}
        left_t = 1 + sequence_length- t
        local lst = network.left_clones[left_t]:forward{embeddings[left_t], unpack(left_state[left_t + 1])}
        
        -- since no prediction, put it straight away
        right_state[t] = rst
        left_state[left_t] = lst
        
        -- put the last hidden into context vectors
        attentee[1][t] = rst[#rst]
        attentee[2][left_t] = lst[#lst]
    end
    
    -- need to merge the two list of tensors
    -- TODO:
    local merge_state = tensor_utils.merge(unpack(attentee))
--    print(merge_state)
    -- only 1 prediction
--    print("Parameters")
--    print(network.classifier:parameters()[2])
--    print("No parameters")
    loss = network.classifier:forward({factor, merge_state, gold})[1]
--    print("loss")
--    print(loss)
--    error("stop here")
        
    ------------------ BACKWARD PASS -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local dright_state = {[sequence_length] = clone_list(network.init_state, true)} -- true also zeros the clones
    local dleft_state = {[1] = clone_list(network.init_state, true)} -- true also zeros the clones
    
    
    local dembeddings = {}
    for t = 1,sequence_length + 2 do
        dembeddings[t] = torch.zeros(opt.rnn_size)
    end
    
    
    -- dattention has:
    --    2 embedding error
    --    d[h_r[last_layer]], d[h_l[last_layer]] for every sequence element
    local dattention = network.classifier:backward({pair, merge_state, gold}, derr)
--    print("d_attention")
--    print(dattention[2]:sum())
--    print(dattention[1])
    local d_merge_state = dattention[2]
    -- TODO: fix this
    local d_attentee1, d_attentee2 = tensor_utils.cut_vectors(d_merge_state)
    -- initiziatl dright_state with error back from attention to the last h 
    
    
    for t=sequence_length,1,-1 do
        -- backprop through loss, and softmax/linear
        ---- Nghia: backward would automaticly adding grad to paraGrad
        -- and return {gradX, gradC1, gradH1,..., gradCt, gradHt}
        
        dright_state[t][2 * num_layers]:add(d_attentee1[t])
        local drst = network.right_clones[t]:backward({embeddings[t], unpack(right_state[t-1])}, dright_state[t])
        
        left_t = 1 + sequence_length - t
        dleft_state[left_t][2 * num_layers]:add(d_attentee2[left_t])
        local dlst = network.left_clones[left_t]:backward({embeddings[left_t], unpack(left_state[left_t+1])}, dleft_state[left_t])
        -- k = 1, x
        -- k = 2i: dc[i]
        -- k = 2i + 1: dh[i]
        -- k = 2 * layer + 1: need to add from attention in the next iteration
        dright_state[t-1] = {}
        for k,v in pairs(drst) do
            if k > 1 then 
                dright_state[t-1][k-1] = v
            else 
                dembeddings[t]:add(v)
            end
        end
        
        dleft_state[left_t+1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then 
                dleft_state[left_t+1][k-1] = v
            else
                dembeddings[left_t]:add(v)
            end
        end        
    end
    
    -- backward embedding
    -- TODO: fix this, i.e. deal with sequence
    
    for t=1,sequence_length do
--        print(dembeddings[t])
        network.embedding_clones[t]:backward(sequence:sub(t,t), dembeddings[t])
    end
    
    local dfactor = dattention[1]
--    print("dfactor")
--    print(dfactor)
--    error("stop here")
    network.embedding_clones[sequence_length + 1]:backward(pair:sub(1,1), dfactor:sub(1,opt.rnn_size))
    network.embedding_clones[sequence_length + 2]:backward(pair:sub(2,2), dfactor:sub(opt.rnn_size + 1, 2 * opt.rnn_size))
        
    -- TODO: backward from pair???
    
    ------------------------ misc ----------------------
    -- clip gradient element-wise
    --grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

return AttentionLSTM