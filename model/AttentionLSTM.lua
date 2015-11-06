---- to see whether we need to declare these require
-- TODO: uncomment these requires if needed
-- require 'LinearAttention'
-- require 'MVMul'

---- here everything is word embedding and not char embedding
-- therefore, the input is not a one hot vector but the word embedding


local AttentionLSTM = {}

function AttentionLSTM.softmax_attention_layer(factor_size, attentee_size)
    -- let try to use this guy as well as a component to build another module
    local inputs={}
    local outputs={}
    table.insert(inputs, nn.Identity()()) -- factor
    table.insert(inputs, nn.Identity()()) -- attentee
    local linearAttention = nn.AttentionLSTM(factor_size, attentee_size)(inputs)
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
    local softmaxAttention = AttentionLSTM.softmax_attention_layer(factor_size, attentee_size)({factor, attentee})
    
    --- todo weighted application
    local weightedSumVec = nn.MVMul()({attentee, softmaxAttention})
    local proj = nn.Linear(attentee_size, output_size)(weightedSumVec)
    
    -- TODO: add log softmax here, for ClassNLL criteria
    -- if softmax, need to use CrossEntropy or KLDivergion, rumours has it that
    -- softmax + Cross Entropy is slower than log softmax + ClassNLL
    
    local logsoft = nn.LogSoftMax()(proj)
    local err = nn.ClassNLLCriterion()({logsoft, y})
    table.insert(outputs, err)
    return nn.gModule(inputs, outputs)
end

---- So there are two inputs here
--  - the sequence for attentive LSTM (i.e. sentence -> sentence representation)
--  - the word/pair representation (question is, where is this representation)
--  TODO: fix the BareLSTM so that it doesn't encode the character, but the word
function AttentionLSTM.create_network(vocab_size, rnn_size, no_layer, dropout, 
            max_seq_length, factor_size, output_size)
    local network = {}
    local base_embedding_layer = nn.LookupTable(vocab_size, rnn_size)
    -- require model.BareLSTM
    local right_rnn = BareLSTM.lstm(rnn_size, no_layer, dropout)
    local left_rnn = BareLSTM.lstm(rnn_size, no_layer, dropout)
    -- require model.LinearAttention
    network.classifier = AttentionLSTM.simple_attention_classifier(factor_size, rnn_size * 2, output_size)
    -- require util.model_utils
    -- this probably create a bigger place that set the metatable for every poor
    -- model
    local params, grad_params = model_utils.combine_all_parameters(base_embedding_layer, right_rnn, 
            left_rnn, network.attention)
    
    -- TODO: add the number of thing here
    network.embedding_clones = model_utils.clone_many_times(base_embedding_layer, max_seq_length + 2)        
    network.right_clones = model_utils.clone_many_times(right_rnn, max_seq_length)
    network.left_clones = model_utils.clone_many_times(left_rnn, max_seq_length)
    
    network.params = params
    network.grad_params = grad_params
    return network
end

function AttentionLSTM.prepare_training(vocab_size, rnn_size, no_layer, dropout, 
            max_seq_length, factor_size)
    AttentionLSTM.network = AttentionLSTM.create_network(vocab_size, rnn_size, 
            no_layer, dropout, max_seq_length, factor_size)
end

function AttentionLSTM.feval(x)
    local network = AttentionLSTM.network
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
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    local sequence = {}
    local pair = {}
    local gold = {}
    
    ------------------- forward pass -------------------
    -- TODO: where the hell is this init_fstate_global
    local right_state = {[0] = init_rstate_global}
    local left_state = {[t+1] = init_lstate_global}
    local predictions = {}           
    local loss = 0
    
    local embeddings = {}
    
    -- get embedding
    -- TODO: fix this, i.e. deal with sequence
    for t=1,opt.seq_length do
        embeddings[t] = network.embedding_clones[t].forward(sequence[{{}, t}])
    end
    
    local left_t = 0
    -- forward through bidirectional lstm
    for t=1,opt.seq_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        network.embedding_clones[t]:training()
        network.right_clones[t]:training() 
        network.left_clones[t]:training()
        
        
        ---- Nghia: output lst will be {c1,h1,c2,h2..., ct,ht,prediction}
        -- init state is {c1,h1,c2,h2...,ct,ht}
        -- Don't understand this x[] crap, TODO: fix this
        local rst = network.right_clones.rnn[t]:forward{embeddings[t], unpack(right_state[t-1])}
        left_t = 1 + opt.seq_length - t
        local lst = network.left_clones.rnn[left_t]:forward{embeddings[left_t], unpack(left_state[left_t + 1])}
        
        -- since no prediction, put it straight away
        right_state[t] = rst
        left_state[left_t] = lst
    end
    
    -- need to merge the two list of tensors
    -- TODO:
    local merge_state = {} 
    
    -- only 1 prediction
    loss = network.classifier:forward({pair, merge_state, gold})
        
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local dright_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    local dleft_state = {[1] = clone_list(init_state, true)} -- true also zeros the clones
    local dembeddings = {}
    
    derr = transfer_data(torch.ones(1))
    
    -- dattention has:
    --    2 embedding error
    --    d[h_r[last_layer]], d[h_l[last_layer]] for every sequence element
    dattention = network.classifier:backward({pair, merge_state, gold}, derr)
    
    d_merge_sate = dattention[2]
    -- TODO: fix this
    -- initiziatl dright_state with error back from attention to the last h 
    
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        ---- Nghia: backward would automaticly adding grad to paraGrad
        -- and return {gradX, gradC1, gradH1,..., gradCt, gradHt}
        local drst = clones.rnn[t]:backward({embeddings[{{}, t}], unpack(right_state[t-1])}, dright_state[t])
        left_t = 1 + opt.seq_length - t
        local dlst = clones.rnn[left_t]:backward({embeddings[{{}, left_t}], unpack(left_state[left_t+1])}, dleft_state[left_t])
        -- k = 1, x
        -- k = 2i: dc[i]
        -- k = 2i + 1: dh[i]
        -- k = 2 * layer + 1: need to add from attention
        for k,v in pairs(drst) do
            if k > 1 then 
                dembeddings[t]:add(v)
            else
                if k == 2 * no_layer + 1 then 
                    dright_state[t-1][k-1]:add(v) 
                else
                    dright_state[t-1][k-1] = v
                end
            end
        end
        
        for k,v in pairs(dlst) do
            if k > 1 then 
                dembeddings[left_t]:add(v)
            else
                if k == 2 * no_layer + 1 then 
                    drnn_state[left_t+1][k-1]:add(v) 
                else
                    drnn_state[left_t+1][k-1] = v
                end
            end
        end        
    end
    
    -- backward embedding
    -- TODO: fix this, i.e. deal with sequence
    for t=1,opt.seq_length do
        embeddings[t] = network.embedding_clones[t].backward(sequence[{{}, t}], dembeddings[t])
    end
        
    -- TODO: deal with this later
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end