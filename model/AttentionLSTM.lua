---- to see whether we need to declare these require
-- TODO: uncomment these requires if needed
-- require 'LinearAttention'
-- require 'MVMul'

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
    -- attempt to use the above module, if doesn't work, just copy & paste, should be short
    local softmaxAttention = AttentionLSTM.softmax_attention_layer(factor_size, attentee_size)(inputs)
    local attentee = inputs[1]
    --- todo weighted application
    local weightedSumVec = nn.MVMul()({attentee, softmaxAttention})
    local proj = nn.Linear(attentee_size, output_size)(weightedSumVec)
    
    -- TODO: add log softmax here, for ClassNLL criteria
    -- if softmax, need to use CrossEntropy or KLDivergion, rumours has it that
    -- softmax + Cross Entropy is slower than log softmax + ClassNLL
    
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
    return nn.gModule(inputs, outputs)
end

---- So there are two inputs here
--  - the sequence for attentive LSTM (i.e. sentence -> sentence representation)
--  - the word/pair representation (question is, where is this representation)
--  TODO: fix the BareLSTM so that it doesn't encode the character, but the word
function AttentionLSTM.create_network(input_size, rnn_size, no_layer, dropout, 
            max_seq_length, factor_size, output_size)
    local network = {}
    -- require model.BareLSTM
    network.forward_rnn = BareLSTM.lstm(input_size, rnn_size, no_layer, dropout)
    network.backward_rnn = BareLSTM.lstm(input_size, rnn_size, no_layer, dropout)
    -- require model.LinearAttention
    network.attention = AttentionLSTM.simple_attention_classifier(factor_size, rnn_size * 2, output_size)
    -- require util.model_utils
    -- this probably create a bigger place that set the metatable for every poor
    -- model
    local params, grad_params = model_utils.combine_all_parameters(network.forward_rnn, 
            network.backward_rnn, network.attention)
            
    network.forwardClones = model_utils.clone_many_times(network.forward_rnn, max_seq_length)
    network.backwardClones = model_utils.clone_many_times(network.backward_rnn, max_seq_length)
    
    network.params = params
    network.grad_params = grad_params
    return network
end

function AttentionLSTM.prepare_training(input_size, rnn_size, no_layer, dropout, 
            max_seq_length, factor_size)
    AttentionLSTM.network = AttentionLSTM.create_network(input_size, rnn_size, 
            no_layer, dropout, max_seq_length, factor_size)
end

function AttentionLSTM.feval(x)
    local params = AttentionLSTM.network.params
    local grad_params = AttentionLSTM.network.grad_params
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

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
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        
        ---- Nghia: output lst will be {c1,h1,c2,h2..., ct,ht,prediction}
        -- init state is {c1,h1,c2,h2...,ct,ht}
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do
            -- insert(a,b) is like a.append(b) in python 
            table.insert(rnn_state[t], lst[i])
        end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        ---- Nghia: backward would automaticly adding grad to paraGrad
        -- and return {gradX, gradC1, gradH1,..., gradCt, gradHt}
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end