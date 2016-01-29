
local SCRNN = {}
function SCRNN.scrnn(input_size, rnn_size, alpha, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_s[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_s = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2s = nn.Linear(input_size_L, rnn_size)(x)
    local s2s = nn.MulConstant(alpha, true)(prev_s)
    
    local ai2s = nn.MulConstant(1 - alpha)(i2s)
    local next_s = nn.CAddTable()({ai2s, s2s})

    
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local s2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, s2h, h2h})
    local next_h = nn.Sigmoid()(all_input_sums)
    
    table.insert(outputs, next_s)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  local top_s = outputs[#outputs - 1]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local sproj = nn.Linear(rnn_size, input_size)(top_s)
  local hproj = nn.Linear(rnn_size, input_size)(top_h)
  local proj = nn.CAddTable()({sproj, hproj})
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return SCRNN

