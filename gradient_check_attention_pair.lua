require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'util.table_io'
local FakeLoader = require 'model.loader_util'

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local function parse_opt()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a character-level language model')
  cmd:text()
  cmd:text('Options')
  -- data
  cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
  -- model params
  cmd:option('-vocab_size',4,'number of words in the vocab')
  cmd:option('-rnn_size', 3, 'size of LSTM internal state')
  cmd:option('-num_layers', 1, 'number of layers in the LSTM')
  cmd:option('-output_size', 3, 'number of layers in the LSTM')
  cmd:option('-model', 'lstm', 'lstm,gru or rnn')
    -- optimization
  cmd:option('-max_seq_length',50,'maximum sequence length')
  cmd:option('-learning_rate',2e-3,'learning rate')
  cmd:option('-learning_rate_decay',0.97,'learning rate decay')
  cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
  cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-seq_length',50,'number of timesteps to unroll for')
  ---- TODO set batch size > 1 to gradient check batch later
  -- this time 1 to check simple first
  cmd:option('-batch_size',3,'number of sequences to train on in parallel')
  cmd:option('-max_epochs',50,'number of full passes through the training data')
  cmd:option('-grad_clip',5,'clip gradients at this value')
  cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
  cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
              -- test_frac will be computed as (1 - train_frac - val_frac)
  cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
  -- bookkeeping
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
  cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
  cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
  cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
  -- GPU/CPU
  cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:text()
  
  -- parse input params
  local opt = cmd:parse(arg)
  return opt
end

local opt = parse_opt()
local loader = FakeLoader.create()
opt.loader = loader
local AttentionLSTM = require 'model.AttentionLSTM'
AttentionLSTM.prepare_training(opt)
local params = AttentionLSTM.network.params
local grad_params = AttentionLSTM.network.grad_params
print(#params)
print(#grad_params)

local initialization_file = "/home/nghia/test_attention.txt"
if not file_exists(initialization_file) then
    params:uniform(-0.2, 0.2)
    local param_table = {}
    for t = 1,(#params)[1] do
        param_table[t] = params[t]
    end
    table.save(param_table, initialization_file)
    print("save")
else
    local param_table = table.load(initialization_file)
    local saved_params = torch.Tensor(param_table)
    params:copy(saved_params)
    print("load")
end
local diff,dC,dC_est = optim.checkgrad(AttentionLSTM.feval, params, 1e-7)
--eval(params)
print(diff)
local merge = torch.cat({dC, dC_est},2)
print(merge)