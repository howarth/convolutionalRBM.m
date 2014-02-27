function params = getparams(method)
% GETPARAMS  Get default params for trainCRBM
%
%   See also TRAINCRBM
%
%   Written by: Peng Qi, Sep 27, 2012

%% Model parameters
params.nmap = 16; % number of filters
%params.szFilter = 8; % size of square filter
params.filterWidth = 8;
params.filterHeight = 7;
params.szPool = 2; % size of square max pool filter
params.method = 'CD'; % learning method

if (nargin > 0)
    if strcmp(method, 'PCD'),
        params.method = 'PCD';
    end
end

%% Learining parameters
params.epshbias = 1e-1; % learning rate for hidden bias
params.epsvbias = 1e-1; % learning rate for visible bias
params.epsW = 1e-2; % learning rate for weights
params.phbias = 0.5; % momentum of hidden bias
params.pvbias = 0.5; % momentum of visual bias
params.pW = 0.5; % momentum of W
params.decayw = .01; % exponential decay value
params.szBatch = 10; % size of training batch
params.sparseness = .02; % sparseness of hidden nodes
params.whitenData = 1; % 1 if you want to whiten data, 0 if not

%% Running parameters
params.iter = 10000; % number of learning iterations
params.verbose = 2; % verbosity level
params.mfIter = 5; % ?
params.saveInterv = 5; 
params.useCuda = 0; 
params.saveName = 'model.mat';

end
