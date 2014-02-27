function [model, output] = trainCRBM(data, params, oldModel)
% TRAINCRBM  Trains a convolutional restricted Boltzmann machine 
%   with the specified parameters.
%
%   [model output] = TRAINCRBM(data, params, oldModel)
%
%   data should be a structure, containing:
%       data.x      The input images / pooling states of the previous layer
%                   of CRBM. This matrix is 4-D the first three dimensions
%                   define an image (coloum-stored with a color channel),
%                   and the last dimension indexes through the batch of
%                   images. I.e. the four dimensions are: height, width,
%                   channels (1 for grayscale, 3 for RGB), and number of
%                   images.
%
%   Written by: Peng Qi, Sep 27, 2012
%   Last Updated: Feb 8, 2014
%   Version: 0.3 alpha

if params.verbose > 0,
    fprintf('Starting training CRBM with the following parameters:\n');
    disp(params);
    fprintf('Initializing parameters...');
end

useCuda = params.useCuda;

if isfield(params, 'method'),
    if strcmp(params.method, 'CD'),
        method = 1; % Contrastive Divergence
    elseif strcmp(params.method, 'PCD'),
        method = 2; % Persistent Contrastive Divergence
    end
else
    method = 1;     % use Contrastive Divergence as default
end

%% initialization
N = size(data.x, 4); % Number of data points
Nfilters = params.nmap; % Number of filters
%Wfilter = params.szFilter; % Width of filter
filterWidth = params.filterWidth;
filterHeight = params.filterHeight;
p = params.szPool; % Size of pooling
H = size(data.x, 1); % Height of images
W = size(data.x, 2); % Width of images
colors = size(data.x, 3); % Number of colors
Hhidden = H - filterHeight + 1; % Height of hidden layer
Whidden = W - filterWidth + 1; % Width of hidden layer
Hpool = floor(Hhidden / p); % Height of pool layer
Wpool = floor(Whidden / p); % Width of pool layer
param_iter = params.iter; % Number of training iterations
param_szBatch = params.szBatch; % Number of examples in each batch
output_enabled = nargout > 1;

%vmasNfilters = conve(ones(nh), ones(m), useCuda);

% Initial hidden bias? WTC: parameterize
hinit = 0;
if params.sparseness > 0,
    hinit = -.1;
end

% Initialize model
if exist('oldModel','var') && ~isempty(oldModel),
    % Set model to old model
    model = oldModel;
    % WTC: 0.01 below should be paramaterized. Also, should make gaussian distribution of weights?
    if (~isfield(model,'W')), 
        model.W = 0.01 * randn(filterHeight, filterWidth, colors, Nfilters);
    else
        if (size(model.W) ~= [filterHeight filterWidth colors Nfilters]), error('Incompatible input model.'); end
    end
    if (~isfield(model,'vbias')), model.vbias = zeros(1, colors);end
    if (~isfield(model,'hbias')), model.hbias = ones(1, Nfilters) * hinit;end
    if (~isfield(model,'sigma')),
        if (params.sparseness > 0)
            % What?
            model.sigma = 0.1;
        else
            model.sigma = 1;    
        end
    end
else
    %WTC: remove repeated code from above
    model.W = 0.01 * randn(filterHeight, filterWidth, colors, Nfilters);
    model.vbias = zeros(1, colors);
    model.hbias = ones(1, Nfilters) * hinit;
    if (params.sparseness > 0)
        model.sigma = 0.1;
    else
        model.sigma = 1;    
    end
end


% Change vars
dW = 0;
dvbias = 0;
dhbias = 0;

% Momentum variables
pW = params.pW;
pvbias = params.pvbias;
phbias = params.phbias;

% WTC: why?
if output_enabled,
    output.x = zeros(Hpool, Wpool, Nfilters, N);
end

% Total number of batches
total_batches = floor(N / param_szBatch);


if params.verbose > 0,
    fprintf('Completed.\n');
end

% WAT
hidq = params.sparseness;
lambdaq = 0.9;

if ~isfield(model,'iter')
    model.iter = 0;
end

% Whiten data if set to
if (params.whitenData),
    try
        load(sprintf('whitM_%d', params.szFilter));
    catch e,
        if (params.verbose > 1), fprintf('\nComputing whitening matrix...');end
        compWhitMatrix(data.x, params.szFilter);
        load(sprintf('whitM_%d', params.szFilter));
        if (params.verbose > 1), fprintf('Completed.\n');end
    end
    if (params.verbose > 0), fprintf('Whitening data...'); end
    data.x = whiten_data(data.x, whM, useCuda);
    if (params.verbose > 0), fprintf('Completed.\n'); end
end

if method == 2,
    phantom = randn(H, W, colors, N);
end

% Iterate through data
for iter = model.iter+1:param_iter,
    
    % shuffle data
    batch_idx = randperm(N);
    
    if params.verbose > 0,
        fprintf('Iteration %d\n', iter);
        if params.verbose > 1,
            fprintf('Batch progress (%d total): ', total_batches);
        end
    end
    

    % What?
    hidact = zeros(1, Nfilters);
    errsum = 0;
    
    % What?
    if (iter > 5),
        params.pW = .9;
        params.pvbias = 0;
        params.phbias = 0;
    end
    
    for batch = 1:total_batches,
        % Select data for this batch
        % WTC: Does creating a new var slow this code down?
        batchdata = data.x(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
            batch * param_szBatch));

        % I don't understand persistent
        if method == 2,
            phantomdata = phantom(:,:,:,((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch));
        end

        % Why not originally zeros?
        recon = batchdata;
        
        %% positive phase

        %% hidden update
        
        model_W = model.W;
        model_hbias = model.hbias;
        model_vbias = model.vbias;
        

        % Pre-sigmoid activation of hidden layer?
        poshidacts = convs(recon, model_W, useCuda);

        % hidden probs (cant be, not 0-1 (values in hundereds)), pool probs, (?) hiddenstates
        [poshidprobs, pospoolprobs, poshidstates] = poolHidden(poshidacts / model.sigma, model_hbias / model.sigma, p, useCuda);
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output_x = pospoolprobs;
        end
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output.x(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
            batch * param_szBatch)) = output_x;
        end
        
        %% negative phase
        
        %% reconstruct data from hidden variables

        if method == 1,
            recon = conve(poshidstates, model_W, useCuda);
        elseif method == 2,
            recon = phantomdata;
        end
        
        % Add visible bias
        recon = bsxfun(@plus, recon, reshape(model_vbias, [1 1 colors]));

        % Add noise if sparse? What?
        if (params.sparseness > 0),
            recon = recon + model.sigma * randn(size(recon));
        end
        
        %% mean field hidden update
        
        neghidacts = convs(recon, model_W, useCuda);
        neghidprobs = poolHidden(neghidacts / model.sigma, model_hbias / model.sigma, p, useCuda);
            
        if (params.verbose > 1),
            fprintf('.');
            err = batchdata - recon;
            errsum = errsum + sum(err(:).^2);
            if (params.verbose > 4),
                %% visualize data, reconstruction, and filters (still experimental)
                figure(1);
                for i = 1:16,subplot(4,8,i+16);imagesc(model.W(:,:,:,i));axis image off;end;colormap gray;drawnow;
                subplot(2,2,1);imagesc(batchdata(:,:,1));colormap gray;axis off;title('data');
                subplot(2,2,2);imagesc(recon(:,:,1));colormap gray;axis off;title('reconstruction');
                drawnow;
            end
        end
        
        %% contrast divergence update on params
        
        if (params.sparseness > 0),
            hidact = hidact + reshape(sum(sum(sum(pospoolprobs, 4), 2), 1), [1 Nfilters]);
        else
            dhbias = phbias * dhbias + ...
                reshape((sum(sum(sum(poshidprobs, 4), 2), 1) - sum(sum(sum(neghidprobs, 4), 2), 1))...
                / Whidden / Hhidden / param_szBatch, [1 Nfilters]);
        end
        
        dvbias = pvbias * dvbias + ...
            reshape((sum(sum(sum(batchdata, 4), 2), 1) - sum(sum(sum(recon, 4), 2), 1))...
            / H / W / param_szBatch, [1 colors]);
        ddw = convs4(batchdata(filterHeight:H-filterHeight+1,filterWidth:W-filterWidth+1,:,:), poshidprobs(filterHeight:H-filterHeight+1,filterWidth:W-filterWidth+1,:,:), useCuda) ...
            - convs4(    recon(filterHeight:H-filterHeight+1,filterWidth:W-filterWidth+1,:,:), neghidprobs(filterHeight:H-filterHeight+1,filterWidth:W-filterWidth+1,:,:), useCuda);
        dW = pW * dW + ddw / (Hhidden - 2 * filterHeight + 2) / (Whidden - 2 * filterWidth + 2) / param_szBatch;
        
        model.vbias = model.vbias + params.epsvbias * dvbias;
        if params.sparseness <= 0,
            model.hbias = model.hbias + params.epshbias * dhbias; 
        end
        model.W = model.W + params.epsW * (dW  - params.decayw * model.W);
        
        %% experimental code for saving debugging info for mex implementations
%         save dbgInfo model poshidacts poshidprobs poshidstates recon neghidacts neghidprobs model_W
%         if any(isnan(model.W(:))) || any(isnan(poshidacts(:))) || any(isnan(poshidprobs(:))) || any(isnan(poshidstates(:))) ...
%                 || any(isnan(recon(:))) || any(isnan(neghidacts(:))) || any(isnan(neghidprobs(:))),
%             return;
%         end
        if method == 2,
            phantom(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch)) = conve(neghidprobs, model_W, useCuda);
        end
    end
    
    if (params.verbose > 1),
        fprintf('\n\terror:%f', errsum);
    end
    
    if params.sparseness > 0,
        hidact = hidact / Hhidden / Whidden / N;
        hidq = hidq * lambdaq + hidact * (1 - lambdaq);
        dhbias = phbias * dhbias + ((params.sparseness) - (hidq));
        model.hbias = model.hbias + params.epshbias * dhbias;
        if params.verbose > 0,
            if (params.verbose > 1),
                fprintf('\tsigma:%f', model.sigma);
            end
            fprintf('\n\tsparseness: %f\thidbias: %f\n', sum(hidact) / Nfilters, sum(model.hbias) / Nfilters);
        end
        if (model.sigma > 0.01),
            model.sigma = model.sigma * 0.95;
        end
    end
    
    if ~rem(iter, params.saveInterv),
        if (params.verbose > 3),
            figure(1);
            for i = 1:16,subplot(4,8,i+16);imagesc(model.W(:,:,:,i));axis image off;end;colormap gray;drawnow;
            subplot(2,2,1);imagesc(batchdata(:,:,1));colormap gray;axis off;title('data');
            subplot(2,2,2);imagesc(recon(:,:,1));colormap gray;axis off;title('reconstruction');
            drawnow;
        end
        if output_enabled,
            model.iter = iter;
            save(params.saveName, 'model', 'output', 'iter');
            if params.verbose > 1,  
                fprintf('Model and output saved at iteration %d\n', iter);
            end
        else 
            model.iter = iter;
            save(params.saveName, 'model', 'iter');
            if params.verbose > 1,
                fprintf('Model saved at iteration %d\n', iter);
            end
        end
    end

end
end
