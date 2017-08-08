
%@Rahaf Aljundi 2016
%trained a one layer autoecnoder on a given dataset to be used in the
%expert gate.
function [net, opts, imdb, info] = cnn_autoencoder_layer_relusig(varargin)

run('../matlab/vl_setupnn');
opts = getAutoencoderOpts;
opts = vl_argparse(opts, varargin) ;

net  = get_onelayer_autoencoder(opts);



if exist(opts.imdbPath, 'file')
    
    imdb=load(opts.imdbPath);
    if(isfield(imdb,'imdb'))
        imdb=imdb.imdb;
    end

else
    
    imdb = load(opts.imdbPath);
    

    if ~exist(opts.expDir, 'dir')
        
        mkdir(opts.expDir);
        
    end
    
    save(opts.imdbPath, 'imdb');
    
end
%--------------------------------------------------------------------------
%                                                   create a validation set
%--------------------------------------------------------------------------
if(opts.useValidation)
    sets=unique(imdb.images.set);
    if(numel(sets)==2)
        
        test_set=find(imdb.images.set~=1);
        imdb.images.set(test_set)=3;
        training_inds=find(imdb.images.set==1);
        training_size=numel(training_inds);
        %create validation inds
        val_inds= randi(training_size,floor(training_size/10),1);
        imdb.images.set(training_inds(val_inds))=2;
        
    end
else
    test_set=find(imdb.images.set~=1);
    imdb.images.set(test_set)=2;
end

[net, info] = cnn_train_adagrad_oneLayer(net,[], imdb, @(imdb, batch) getAutoencoderBatch( imdb, batch), opts);



net.layers{end} = struct('name', 'data_hat_sigmoid', ...
    'type', 'sigmoid'         );

net.layers{end + 1} = struct('type', 'sigmoidcrossentropyloss');

end

% -------------------------------------------------------------------------
% creat one layer autoencoder
function net = get_onelayer_autoencoder(opts)
% -------------------------------------------------------------------------
if (~isempty(opts.initial_encoder))
    load(opts.initial_encoder);
else
    
    
  
    net.layers{1} = struct('biases'             , zeros(1, opts.code_size, 'single')             , ...
        'biasesLearningRate' , 1                                  , ...
        'biasesWeightDecay'  , 0                                  , ...
        'filters'            , sparse_initialization([1 1 opts.input_size opts.code_size]), ...
        'filtersLearningRate', 1                                  , ...
        'filtersWeightDecay' , 1                                  , ...
        'name'               , 'code'                             , ...
        'pad'                , [0 0 0 0]                          , ...
        'stride'             , [1 1]                              , ...
        'type'               , 'conv'                             );
    %
    net.layers{2} = struct('name', 'encoder_1_relu', ...
        'type', 'relu'          );
    % Layer 2
    
    net.layers{3} = struct('biases'             , zeros(1, opts.input_size, 'single')              , ...
        'biasesLearningRate' , 1                                    , ...
        'biasesWeightDecay'  , 0                                    , ...
        'filters'            , sparse_initialization([1 1 opts.code_size opts.input_size]), ...
        'filtersLearningRate', 1                                    , ...
        'filtersWeightDecay' , 1                                    , ...
        'name'               , 'data_hat'                           , ...
        'pad'                , [0 0 0 0]                            , ...
        'stride'             , [1 1]                                , ...
        'type'               , 'conv'                               );
    
   
    
    net.layers{4} = struct('name', 'decoder_1_sigmoid', ...
        'type', 'sigmoid'          );
    %loss
    net.layers{5} = struct('type', 'crossentropyloss');
end


end

% -------------------------------------------------------------------------
function filters = sparse_initialization(d)
% -------------------------------------------------------------------------

filters = zeros(d, 'single');

for index = 1 : d(4)
    
    p = randperm(d(3), 15);
    
    filters(1, 1, p, index) = randn(1, 1, 15, 1);
    
end

end

% -------------------------------------------------------------------------

function opts=getAutoencoderOpts()
%please replace these by options pointing at your own parametrs
opts.useValidation=true;
opts.imdbPath= './data/scences/encoder_input_scenes_imdb.mat';
opts.expDir= './Scenes/autoencoder/onelayer_direct_input_encodernorm/';
opts.code_size=500;
opts.input_size=43264;
opts.batchSize= 12;
opts.initial_encoder=[];
opts.errorType       = 'euclideanloss';
opts.display         = 1;
opts.delta           = 1e-8;
opts.continue        = false;
opts.learningRate    = 1e-2;
opts.numEpochs       = 100; 
opts.plotDiagnostics = false;
opts.prefetch        = false;
opts.snapshot        = 1;
opts.sync            = true;
opts.test_interval   = 1;
opts.train           = [];
opts.useGpu          = true;
opts.val             = [];
opts.weightDecay     = 5e-4;
end

% -------------------------------------------------------------------------
function [input, labels] = getAutoencoderBatch(imdb, batch)
% -------------------------------------------------------------------------
%the imdb contains the features extracted form conv5 of alexnet trained on
%Imagenet and flattened. You could replace it by any features of your choice or even
%plain images. The  mean and standard deviation here are of Imagenet conv5 features as a
%general moments approximation. You could replace them by any other.
input=[];
im = imdb.images.data(1,batch) ;

for(i=1:numel(im))
    tem=load(im{i});
    input=cat(4,input,tem.input);
end

 load('imagenet_mean');
input= bsxfun(@minus,input,imagenet_mean);
 load('imagenet_std');
 input = bsxfun(@rdivide, input,imagenet_std);


input=sigmoid(input);
labels = input;

end

