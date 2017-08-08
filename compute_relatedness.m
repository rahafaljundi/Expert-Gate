%compute the relatedness between two tasks
function  [relatedness]=compute_relatedness(varargin)
run('../matlab/vl_setupnn');
opts=getDefaultOpts;
opts = vl_argparse(opts, varargin) ;
first_task_encoder=load(opts.first_task_encoder_path);
if(isfield(first_task_encoder,'net'))
    first_task_encoder=first_task_encoder.net;
end
%-----------loading theimage mean and standard deviation--------------
load('imagenet_mean.mat');
load('imagenet_std.mat');
%load first and second task autoencoders
first_task_encoder.normalization.averageImage=imagenet_mean;
first_task_encoder.normalization.deviation=imagenet_std;
first_task_encoder.layers{end} = struct('type', 'euclideanloss');

second_task_encoder=load(opts.second_task_encoder_path);
if(isfield(second_task_encoder,'net'))
    second_task_encoder=second_task_encoder.net;
end
second_task_encoder.layers{end} = struct('type', 'euclideanloss');

%load the data set (usually it is the new task dataset in this context the
%second
second_imdb = load(opts.test_dataset_path) ;

acc_t2=0;
%get images and labels
second_validation_inds=find(second_imdb.images.set~=1);

%------------------------------------------------------------

%------------------------------------------------------------
for i=1:numel(second_validation_inds)
    % run the CNN
    
    
    
    x=load(second_imdb.images.data{1,second_validation_inds(i)}) ;
    
    %----------------normalizing------------------------------------------------------------
    
    input_first = bsxfun(@minus, x.input, first_task_encoder.normalization.averageImage);
    
    input_first = bsxfun(@rdivide, input_first, first_task_encoder.normalization.deviation);
    %---------------------------------------------------------------------------------------
    input_first=sigmoid(input_first);
    input_first = gpuArray(input_first);
    first_task_encoder.layers{end}.class = input_first;
    
    res = vl_simplenn_autoencoder(first_task_encoder, input_first, [], [], 'disableDropout', true);
    first_reconstruction_err = gather(res(end).x);
    %-----------------
    %second encoder path
    %-----------------
    %pass through the input net
    
    %----------------normalizing------------------------------------------------------------
    
    input_second = bsxfun(@minus, x.input, first_task_encoder.normalization.averageImage);
    
    input_second = bsxfun(@rdivide, input_second, first_task_encoder.normalization.deviation);
    %----------------------------------------------------------------------------------------
    
    input_second=sigmoid(input_second);
    %input_second=input;
    input_second = gpuArray(input_second);
    second_task_encoder.layers{end}.class = input_second;
    
    res = vl_simplenn_autoencoder(second_task_encoder, input_second, [], [], 'disableDropout', true);
    second_reconstruction_err = gather(res(end).x);
    %-----------------
    
    if (second_reconstruction_err<(first_reconstruction_err))
        acc_t2=acc_t2+1;
        
    end
    all_second_err(i)=second_reconstruction_err;
    all_first_err(i)=first_reconstruction_err;
end


%the accuracy on the second task
acc_t2=acc_t2*100/numel(second_validation_inds);
%compute relatedness
second_avg_err=mean(all_second_err);
first_avg_err=mean(all_first_err);
confusion=(first_avg_err-second_avg_err)*100/second_avg_err;
relatedness=100-confusion;

save(strcat('results/',opts.output,'_accuracy'),'acc_t2','second_avg_err','first_avg_err','relatedness','confusion');

end
function opts=getDefaultOpts
%Sample options
%input_net_path,first_task_encoder_path,second_task_encoder_path,test_dataset_path,input_size,output
opts.input_net_path='/users/visics/raljundi/Code/MyOwnCode/MatConvNet/data/mnist-bnorm/new/net-epoch-100';
%opts.first_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/SVHN/mnist/onelayer/autoencoder/net-epoch-650';
opts.first_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/image-net/autoencoder/layerafternet/net-epoch-6.mat';

opts.first_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/image-net/autoencoder/final/net-epoch-1.mat';

%opts.second_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/SVHN/onelayer/autoencoder/net-epoch-690';
opts.second_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/SVHN/autoencoder/onelayerSmallCode/net-epoch-1000';

opts.dataDir= fullfile('/users/visics/raljundi/Code/MyOwnCode/MatConvNet/data','mnist-bnorm');
opts.input_size=3072;
opts.test_dataset_path=fullfile(opts.dataDir, 'mnist&SVHN_imdb.mat');
opts.output='SVHN_autoencoder_Mnist_gate';
opts.output_layer_id=6;
load('mnist_autoencoder_mean.mat');
opts.data_mean=data_mean;
opts.border = [29, 29] ;
opts.imageSize = [227, 227] ;
%opts.averageImage=0;
opts.keepAspect=true;
opts.numAugments=1;
opts.direct_input=0;
end