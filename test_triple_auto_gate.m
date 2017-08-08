%@Rahaf Aljundi
%Demonstrates the  gate test on 3 tasks sequence. 
%pass the sample throught the gating function and then check if the
%predicted task is correct one
%standarize using imagenet statistics
%This is a sample code. Please replace the datasets and networks parts with
%your own experts and trained autoecndoers
function  [acc_all]=test_triple_auto_gate(varargin)
run('../matlab/vl_setupnn');
opts=getDefaultOpts;
opts = vl_argparse(opts, varargin) ;
%load the different parts of the netowrk
if(~opts.direct_input)
    input_net=load(opts.input_net_path);
    
    if(isfield(input_net,'net'))
        input_net=input_net.net;
    end
    input_net.layers{end}.type='softmax';
end

%initialization
imdb.images.data=[];
imdb.images.set=[];
imdb.images.labels=[];
imdb.images.domain_labels=[];
%containes the list of autoencoders that you have previously trained on
%each of the  previous task
for(i=1:numel(opts.encoders))
    encoders{i}=load(opts.encoders{i})
    if(isfield( encoders{i},'net'))
        encoders{i}=encoders{i}.net;
    end
    encoders{i}.layers{end} = struct('type', 'euclideanloss');
    %loading the datasets for each task to get the test samples
    this_imdb=load(opts.imdbs{i});
    imdb.images.data=[imdb.images.data,this_imdb.images.data];
    imdb.images.labels=[imdb.images.labels,this_imdb.images.labels];
    imdb.images.set=[imdb.images.set,this_imdb.images.set];
    imdb.images.domain_labels=[imdb.images.domain_labels,i*ones(1,numel(this_imdb.images.set))];
    acc_all{i}=0;%initialize the accuracies.
    
end



%-----------loading theimage mean and standard deviation--------------
load('imagenet_mean.mat');
load('imagenet_std.mat');
imagenet_encoder.normalization.averageImage=imagenet_mean;
imagenet_encoder.normalization.deviation=imagenet_std;



%get test images and labels

test_inds=find(imdb.images.set==3);
labels = imdb.images.labels(test_inds)';

domain_labels = imdb.images.domain_labels(test_inds)';

for i=1:numel(test_inds)
    % 
    if(~opts.direct_input)
        %if the dataset is composed of images we need to extract the
        %feature vector before passing it to the autoencoder
        if(iscell(imdb.images.data))
            
            im = imdb.images.data(1,test_inds(i)) ;
            
            batchOpts.border = opts.border ;
            batchOpts.imageSize = opts.imageSize;
            batchOpts.averageImage=opts.averageImage;
            batchOpts.keepAspect=opts.keepAspect;
            batchOpts.numAugments=opts.numAugments;
            
            image = cnn_imagenet_get_batch(input_net,im,batchOpts);
        else
            image = imdb.images.data(:, :, :, test_inds(i));
            
        end
       
        
        
        
        %get the feature vector for the test image
        origin_res = vl_simplenn_autoencoder(input_net, image) ;
        input=origin_res(opts.output_layer_id);
        %reshape%
        input=reshape(input.x,1,1,opts.input_size,[]);
    else
        load(imdb.images.data{1,test_inds(i)}) ;
    end
    %standarization
    input_first = bsxfun(@minus, input, imagenet_encoder.normalization.averageImage);
    
    input_first = bsxfun(@rdivide, input_first, imagenet_encoder.normalization.deviation);

    input_first=sigmoid(input_first);
    %pass the test sample to the gate (composd of the different autoencoders) 
    for(i_enc=1:numel(encoders))
    
        encoders{i_enc}.layers{end}.class = input_first;
        
        res = vl_simplenn_autoencoder(encoders{i_enc}, input_first, [], [], 'disableDropout', true);
        reconstruction_err{i_enc} = gather(res(end).x);
        
    end
    
    [val,enc_index]=min(cell2mat(reconstruction_err));
   
    %check if the gate correctly predicts sample task label
    if(enc_index==domain_labels(i))
        acc_all{enc_index}=acc_all{enc_index}+1;
    end
    
end
for(i=1:numel(opts.encoders))
    
    acc_all{i}= acc_all{i}*100/numel(find(domain_labels==i));
end
save(strcat('results/',opts.output,'_accuracy'),'acc_all');
end
function opts=getDefaultOpts
%input_net_path,first_task_encoder_path,second_task_encoder_path,test_dataset_path,input_size,output
opts.input_net_path='/users/visics/raljundi/Code/MyOwnCode/MatConvNet/data/mnist-bnorm/new/net-epoch-100';
%opts.first_task_encoder_path='/esat/jade/raljundi/netoutput/matconvnet/SVHN/mnist/onelayer/autoencoder/net-epoch-650';
opts.encoders{1}='/esat/jade/raljundi/netoutput/matconvnet/Flowers/autoencoder/onelayer_test_relsig_std/net-epoch-150.mat';
opts.encoders{2}='/esat/jade/raljundi/netoutput/matconvnet/CUB_Training/autoencoder/onelayer_test_relsig_std//net-epoch-150.mat';
opts.encoders{3}='/esat/jade/raljundi/netoutput/matconvnet/Scenes/autoencoder/onelayer_test_relsig_std/net-epoch-150.mat';
opts.imdbs{1}='data/flowers/encoder_input_flowers_imdb';
opts.imdbs{2}='data/CUB/encoder_input_cub_imdb';
opts.imdbs{3}='data/scences/encoder_input_scenes_imdb';
opts.output='fl_cub_sc_autoencoder_gate_acc';
opts.border = [29, 29] ;
opts.imageSize = [227, 227] ;
opts.averageImage=0;
opts.keepAspect=true;
opts.numAugments=1;
opts.direct_input=1;
end