%@Rahaf Aljundi
%demonstrates expert gate system on a sequence of six tasks
%activates one task only
%nets_path: path to each task expert network
%imdbs: path to each task dataset
%encoders_imdbs_path: path to each of the autoencoders trained on each
%previous task
function [accuracy] = test_expret_networks_autoendoer_hard_gate(nets_path,imdbs,encoders_path,output,encoders_imdbs_path)
% setup MatConvNet
cd ..
run  matlab/vl_setupnn
if(~exist('th','var'))
    th=0.1;
end
T=2;
% load the pre-trained CNN
for(i=1:numel(nets_path))
    load(nets_path{i}) ;
    
    net.layers{end}.type='softmax';
    
    nets{i}=net;
    clear net
    
    
end
%load the data set
if(~exist('encoders_imdbs_path','var'))
    input_net=load('/users/visics/raljundi/Code/MyOwnCode/MatConvNet/imagenet-caffe-alex.mat');
    
    input_size= 43264;
    output_layer_id= 15;
else
    
    for i=1:numel(encoders_imdbs_path)
        encoders_imdbs{i}=load(encoders_imdbs_path{i});
    end
end
%load the standarization statistics
load('autoencoder/imagenet_mean.mat');
load('autoencoder/imagenet_std.mat');
%initialization
imdb.images.data=[];
imdb.images.set=[];
imdb.images.labels=[];
%the labels should change. each dataset labels starts from the last label of the previous dataset
init_lb=0;
%no augmentations for test
opts.numAugments=1;
opts.transformation='none';
all_val=[];
%load the encoders
for(i=1:numel(encoders_path))
    encoders{i}=load(encoders_path{i})
    if(isfield( encoders{i},'net'))
        encoders{i}=encoders{i}.net;
        
    end
    error_rate{i}=[];
    success_rate{i}=[];
    encoders{i}.layers{end} = struct('type', 'euclideanloss');
end
%start the test
for imdb_ind=1:numel(imdbs)
    
    this_imdb=load(imdbs{imdb_ind});
    
    this_imdb.images.labels=this_imdb.images.labels+init_lb;
    
    numOfClasses=numel(unique(this_imdb.images.labels));
    
    
    inds=find(this_imdb.images.set==3); %get test images
    if(ndims(this_imdb.images.data)==4)
        im_links= this_imdb.images.data(:,:,:,inds);
    else
        im_links= this_imdb.images.data(inds);
    end
    
    accuracy=0;
    
    for i=1:numel(im_links)
        %get images
        
        
        image = cnn_imagenet_get_batch(im_links(i),opts);
        
        
        origin_res = vl_simplenn_autoencoder(input_net, image) ;
        input=origin_res(output_layer_id);
        %reshape%
        input=reshape(input.x,1,1,input_size,[]);
        
        %------standarization----------
        input = bsxfun(@minus, input, imagenet_mean);
        
        input = bsxfun(@rdivide, input,imagenet_std);
        %--------------------------------------
        input=sigmoid(input);
        
        %---
        for enc_ind=1:numel(encoders)
            this_input=input;
            this_input = gpuArray(this_input);
            encoders{enc_ind}.layers{end}.class = this_input;
            
            res = vl_simplenn_autoencoder(encoders{enc_ind}, this_input, [], [], 'disableDropout', true);
            reconstruction_err(enc_ind) = gather(res(end).x);
        end
        %pass to the softmax
        reconstruction_err1=-reconstruction_err;
        reconstruction_err1=reshape(reconstruction_err1,1,1,numel(imdbs),1);
        
        
        
        soft_rec=tempy(reconstruction_err1,T);
        soft_rec=reshape(soft_rec,1,numel(imdbs));
        all_val=[all_val;soft_rec reconstruction_err];
        [x,min_ind]=max(soft_rec);
        
        
        %if correctly predicted task pass it to the expert model
        if((min_ind==imdb_ind ))
            
            %---------------------------------------------------------------------------------------
            smopts.disableDropout=1;
            
            for j=1:size(image,4)
                
                
                res = vl_simplenn(nets{imdb_ind}, image(:,:,:,j),[], [],smopts) ;
                
                % show the classification result
                scores = squeeze(gather(res(end).x)) ;
                dec=zeros(numel(scores),1);
                [bestScore, best] = max(scores) ;
                dec(best)=dec(best)+1;
            end
            [votes,best]=max(dec);
            predicted{i}=best;
            if(best==labels(i))
                accuracy=accuracy+1;
                
                
            end
        end
        %         fprintf('the predicted labels is %g the actual label is %g \n',best,  labels(i));
    end
    
    accuracy= (accuracy*100)/numel(im_links);
    acc{imdb_ind}=accuracy;
end
save(strcat('results/',output,'_accuracy'),'acc');
