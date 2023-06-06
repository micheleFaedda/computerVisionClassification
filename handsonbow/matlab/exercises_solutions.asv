
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ICPR 2014 Tutorial                                                     %
%  Hands on Advanced Bag-of-Words Models for Visual Recognition           %
%                                                                         %
%  Instructors:                                                           %
%  L. Ballan     <lamberto.ballan@unifi.it>                               %
%  L. Seidenari  <lorenzo.seidenari@unifi.it>                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   BOW pipeline: Image classification using bag-of-features              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Part 1:  Load and quantize pre-computed image features                %
%   Part 2:  Represent images by histograms of quantized features         %
%   Part 3:  Classify images with nearest neighbor classifier             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
clc

% DATASET
%dataset_dir='4_ObjectCategories';
%dataset_dir = '15_ObjectCategories';
dataset_dir = 'train';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

%desc_name = 'sift';
desc_name = 'dsift';
%desc_name = 'msdsift';
%desc_name = 'sift_pyramid';
%desc_name = 'gyst';

% FLAGS
do_feat_extraction = 1;
do_split_sets = 1;

do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 0;
do_chi2_NN_classification = 0;
do_svm_linar_classification = 0;
do_svm_llc_linar_classification = 0;
do_svm_precomp_linear_classification = 0;
do_svm_inter_classification = 0;
do_svm_chi2_classification = 1;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 1;
visualize_res = 0;
%have_screen = ~isempty(getenv('DISPLAY'));
have_screen = 1;

% PATHS
basepath = '/Users/michele/Desktop/CV project/handsonbow';
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 50; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

%%ROBA AGGIUNTA%%%%%%%
datasetDim = 2000;
perc = [0.5 0.3 0.2];
subdivision = [datasetDim datasetDim datasetDim];
subdivision = perc .* subdivision;

bar_values = [];

methods_name= string([]);


% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = subdivision(1,1);

% number of images selected for validation (e.g. 30 for Caltech-101)
num_val_img = subdivision(1,2);

% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img =  subdivision(1,3);

% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 500;

% image file extension
file_ext='jpg';

% Create a new dataset split
file_split = 'split.mat';
if do_split_sets
    data = create_dataset_split_structure(fullfile(basepath, 'img', ...
        dataset_dir),num_train_img,num_test_img, num_val_img ,file_ext);
    save(fullfile(basepath,'img',dataset_dir,file_split),'data');
else
    load(fullfile(basepath,'img',dataset_dir,file_split));
end
classes = {data.classname}; % create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction
   extract_sift_features(fullfile('..','img',dataset_dir),desc_name)
%extract_gist_features(fullfile('..','img',dataset_dir),desc_name);

end

%crea il dataset 

%suddividilo 


%passa le immagini una ad una alla funzuone per pyramid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 1: quantize pre-computed image features %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load pre-computed SIFT features for training images

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'train');
    for j = 1:length(images_descs)
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_train(lasti)=tmp.desc;
        desc_train(lasti).sift = single(desc_train(lasti).sift);
        lasti=lasti+1;
    end;
end;


%% Visualize SIFT features for training images
if (visualize_feat && have_screen)
    nti=10;
    fprintf('\nVisualize features for %d training images\n', nti);
    imgind=randperm(length(desc_train));
    for i=1:nti
        d=desc_train(imgind(i));
        clf, showimage(imread(strrep(d.imgfname,'_train','')));
        x=d.c;
        y=d.r;
        rad=d.rad;
        showcirclefeaturesrad([x,y,rad]);
        title(sprintf('%d features in %s',length(d.c),d.imgfname));
        %pause
    end
end


%% Load pre-computed SIFT features for test images

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
    for j = 1:length(images_descs)
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
    end;
end;



%% Load pre-computed SIFT features for validation images

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'val');
    for j = 1:length(images_descs)
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_val(lasti)=tmp.desc;
        desc_val(lasti).sift = single(desc_val(lasti).sift);
        lasti=lasti+1;
    end;
end;


%% Build visual vocabulary using k-means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_form_codebook
    fprintf('\nBuild visual vocabulary:\n');

    % concatenate all descriptors from all images into a n x d matrix
    DESC = [];
    labels_train = cat(1,desc_train.class);
    for i=1:length(data)
        desc_class = desc_train(labels_train==i);
        randimages = randperm(num_train_img);
        randimages =randimages(1:5);
        DESC = vertcat(DESC,desc_class(randimages).sift);
    end

    % sample random M (e.g. M=20,000) descriptors from all training descriptors
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);

    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters...\n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects
    % one point per column rather than per row

    % form options structure for clustering
    cluster_options.maxiters = max_km_iters;
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    clear DESC;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 1: K-means Descriptor quantization                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% K-means descriptor quantization means assignment of each feature
% descriptor with the identity of its nearest cluster mean, i.e.
% visual word. Your task is to quantize SIFT descriptors in all
% training and test images using the visual dictionary 'VC'
% constructed above.
%
% TODO:
% 1.1 compute Euclidean distances between VC and all descriptors
%     in each training and test image. Hint: to compute all-to-all
%     distance matrix for two sets of descriptors D1 & D2 use
%     dmat=eucliddist(D1,D2);
% 1.2 compute visual word ID for each feature by minimizing
%     the distance between feature SIFT descriptors and VC.
%     Hint: apply 'min' function to 'dmat' computed above along
%     the dimension (1 or 2) corresponding to VC, i.g.:
%     [mv,visword]=min(dmat,[],2); if you compute dmat as
%     dmat=eucliddist(dscr(i).sift,VC);

if do_feat_quantization
    fprintf('\nFeature quantization (hard-assignment)...\n');
    for i=1:length(desc_train)
        sift = desc_train(i).sift(:,:);
        dmat = eucliddist(sift,VC);
        [quantdist,visword] = min(dmat,[],2);
        % save feature labels
        desc_train(i).visword = visword;
        desc_train(i).quantdist = quantdist;
    end

    for i=1:length(desc_test)
        sift = desc_test(i).sift(:,:);
        dmat = eucliddist(sift,VC);
        [quantdist,visword] = min(dmat,[],2);
        % save feature labels
        desc_test(i).visword = visword;
        desc_test(i).quantdist = quantdist;
    end

    for i=1:length(desc_val)
        sift = desc_val(i).sift(:,:);
        dmat = eucliddist(sift,VC);
        [quantdist,visword] = min(dmat,[],2);
        % save feature labels
        desc_val(i).visword = visword;
        desc_val(i).quantdist = quantdist;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 1                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Visualize visual words (i.e. clusters) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  To visually verify feature quantization computed above, we can show
%  image patches corresponding to the same visual word.

if (visualize_words && have_screen)
    figure;
    %num_words = size(VC,1) % loop over all visual word types
    num_words = 10;
    fprintf('\nVisualize visual words (%d examples)\n', num_words);
    for i=1:num_words
        patches={};
        for j=1:length(desc_train) % loop over all images
            d=desc_train(j);
            ind=find(d.visword==i);
            if length(ind)
                %img=imread(strrep(d.imgfname,'_train',''));
                img=im2gray(imread(d.imgfname));

                x=d.c(ind); y=d.r(ind); r=d.rad(ind);
                bbox=[x-2*r y-2*r x+2*r y+2*r];
                for k=1:length(ind) % collect patches of a visual word i in image j
                    patches{end+1}=cropbbox(img,bbox(k,:));
                end
            end
        end
        % display all patches of the visual word i
        clf, showimage(combimage(patches,[],1.5))
        title(sprintf('%d examples of Visual Word #%d',length(patches),i))
        %pause
    end
end


%CACCA%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 2: represent images with BOF histograms %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 2: Bag-of-Features image classification                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Represent each image by the normalized histogram of visual
% word labels of its features. Compute word histogram H over
% the whole image, normalize histograms w.r.t. L1-norm.
%
% TODO:
% 2.1 for each training and test image compute H. Hint: use
%     Matlab function 'histc' to compute histograms.


    N = size(VC,1); % number of visual words

    for i=1:length(desc_train)
        visword = desc_train(i).visword;

        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_train(i).bof=H(:)';
    end

    for i=1:length(desc_test)
        visword = desc_test(i).visword;
        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_test(i).bof=H(:)';
    end


    for i=1:length(desc_val)
        visword = desc_val(i).visword;
        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_val(i).bof=H(:)';
    end


 



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%% Part 3: image classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Concatenate bof-histograms into training and test matrices
    bof_train=cat(1,desc_train.bof);
    bof_test=cat(1,desc_test.bof);
    bof_val=cat(1,desc_val.bof);

    if do_svm_llc_linar_classification
        llc_train = cat(1,desc_train.llc);
        llc_test = cat(1,desc_test.llc);
        llc_val = cat(1,desc_val.llc);
    end

    % Construct label Concatenate bof-histograms into training and test matrices
    labels_train=cat(1,desc_train.class);
    labels_test=cat(1,desc_test.class);
    labels_val=cat(1,desc_val.class);


    %% 4.3 & 4.4: CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%

    if do_svm_chi2_classification
        % compute kernel matrix
        Ktrain = kernel_expchi2(bof_train,bof_train);
        Ktest = kernel_expchi2(bof_test,bof_train);
        Kval = kernel_expchi2(bof_val,bof_train);

        % cross-validation
        C_vals=log2space(2,10,5);
        for i=1:length(C_vals);
            opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
            xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
        end
        [v,ind]=max(xval_acc);

        % train the model and test
        model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
        % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros....
        % consider this if the kernel is computationally inefficient.
        disp('*** SVM - Chi2 kernel ***');
        [precomp_chi2_svm_lab_test,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
        [precomp_chi2_svm_lab_val,conf]=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
        [precomp_chi2_svm_lab_train,conf]=svmpredict(labels_train,[(1:size(Ktrain,1))' Ktrain],model);

        method_name="SVM Chi2";
        % Compute classification accuracy
        acc_SVM_CHI2_train = compute_accuracy(data,labels_train,precomp_chi2_svm_lab_train,classes,method_name,desc_train,...
            visualize_confmat & have_screen,...
            visualize_res & have_screen,"TRAINING SET");
        acc_SVM_CHI2_val = compute_accuracy(data,labels_val,precomp_chi2_svm_lab_val,classes,method_name,desc_val,...
            visualize_confmat & have_screen,...
            visualize_res & have_screen,"VALIDATION SET");
        acc_SVM_CHI2_test = compute_accuracy(data,labels_test,precomp_chi2_svm_lab_test,classes,method_name,desc_test,...
            visualize_confmat & have_screen,...
            visualize_res & have_screen,"TEST SET");


        methods_name(end+1) = method_name + ' k=' + nwords_codebook;
        bar_values(end+1, :) = [acc_SVM_CHI2_train,acc_SVM_CHI2_val,acc_SVM_CHI2_test];

    end
  


%display accurancy bars
%methods_name(end) = [];
f_accurancy_final= figure;
f_accurancy_final.Name = "Accurancy methods";
display_bar_accurancy(f_accurancy_final, bar_values,methods_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.3 and 4.4                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

