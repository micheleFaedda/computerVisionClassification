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


function [acc_SVM_CHI2_train,acc_SVM_CHI2_val,acc_SVM_CHI2_test] = test_multidense_SIFT_pyramid_color(path,k_kmeans)
clc

% DATASET
dataset_dir = 'traincleaned';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

desc_name = 'all_pyramid_color';


% FLAGS
do_feat_extraction = 0;
do_split_sets = 0;

do_form_codebook = 1;
do_feat_quantization = 1;

do_svm_chi2_classification = 1;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 1;
visualize_res = 1;
%have_screen = ~isempty(getenv('DISPLAY'));
have_screen = 1;

% PATHS
basepath = path;
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 250; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

%%ROBA AGGIUNTA%%%%%%%
datasetDim = 1800;
perc = [0.6 0.2 0.2];
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
nwords_codebook = k_kmeans;

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
end

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
    fprintf('\n Loading SIFT training set for class %d/%d ',i,length(data));
    for j = 1:length(images_descs)
        for l = 1 : 23
            fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(l),'.all_pyramid_color']);
            tmp = load(fname,'-mat');
            tmp.desc.class=i;
            tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
            desc_train(lasti,l)=tmp.desc;
            desc_train(lasti,l).sift = single(desc_train(lasti,l).sift);

        end
        lasti=lasti+1;
    end
end


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
    fprintf('\n Loading SIFT test set for class %d/%d ',i,length(data));
    for j = 1:length(images_descs)
        for l = 1 : 23

            fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(l),'.all_pyramid_color']);

            tmp = load(fname,'-mat');
            tmp.desc.class=i;
            tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
            desc_test(lasti,l)=tmp.desc;
            desc_test(lasti,l).sift = single(desc_test(lasti,l).sift);
        end
        lasti=lasti+1;
    end;
end;



%% Load pre-computed SIFT features for validation images

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'val');
    fprintf('\n Loading SIFT validation set for class %d/%d ',i,length(data));
    for j = 1:length(images_descs)
        for l = 1 : 23

            fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(l),'.all_pyramid_color']);

            tmp = load(fname,'-mat');
            tmp.desc.class=i;
            tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
            desc_val(lasti,l)=tmp.desc;
            desc_val(lasti,l).sift = single(desc_val(lasti,l).sift);
        end
        lasti=lasti+1;
    end;
end;


%% Build visual vocabulary using k-means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_form_codebook
    fprintf('\nBuild visual vocabulary:\n');

    % concatenate all descriptors from all images into a n x d matrix
    DESC = [];
    labels_train_1 = cat(1,desc_train(:,1).class);
    labels_train_2 = cat(1,desc_train(:,2).class);
    labels_train_3 = cat(1,desc_train(:,3).class);
    labels_train_4 = cat(1,desc_train(:,4).class);

    labels_train_5 = cat(1,desc_train(:,5).class);
    labels_train_6 = cat(1,desc_train(:,6).class);
    labels_train_7 = cat(1,desc_train(:,7).class);
    labels_train_8 = cat(1,desc_train(:,8).class);

    labels_train_9 = cat(1,desc_train(:,9).class);
    labels_train_10 = cat(1,desc_train(:,10).class);
    labels_train_11= cat(1,desc_train(:,11).class);
    labels_train_12 = cat(1,desc_train(:,12).class);

    labels_train_13 = cat(1,desc_train(:,13).class);
    labels_train_14 = cat(1,desc_train(:,14).class);
    labels_train_15= cat(1,desc_train(:,15).class);
    labels_train_16 = cat(1,desc_train(:,16).class);

    labels_train_17 = cat(1,desc_train(:,17).class);
    labels_train_18 = cat(1,desc_train(:,18).class);
    labels_train_19= cat(1,desc_train(:,19).class);
    labels_train_20 = cat(1,desc_train(:,20).class);

    labels_train_21 = cat(1,desc_train(:,21).class);
    labels_train_22 = cat(1,desc_train(:,22).class);
    labels_train_23= cat(1,desc_train(:,23).class);

    for i=1:length(data)
        fprintf('Processing %d/%d \n',i,length(data));

        desc_train_1 = desc_train(:,1);
        desc_train_2 = desc_train(:,2);
        desc_train_3 = desc_train(:,3);
        desc_train_4 = desc_train(:,4);

        desc_train_5 = desc_train(:,5);
        desc_train_6 = desc_train(:,6);
        desc_train_7 = desc_train(:,7);
        desc_train_8 = desc_train(:,8);


        desc_train_9 = desc_train(:,9);
        desc_train_10 = desc_train(:,10);
        desc_train_11 = desc_train(:,11);
        desc_train_12 = desc_train(:,12);


        desc_train_13 = desc_train(:,13);
        desc_train_14 = desc_train(:,14);
        desc_train_15= desc_train(:,15);
        desc_train_16 = desc_train(:,16);

        desc_train_17 = desc_train(:,17);
        desc_train_18 = desc_train(:,18);
        desc_train_19= desc_train(:,19);
        desc_train_20 = desc_train(:,20);

        desc_train_21 = desc_train(:,21);
        desc_train_22 = desc_train(:,22);
        desc_train_23= desc_train(:,23);

        desc_class_1 = desc_train_1(labels_train_1==i);
        desc_class_2 = desc_train_2(labels_train_2==i);
        desc_class_3 = desc_train_3(labels_train_3==i);
        desc_class_4 = desc_train_4(labels_train_4==i);

        desc_class_5 = desc_train_5(labels_train_5==i);
        desc_class_6 = desc_train_6(labels_train_6==i);
        desc_class_7 = desc_train_7(labels_train_7==i);
        desc_class_8 = desc_train_8(labels_train_8==i);

        desc_class_9 = desc_train_9(labels_train_9==i);
        desc_class_10 = desc_train_10(labels_train_10==i);
        desc_class_11 = desc_train_11(labels_train_11==i);
        desc_class_12 = desc_train_12(labels_train_12==i);

        desc_class_13 = desc_train_13(labels_train_13==i);
        desc_class_14 = desc_train_14(labels_train_14==i);
        desc_class_15 = desc_train_15(labels_train_15==i);
        desc_class_16 = desc_train_16(labels_train_16==i);

        desc_class_17 = desc_train_17(labels_train_17==i);
        desc_class_18 = desc_train_18(labels_train_18==i);
        desc_class_19 = desc_train_19(labels_train_19==i);
        desc_class_20 = desc_train_20(labels_train_20==i);

        desc_class_21 = desc_train_21(labels_train_21==i);
        desc_class_22 = desc_train_22(labels_train_22==i);
        desc_class_23 = desc_train_23(labels_train_23==i);

        randimages = randperm(num_train_img);
        randimages =randimages(1:5);

        DESC = vertcat(DESC,desc_class_1(randimages,1).sift,desc_class_2(randimages,1).sift,desc_class_3(randimages,1).sift,desc_class_4(randimages,1).sift, ...
            desc_class_5(randimages,1).sift,desc_class_6(randimages,1).sift,desc_class_7(randimages,1).sift,desc_class_8(randimages,1).sift, ...
            desc_class_9(randimages,1).sift,desc_class_10(randimages,1).sift,desc_class_11(randimages,1).sift,desc_class_12(randimages,1).sift, ...
            desc_class_13(randimages,1).sift,desc_class_14(randimages,1).sift,desc_class_15(randimages,1).sift,desc_class_16(randimages,1).sift,...
            desc_class_17(randimages,1).sift,desc_class_18(randimages,1).sift,desc_class_19(randimages,1).sift,desc_class_20(randimages,1).sift, ...
            desc_class_21(randimages,1).sift,desc_class_22(randimages,1).sift,desc_class_23(randimages,1).sift);
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
%               K-means Descriptor quantization                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% K-means descriptor quantization means assignment of each feature
% descriptor with the identity of its nearest cluster mean, i.e.
% visual word. Your task is to quantize SIFT descriptors in all
% training and test images using the visual dictionary 'VC'
% constructed above.

if do_feat_quantization
    fprintf('\n%%%%%%%%%Feature quantization (hard-assignment)...\n');
    for i=1:length(desc_train)
        fprintf('Feature quantization training set: %d/%d \n',i,length(desc_train));
        for j = 1 : 23
            sift = desc_train(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_train(i,j).visword = visword;
            desc_train(i,j).quantdist = quantdist;
        end
    end
    for i=1:length(desc_test)
        fprintf('Feature quantization test set: %d/%d \n',i,length(desc_test));
        for j = 1 : 23
            sift = desc_test(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_test(i,j).visword = visword;
            desc_test(i,j).quantdist = quantdist;
        end
    end
    for i=1:length(desc_val)
        fprintf('Feature quantization validation set: %d/%d \n',i,length(desc_val));
        for j = 1 : 23
            sift = desc_val(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_val(i,j).visword = visword;
            desc_val(i,j).quantdist = quantdist;
        end
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 2: represent images with BOF histograms %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%               Bag-of-Features image classification                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n%%%%%%%%%%  Start BoF   %%%%%%%%%%%%%%%');
pause(5);

N = size(VC,1); % number of visual words

for i=1:length(desc_train)
    for j = 1 : 23
        visword = desc_train(i,j).visword;

        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_train(i,j).bof=H(:)';
    end
end
fprintf('\n     BoF training set completed        ');
pause(3);

for i=1:length(desc_test)
    for j = 1 : 23
        visword = desc_test(i,j).visword;
        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_test(i,j).bof=H(:)';
    end
end
fprintf('\n      BoF test set completed        ');
pause(3);

for i=1:length(desc_val)
    for j = 1 : 23
        visword = desc_val(i,j).visword;
        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_val(i,j).bof=H(:)';
    end
end
fprintf('\n      BoF validation set completed        ');
pause(3);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 3: image classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n%%%%%%%%%%  Manipulating BoFs   %%%%%%%%%%%%%%%');
pause(3);
desc_train_bof = [];
for i = 1 : size(desc_train)
    desc_train_bof(end+1,:) =horzcat(desc_train(i,1).bof,desc_train(i,2).bof,desc_train(i,3).bof,desc_train(i,4).bof,...
        desc_train(i,5).bof,desc_train(i,6).bof,desc_train(i,7).bof,desc_train(i,8).bof,...
        desc_train(i,9).bof,desc_train(i,10).bof,desc_train(i,11).bof,desc_train(i,12).bof,...
        desc_train(i,13).bof,desc_train(i,14).bof,desc_train(i,15).bof,desc_train(i,16).bof,...
        desc_train(i,17).bof,desc_train(i,18).bof,desc_train(i,19).bof,desc_train(i,20).bof, ...
        desc_train(i,21).bof,desc_train(i,22).bof,desc_train(i,23).bof);
end
fprintf('\nBoFs training set completed!');
pause(3);
desc_val_bof = [];
for i = 1 : size(desc_val)
    desc_val_bof(end+1,:) =horzcat(desc_val(i,1).bof,desc_val(i,2).bof,desc_val(i,3).bof,desc_val(i,4).bof,...
        desc_val(i,5).bof,desc_val(i,6).bof,desc_val(i,7).bof,desc_val(i,8).bof,...
        desc_val(i,9).bof,desc_val(i,10).bof,desc_val(i,11).bof,desc_val(i,12).bof,...
        desc_val(i,13).bof,desc_val(i,14).bof,desc_val(i,15).bof,desc_val(i,16).bof, ...
        desc_val(i,17).bof,desc_val(i,18).bof,desc_val(i,19).bof,desc_val(i,20).bof,...
        desc_val(i,21).bof,desc_val(i,22).bof,desc_val(i,23).bof);
end
fprintf('\nBoFs validation set completed!');
pause(3);
desc_test_bof = [];
for i = 1 : size(desc_test)
    desc_test_bof(end+1,:) = horzcat(desc_test(i,1).bof,desc_test(i,2).bof,desc_test(i,3).bof,desc_test(i,4).bof,...
        desc_test(i,5).bof,desc_test(i,6).bof,desc_test(i,7).bof,desc_test(i,8).bof,...
        desc_test(i,9).bof,desc_test(i,10).bof,desc_test(i,11).bof,desc_test(i,12).bof,...
        desc_test(i,13).bof,desc_test(i,14).bof,desc_test(i,15).bof,desc_test(i,16).bof, ...
        desc_test(i,17).bof,desc_test(i,18).bof,desc_test(i,19).bof,desc_test(i,20).bof,...
        desc_test(i,21).bof,desc_test(i,22).bof,desc_test(i,23).bof);
end
fprintf('\nBoFs test set completed!');
pause(3);


% Concatenate bof-histograms into training and test matrices

bof_train=cat(1,desc_train_bof);
bof_test=cat(1,desc_test_bof);
bof_val=cat(1,desc_val_bof);

% Construct label Concatenate bof-histograms into training and test matrices
labels_train=cat(1,desc_train(:,1).class);
labels_test=cat(1,desc_test(:,1).class);
labels_val=cat(1,desc_val(:,1).class);

fprintf('\nBoFs ready to classification!\n');
pause(3);

%% 4.3 & 4.4: CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_svm_chi2_classification

    fprintf('\n%%%%%%%%%% COMPUTE KERNELS  %%%%%%%%%%%%%%%');
    pause(3);

    % compute kernel matrix
    Ktrain = kernel_expchi2(bof_train,bof_train);
    fprintf('\nKernel training set completed!');
    pause(3);
    Ktest = kernel_expchi2(bof_test,bof_train);
    fprintf('\nKernel test set completed!');
    pause(3);
    Kval = kernel_expchi2(bof_val,bof_train);
    fprintf('\nKernel validation set completed!');
    pause(3);


    % Compute the chi-squared kernel matrix

    fprintf('\n%%%%%%%%%% TRAINING THE MODEL  %%%%%%%%%%%%%%%');
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