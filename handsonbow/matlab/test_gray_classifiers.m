
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

% DATASET
dataset_dir='4_ObjectCategories';
%dataset_dir = '15_ObjectCategories';
dataset_dir = 'traincleaned';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

%desc_name = 'sift';
desc_name = 'dsift';

desc_name = 'all_pyramid_color';

%desc_name = 'msdsift';

% FLAGS
do_feat_extraction = 0;
do_split_sets = 0;

do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 1;
do_chi2_NN_classification = 1;
do_svm_linar_classification = 1;
do_svm_llc_linar_classification = 1;
do_svm_precomp_linear_classification = 1;
do_svm_inter_classification = 1;
do_svm_chi2_classification = 1;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 0;
visualize_res = 0;
have_screen = ~isempty(getenv('DISPLAY'));

% PATHS
path = '/Users/michele/Documents/GitHub/computerVisionClassification/handsonbow';

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

method_names= string([]);


% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = subdivision(1,1);

% number of images selected for validation (e.g. 30 for Caltech-101)
num_val_img = subdivision(1,2);

% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img =  subdivision(1,3);
nwords_codebook = 1000;

% image file extension
file_ext='jpg';

% Create a new dataset split
file_split = 'split.mat';
if do_split_sets
    data = create_dataset_split_structure(fullfile(basepath, 'img', ...
        dataset_dir),num_train_img,num_test_img ,file_ext);
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
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(24),'.all_pyramid_color']);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_train(lasti)=tmp.desc;
        desc_train(lasti).sift = single(desc_train(lasti).sift);
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
        pause
    end
end


%% Load pre-computed SIFT features for test images

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
    for j = 1:length(images_descs)
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(24),'.all_pyramid_color']);
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
    end;
end;




%% Load pre-computerd SIFT features for validation images
lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files(data,i,file_ext,desc_name,'val');
    for j = 1:length(images_descs)
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-18),'_',num2str(24),'.all_pyramid_color']);
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

    for i=1:length(desc_val)
        sift = desc_val(i).sift(:,:);
        dmat = eucliddist(sift,VC);
        [quantdist,visword] = min(dmat,[],2);
        % save feature labels
        desc_val(i).visword = visword;
        desc_val(i).quantdist = quantdist;
    end

    for i=1:length(desc_test)
        sift = desc_test(i).sift(:,:);
        dmat = eucliddist(sift,VC);
        [quantdist,visword] = min(dmat,[],2);
        % save feature labels
        desc_test(i).visword = visword;
        desc_test(i).quantdist = quantdist;
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
                img=rgb2gray(imread(d.imgfname));

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
        pause
    end
end



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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 2                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%LLC Coding
if do_svm_llc_linar_classification
    for i=1:length(desc_train)
        disp(desc_train(i).imgfname);
        desc_train(i).llc = max(LLC_coding_appr(VC,desc_train(i).sift)); %max-pooling
        desc_train(i).llc=desc_train(i).llc/norm(desc_train(i).llc); %L2 normalization
    end
    for i=1:length(desc_val)
        disp(desc_val(i).imgfname);
        desc_val(i).llc = max(LLC_coding_appr(VC,desc_val(i).sift)); %max-pooling
        desc_val(i).llc=desc_val(i).llc/norm(desc_val(i).llc); %L2 normalization
    end
    for i=1:length(desc_test)
        disp(desc_test(i).imgfname);
        desc_test(i).llc = max(LLC_coding_appr(VC,desc_test(i).sift));
        desc_test(i).llc=desc_test(i).llc/norm(desc_test(i).llc);
    end
end
%%%%end LLC coding



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 3: image classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Concatenate bof-histograms into training and test matrices
bof_train=cat(1,desc_train.bof);
bof_val=cat(1,desc_val.bof);
bof_test=cat(1,desc_test.bof);
if do_svm_llc_linar_classification
    llc_train = cat(1,desc_train.llc);
    llc_val = cat(1,desc_val.llc);
    llc_test = cat(1,desc_test.llc);
end

% Construct label Concatenate bof-histograms into training and test matrices
labels_train=cat(1,desc_train.class);
labels_val=cat(1,desc_val.class);
labels_test=cat(1,desc_test.class);


%% NN classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_L2_NN_classification
    % Compute L2 distance between BOFs of test and training images
    bof_l2dist_test=eucliddist(bof_test,bof_train);
    bof_l2dist_val=eucliddist(bof_val,bof_train);
    bof_l2dist_train=eucliddist(bof_train,bof_train);

    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi_te] = min(bof_l2dist_test,[],2);
    bof_l2lab_test = labels_train(mi_te);

    [mv,mi_val] = min(bof_l2dist_val,[],2);
    bof_l2lab_val = labels_train(mi_val);


    [mv,mi_tr] = min(bof_l2dist_train,[],2);
    bof_l2lab_train = labels_train(mi_tr);

    method_name='NN L2';
    acc_NNL2_test=sum(bof_l2lab_test==labels_test)/length(labels_test);
    acc_NNL2_val=sum(bof_l2lab_val==labels_val)/length(labels_val);
    acc_NNL2_train=sum(bof_l2lab_train==labels_train)/length(labels_train);

    fprintf('\n*** %s ***\nAccuracy test = %1.4f%% (classification)\n',method_name,acc_NNL2_test*100);
    fprintf('\n*** %s ***\nAccuracy validation = %1.4f%% (classification)\n',method_name,acc_NNL2_val*100);
    fprintf('\n*** %s ***\nAccuracy train = %1.4f%% (classification)\n',method_name,acc_NNL2_train*100);

    % Compute classification accuracy
    acc_NNL2_lab_test = compute_accuracy(data,labels_test,bof_l2lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");

    acc_NNL2_lab_val = compute_accuracy(data,labels_val,bof_l2lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");

    acc_NNL2_lab_train =  compute_accuracy(data,labels_train,bof_l2lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");

    method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_NNL2_lab_train,acc_NNL2_lab_val,acc_NNL2_lab_test];

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 3: Image classification                                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Repeat Nearest Neighbor image classification using Chi2 distance
% instead of L2. Hint: Chi2 distance between two row-vectors A,B can
% be computed with d=chi2(A,B);
%
% TODO:
% 3.1 Nearest Neighbor classification with Chi2 distance
%     Compute and compare overall and per-class classification
%     accuracies to the L2 classification above


if do_chi2_NN_classification
    % compute pair-wise CHI2
    bof_chi2dist_test = zeros(size(bof_test,1),size(bof_train,1));
    bof_chi2dist_val = zeros(size(bof_val,1),size(bof_train,1));
    bof_chi2dist_train = zeros(size(bof_train,1),size(bof_train,1));

    % bof_chi2dist = slmetric_pw(bof_train, bof_test, 'chisq');
    for i = 1:size(bof_test,1)
        for j = 1:size(bof_train,1)
            bof_chi2dist_test(i,j) = chi2(bof_test(i,:),bof_train(j,:));
        end
    end
    for i = 1:size(bof_val,1)
        for j = 1:size(bof_train,1)
            bof_chi2dist_val(i,j) = chi2(bof_val(i,:),bof_train(j,:));
        end
    end
    for i = 1:size(bof_train,1)
        for j = 1:size(bof_train,1)
            bof_chi2dist_train(i,j) = chi2(bof_train(i,:),bof_train(j,:));
        end
    end

    % Nearest neighbor classification (1-NN) using Chi2 distance
    [mv,mi_te] = min(bof_chi2dist_test,[],2);
    bof_chi2lab_test = labels_train(mi_te);

    [mv,mi_val] = min(bof_chi2dist_val,[],2);
    bof_chi2lab_val = labels_train(mi_val);

    [mv,mi_tr] = min(bof_chi2dist_train,[],2);
    bof_chi2lab_train = labels_train(mi_tr);

    method_name='NN Chi-2';
    acc_NNCHI2_test=sum(bof_chi2lab_test==labels_test)/length(labels_test);
    acc_NNCHI2_val=sum(bof_chi2lab_val==labels_val)/length(labels_val);
    acc_NNCHI2_train=sum(bof_chi2lab_train==labels_train)/length(labels_train);

    fprintf('*** %s ***\nAccuracy test = %1.4f%% (classification)\n',method_name,acc_NNCHI2_test*100);
    fprintf('*** %s ***\nAccuracy val = %1.4f%% (classification)\n',method_name,acc_NNCHI2_val*100);
    fprintf('*** %s ***\nAccuracy training = %1.4f%% (classification)\n',method_name,acc_NNCHI2_train*100);

    % Compute classification accuracy
    acc_NNCHI2_test =  compute_accuracy(data,labels_test,bof_chi2lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");
    acc_NNCHI2_val = compute_accuracy(data,labels_val,bof_chi2lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");
    acc_NNCHI2_train = compute_accuracy(data,labels_train,bof_chi2lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");

    method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_NNCHI2_train,acc_NNCHI2_val,acc_NNCHI2_test];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 3.1                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% SVM classification (using libsvm) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Use cross-validation to tune parameters:
% - the -v 5 options performs 5-fold cross-validation, this is useful to tune
% parameters
% - the result of the 5 fold train/test split is averaged and reported
%
% example: for the parameter C (soft margin) use log2space to generate
%          (say 5) different C values to test
%          xval_acc=svmtrain(labels_train,bof_train,'-t 0 -v 5');


% LINEAR SVM
if do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,bof_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,bof_train,['-t 0 -c ' num2str(C_vals(ind))]);
    disp('*** SVM - linear ***');

    svm_lab_test=svmpredict(labels_test,bof_test,model);
    svm_lab_val=svmpredict(labels_val,bof_val,model);
    svm_lab_train=svmpredict(labels_train,bof_train,model);

    method_name='SVM linear';
    % Compute classification accuracy
    acc_SVML_lab_test = compute_accuracy(data,labels_test,svm_lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");

    acc_SVML_lab_val = compute_accuracy(data,labels_val,svm_lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");

    acc_SVML_lab_train = compute_accuracy(data,labels_train,svm_lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");

    method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_SVML_lab_train,acc_SVML_lab_val,acc_SVML_lab_test];

end

%% LLC LINEAR SVM
if do_svm_llc_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,llc_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,llc_train,['-t 0 -c ' num2str(C_vals(ind))]);

    disp('*** SVM - linear LLC max-pooling ***');
    svm_llc_lab_test=svmpredict(labels_test,llc_test,model);
    svm_llc_lab_val=svmpredict(labels_val,llc_val,model);
    svm_llc_lab_train=svmpredict(labels_train,llc_train,model);
    method_name='llc+max-pooling';

    acc_LLCSVM_lab_test = compute_accuracy(data,labels_test,svm_llc_lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");
    acc_LLCSVM_lab_val = compute_accuracy(data,labels_val,svm_llc_lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");
    acc_LLCSVM_lab_train = compute_accuracy(data,labels_train,svm_llc_lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");

    method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_LLCSVM_lab_train,acc_LLCSVM_lab_val,acc_LLCSVM_lab_test];
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 4: Image classification: SVM classifier                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-computed LINEAR KERNELS.
% Repeat linear SVM image classification; let's try this with a
% pre-computed kernel.
%
% TODO:
% 4.1 Compute the kernel matrix (i.e. a matrix of scalar products) and
%     use the LIBSVM precomputed kernel interface.
%     This should produce the same results.


if do_svm_precomp_linear_classification
    % compute kernel matrix
    Ktrain = bof_train*bof_train';
    Ktest = bof_test*bof_train';
    Kval = bof_val*bof_train';

    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))]);
    % we supply the missing scalar product (actually the values of
    % non-support vectors could be left as zeros....
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - precomputed linear kernel ***');
    precomp_svm_lab_test=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    precomp_svm_lab_val=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
    precomp_svm_lab_train=svmpredict(labels_train,[(1:size(Ktrain,1))' Ktrain],model);

    method_name='SVM precomp linear';
    % Compute classification accuracy
    acc_SVMPrec_lab_test = compute_accuracy(data,labels_test,precomp_svm_lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");
    acc_SVMPrec_lab_val = compute_accuracy(data,labels_val,precomp_svm_lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");
    acc_SVMPrec_lab_train = compute_accuracy(data,labels_train,precomp_svm_lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");
    
    method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_SVMPrec_lab_train,acc_SVMPrec_lab_val,acc_SVMPrec_lab_test];

    % result is the same??? must be!
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.1                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Pre-computed NON-LINAR KERNELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO:
% 4.2 Train the SVM with a precomputed non-linear histogram intersection
%     kernel and select the best C parameter for the trained model using
%     cross-validation.
% 4.3 Experiment with other different non-linear kernels: RBF and Chi^2.
%     Chi^2 must be precomputed as in the previous exercise.
% 4.4 Certain kernels have other parameters (e.g. gamma for RBF/Chi^2)...
%     implement a cross-validation procedure to select the optimal
%     parameters (as in 3).


%% 4.2: INTERSECTION KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%
% try a non-linear svm with the histogram intersection kernel!

if do_svm_inter_classification
    Ktrain=zeros(size(bof_train,1),size(bof_train,1));
    for i=1:size(bof_train,1)
        for j=1:size(bof_train,1)
            hists = [bof_train(i,:);bof_train(j,:)];
            Ktrain(i,j)=sum(min(hists));
        end
    end

    Ktest=zeros(size(bof_test,1),size(bof_train,1));
    for i=1:size(bof_test,1)
        for j=1:size(bof_train,1)
            hists = [bof_test(i,:);bof_train(j,:)];
            Ktest(i,j)=sum(min(hists));
        end
    end

    Kval=zeros(size(bof_val,1),size(bof_train,1));
    for i=1:size(bof_val,1)
        for j=1:size(bof_train,1)
            hists = [bof_val(i,:);bof_train(j,:)];
            Kval(i,j)=sum(min(hists));
        end
    end

    % cross-validation
    C_vals=log2space(3,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... consider this if the kernel is computationally inefficient.
    disp('*** SVM - intersection kernel ***');
    [precomp_ik_svm_lab_test,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    [precomp_ik_svm_lab_val,conf]=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
    [precomp_ik_svm_lab_train,conf]=svmpredict(labels_train,[(1:size(Ktrain,1))' Ktrain],model);

    method_name='SVM IK';
    % Compute classification accuracy
     acc_SVMIK_lab_test = compute_accuracy(data,labels_test,precomp_ik_svm_lab_test,classes,method_name,desc_test,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TEST SET");
     acc_SVMIK_lab_val = compute_accuracy(data,labels_val,precomp_ik_svm_lab_val,classes,method_name,desc_val,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"VALIDATION SET");
     acc_SVMIK_lab_train = compute_accuracy(data,labels_train,precomp_ik_svm_lab_train,classes,method_name,desc_train,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen,"TRAINING SET");

     method_names(end+1) = method_name + " k=" + num2str(nwords_codebook);
    bar_values(end+1, :) = [acc_SVMIK_lab_train,acc_SVMIK_lab_val,acc_SVMIK_lab_test];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.2                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 4.3 & 4.4: CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_svm_chi2_classification
    % compute kernel matrix
 fprintf('\n%%%%%%%%%% COMPUTE KERNELS  %%%%%%%%%%%%%%%\n');
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

    fprintf('\n%%%%%%%%%% TRAINING THE MODEL Chi^2 %%%%%%%%%%%%%%%');
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


    method_names(end+1) = method_name + " k=" + nwords_codebook;
    bar_values(end+1, :) = [acc_SVM_CHI2_train,acc_SVM_CHI2_val,acc_SVM_CHI2_test];

end

 f_accurancy_final= figure(1000);
    f_accurancy_final.Name = "Comparisons classifiers - Multidense SIFT - img. in gray" ;

    display_bar_accurancy(f_accurancy_final, bar_values, method_names);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.3 and 4.4                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

