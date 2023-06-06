function [concatenatedBoF] = BoF_dense_pyramid(image,step, kCluster)
% Set the number of clusters (vocab size)
numClusters = kCluster;

% Set the SIFT parameters
siftOpts = {'step', step};


% Read the RGB image
image = imread(image);

% Get the dimensions of the image
[rows, cols, ~] = size(image);

% Calculate the dimensions for each quadrant
quadrant_rows = rows / 2;
quadrant_cols = cols / 2;

% Extract the four quadrants
quadrant1 = image(1:quadrant_rows, 1:quadrant_cols, :);
quadrant2 = image(1:quadrant_rows, quadrant_cols+1:end, :);
quadrant3 = image(quadrant_rows+1:end, 1:quadrant_cols, :);
quadrant4 = image(quadrant_rows+1:end, quadrant_cols+1:end, :);


% Convert the quadrants to Lab color space
quadrant0_lab = im2gray(image);
quadrant1_lab = rgb2lab(quadrant1);
quadrant2_lab = rgb2lab(quadrant2);
quadrant3_lab = rgb2lab(quadrant3);
quadrant4_lab = rgb2lab(quadrant4);


% Extract color SIFT features for each quadrant
[~, descriptors0] = vl_dsift(single(quadrant0_lab(:,:,:)), siftOpts{:});
[~, descriptors1] = vl_dsift(single(quadrant1_lab(:, :, 1)), siftOpts{:});
[~, descriptors2] = vl_dsift(single(quadrant2_lab(:, :, 1)), siftOpts{:});
[~, descriptors3] = vl_dsift(single(quadrant3_lab(:, :, 1)), siftOpts{:});
[~, descriptors4] = vl_dsift(single(quadrant4_lab(:, :, 1)), siftOpts{:});

[~, descriptors5] = vl_dsift(single(quadrant1_lab(:, :, 2)), siftOpts{:});
[~, descriptors6] = vl_dsift(single(quadrant2_lab(:, :, 2)), siftOpts{:});
[~, descriptors7] = vl_dsift(single(quadrant3_lab(:, :, 2)), siftOpts{:});
[~, descriptors8] = vl_dsift(single(quadrant4_lab(:, :, 2)), siftOpts{:});

[~, descriptors9] = vl_dsift(single(quadrant1_lab(:, :, 3)), siftOpts{:});
[~, descriptors10] = vl_dsift(single(quadrant2_lab(:, :, 3)), siftOpts{:});
[~, descriptors11] = vl_dsift(single(quadrant3_lab(:, :, 3)), siftOpts{:});
[~, descriptors12] = vl_dsift(single(quadrant4_lab(:, :, 3)), siftOpts{:});


% Concatenate descriptors from all quadrants
allDescriptors = single([descriptors0, descriptors1, descriptors2, descriptors3, descriptors4,...
                         descriptors5, descriptors6, descriptors7, descriptors8,...
                         descriptors9, descriptors10, descriptors11, descriptors12]);

% Perform k-means clustering
size(allDescriptors')
[~, vocab] = kmeans(allDescriptors', numClusters);


% Compute BoF representation for each quadrant
bof0 = computeBoF(descriptors0, vocab, numClusters);
bof1 = computeBoF(descriptors1, vocab, numClusters);
bof2 = computeBoF(descriptors2, vocab, numClusters);
bof3 = computeBoF(descriptors3, vocab, numClusters);
bof4 = computeBoF(descriptors4, vocab, numClusters);

bof5 = computeBoF(descriptors5, vocab, numClusters);
bof6 = computeBoF(descriptors6, vocab, numClusters);
bof7 = computeBoF(descriptors7, vocab, numClusters);
bof8 = computeBoF(descriptors8, vocab, numClusters);

bof9 =  computeBoF(descriptors9, vocab, numClusters);
bof10 = computeBoF(descriptors10, vocab, numClusters);
bof11 = computeBoF(descriptors11, vocab, numClusters);
bof12 = computeBoF(descriptors12, vocab, numClusters);

concatenatedBoF = horzcat(bof0, bof1, bof2, bof3, bof4,...
                          bof5, bof6, bof7, bof8,...
                          bof9, bof10, bof11, bof12);
end
% Display the concatenated BoF represetation
%bar(concatenatedBoF);


% Set labels and title
%xlabel('Visual Word (Cluster) Index');
%ylabel('Frequency');
%title('Bag-of-Features (BoF) Histogram');