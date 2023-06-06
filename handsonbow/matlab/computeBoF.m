function bof = computeBoF(features, vocab,numClusters)
    % Compute Euclidean distances between features and the visual vocabulary
    distances = pdist2(double(features'), double(vocab));

    % Assign features to the nearest visual word (cluster)
    [~, visualWordIndices] = min(distances, [], 2);

    % Compute the histogram of visual word occurrences (BoF representation)
    bof = histcounts(visualWordIndices, 1:numClusters+1);
    bof = bof / sum(bof);  % Normalize the histogram
end


