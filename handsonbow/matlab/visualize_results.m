function [ ] = visualize_results( classes, desc_test, labels_test, labels_result )

%VISUALIZE_EXAMPLES Illustrate correcly classified and missclassified
%samples of each class.

figure;

for i=1:length(classes)

    ind=find(labels_test==i);
    %bof_chi2acc_class=mean(result_labels(ind)==labels_test(ind));
    indcorr=ind(find(labels_result(ind)==labels_test(ind)));
    indmiss=ind(find(labels_result(ind)~=labels_test(ind)));

    clf
    imgcorr={};
    if length(indcorr)
        for j=1:length(indcorr)
            imgName = desc_test(indcorr(j)).imgfname;
            imgName = regexprep(imgName, '_(.*?)\.', '.');
            imgcorr{end+1}=imread(imgName);
        end
        subplot(1,2,1), showimage(combimage(imgcorr,[],1))
        title(sprintf('%d Correctly classified %s images',length(indcorr),classes{i}))
    end

    imgmiss={};
    if length(indmiss)
        for j=1:length(indmiss)
            imgName = desc_test(indmiss(j)).imgfname;
            imgName = regexprep(imgName, '_(.*?)\.', '.');
            imgmiss{end+1}=imread(imgName);
            % Specify the folder path and image filename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Create the full file path
            %fullFilePath = fullfile(imgName);

            % Check if the file exists
            %if exist(fullFilePath, 'file')
                % Delete the file
            %    delete(imgName);
             %   disp('Image deleted successfully.');
            %else
              %  disp('Image not found.');
            %end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        subplot(1,2,2), showimage(combimage(imgmiss,[],1))
        title(sprintf('%d Miss-classified %s images',length(indmiss),classes{i}))
    end

    %pause;
end

end

