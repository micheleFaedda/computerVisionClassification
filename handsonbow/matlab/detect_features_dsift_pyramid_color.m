% function [] = detect_features_dsift(im_dir);
%
% Detect and describe features for all images in a directory
%
% IN:   im_dir    ... directory of images (assumed to be *.jpg)
%       file_ext  ... a string (e.g. 'sift')
%
% OUT:  for each image, a matlab file *.file_ext is created in directory
%       im_dir, SIFT descriptors.
%
% The output Matlab file contains structure "desc" with fileds:
%
%  desc.r    ... row index of each feature
%  desc.c    ... column index of each feature
%  desc.rad  ... radius (scale) of each feature
%  desc.sift ... SIFT descriptor for each feature
%
%

function detect_features_dsift_pyramid_color(im_dir,file_ext,varargin)

stride = 6;
do_resizeimage = 1;

dd = dir(fullfile(im_dir,'*.jpg'));
if nargin < 3
    scales = [32];
else
    scales = cell2mat(varargin(1));
end

for i = 1:length(dd)
    %for i = 1:length(dd)

    fname = fullfile(im_dir,dd(i).name);
    I=imread(fname);
    fname_out = [fname(1:end-3),file_ext];

    if exist(fname_out,'file')
        fprintf('File exists! Skipping %s \n',fname_out);
        continue;
    end;

    %resize the max dimension down to 300
    if do_resizeimage
        calingFactorX = 160 / size(I, 2);
        scalingFactorY = 160 / size(I, 1);

        % Resize the image using bicubic interpolation
        resizedImage = imresize(I, [156, 156], 'bicubic');

        %imshow(resizedImage);
        %I = rescale_max_size(I, 320, 1);
        %tmp_img = fullfile(im_dir,dd(i).name);
        %tmp_img = [tmp_img(1:end-4),'_tmp.jpg'];
        %tmp_img = [tmp_img(1:end-4),'.jpg'];
        %imwrite(I, tmp_img, 'jpg', 'quality', 90);
    end;
    %subdivide the image into 16 quadrants
    image = I ;

    % Get the size of the image
    [height, width, ~] = size(image);
    
    % Calculate the dimensions for each quadrant (4)
    quadrant_rows = height / 2;
    quadrant_columns = width / 2;

    % Extract the four quadrants
    quadrant1 = I(1:quadrant_rows, 1:quadrant_columns, :);
    quadrant2 = I(1:quadrant_rows, quadrant_columns+1:end, :);
    quadrant3 = I(quadrant_rows+1:end, 1:quadrant_columns, :);
    quadrant4 = I(quadrant_rows+1:end, quadrant_columns+1:end, :);
    
    [height2, width2, ~] = size(resizedImage);
    

    % Calculate the size of each quadrant (16)
    quadrantHeight = floor(height2 / 4);
    quadrantWidth = floor(width2 / 4);


    % Divide the image into quadrant
    quadrantImages = cell(4, 4);
    for i = 1:4
        for j = 1:4
            rowStart = (i - 1) * quadrantHeight + 1;
            rowEnd = i * quadrantHeight;
            colStart = (j - 1) * quadrantWidth + 1;
            colEnd = j * quadrantWidth;

            % Extract the quadrant
            quadrantImages{i, j} = resizedImage(rowStart:rowEnd, colStart:colEnd, :);
        end
    end
    
    %fprintf('Detecting and describing features: %s \n',fname_out);

    fname_txt = [fname(1:end-3) 'txt' ];

    n_image_store = 1 ;

    %scales = [16 24 32 48];
    sift_cell = cell(1,length(scales));
    

    %calcolo sift 4 quadranti
    for j = 1 : 4
       
        if(j == 1)
            I = quadrant1;
        end
        if(j == 2)
            I = quadrant2;
        end
        if(j == 3)
            I = quadrant3;
        end
        if(j == 4)
            I = quadrant4;
        end

        for psize=1:length(scales);
            [sift_tmp,gx{psize},gy{psize}]=sp_dense_sift(I,stride,scales(psize));
            sift_cell{psize}=reshape(sift_tmp,[size(sift_tmp,1)*size(sift_tmp,2) size(sift_tmp,3)]);
            rad{psize} = scales(psize)*ones(1,size(sift_cell{psize},1))';
        end
        sift = cell2mat(sift_cell');

        rad = cell2mat(rad');
        [gx] = cellfun(@(C)(C(:)),gx,'UniformOutput',false); %make each grid  of coordinates in the cell a vector
        c = cell2mat(gx'); % generate vector of coordinates

        [gy] = cellfun(@(C)(C(:)),gy,'UniformOutput',false); %make each grid of coordinates in the cell a vector
        r = cell2mat(gy');  % generate vector of coordinates

        desc = struct('sift',uint8(512*sift),'r',r,'c',c,'rad',rad);
   
        fname_out = [fname(1:end-4),'_',num2str(n_image_store),'.',file_ext];
    
        n_image_store=n_image_store+1;
        
        
        iSave(desc,fname_out);
        clear rad; 
    end
    %n_image_store = 5;
    %calcolo sift 16 quadranti 
    clear I; 
    for m = 1:4
        for n = 1:4
            I = quadrantImages{m, n};


            for psize=1:length(scales);
                [sift_tmp,gx{psize},gy{psize}]=sp_dense_sift(I,stride,scales(psize));
                sift_cell{psize}=reshape(sift_tmp,[size(sift_tmp,1)*size(sift_tmp,2) size(sift_tmp,3)]);
                rad{psize} = scales(psize)*ones(1,size(sift_cell{psize},1))';
            end
            sift = cell2mat(sift_cell');

            rad = cell2mat(rad');
            [gx] = cellfun(@(C)(C(:)),gx,'UniformOutput',false); %make each grid  of coordinates in the cell a vector
            c = cell2mat(gx'); % generate vector of coordinates

            [gy] = cellfun(@(C)(C(:)),gy,'UniformOutput',false); %make each grid of coordinates in the cell a vector
            r = cell2mat(gy');  % generate vector of coordinates

            desc = struct('sift',uint8(512*sift),'r',r,'c',c,'rad',rad);

            fname_out = [fname(1:end-4),'_',num2str(n_image_store),'.',file_ext];
          
            iSave(desc,fname_out);
            
            clear rad;
           
            n_image_store=n_image_store+1;
        end
    end
    %n_image_store = 17;
    %calcolo delle sift colore
    for j = 1 : 3 
        for psize=1:length(scales);
            
            [sift_tmp,gx{psize},gy{psize}]=sp_dense_sift(image(:,:,j),stride,scales(psize));
            sift_cell{psize}=reshape(sift_tmp,[size(sift_tmp,1)*size(sift_tmp,2) size(sift_tmp,3)]);
            rad{psize} = scales(psize)*ones(1,size(sift_cell{psize},1))';
        end
        sift = cell2mat(sift_cell');

        rad = cell2mat(rad');
        [gx] = cellfun(@(C)(C(:)),gx,'UniformOutput',false); %make each grid  of coordinates in the cell a vector
        c = cell2mat(gx'); % generate vector of coordinates

        [gy] = cellfun(@(C)(C(:)),gy,'UniformOutput',false); %make each grid of coordinates in the cell a vector
        r = cell2mat(gy');  % generate vector of coordinates

        desc = struct('sift',uint8(512*sift),'r',r,'c',c,'rad',rad);
      
        fname_out = [fname(1:end-4),'_',num2str(n_image_store),'.',file_ext];
        n_image_store=n_image_store+1;
        
        iSave(desc,fname_out);
        clear rad; 
    end

end


end


function iSave(desc,fName)
save(fName,'desc');
end

