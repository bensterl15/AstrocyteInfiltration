%% make sure BinV2 is binary with values 0/1

dataPath = '~/Images/Astro_groundtruth/';

dir_info = dir([dataPath, '*BinV2.tif']);
fsn = {dir_info.name}';

for i = 1 : numel(fsn)
    fn_i = [dataPath, fsn{i}];
    im = tiffreadVolume(fn_i);
    
    fnout = [dataPath, fsn{i}(1 : end - 9), 'GT.tif'];
    imwrite(uint8(im > 0), fnout);
end


%% generate Cortex groundtruth by filling the holes for the boundry points

dataPath = '~/Images/Astro_groundtruth/';

imSize = [1200, 1200];

dir_info = dir([dataPath, '*V3.zip']);
fsn = {dir_info.name}';

homedir = '/Users/xruan/';
% for i = 1 : numel(fsn)
for i = 9 : numel(fsn)
    i
    tic
    fn_i = [dataPath, fsn{i}];
    
    fn_i = strrep(fn_i, '~', homedir);
    
    % read RIO coodinates
    sROI = ReadImageJROI(fn_i);
    
    im_rois = false(imSize);
    
    for j = 1 : numel(sROI)
        j
        im_ROI_j = false(imSize);
        mnCoordinates = sROI{j}.mnCoordinates + 1;
        mnCoordinates = max(1, min(mnCoordinates, imSize));
        bbox = [min(mnCoordinates), max(mnCoordinates)];
        [X, Y] = meshgrid(bbox(1) : bbox(3), bbox(2) : bbox(4));
        
        im_box = inpolygon(Y, X, mnCoordinates(:, 2), mnCoordinates(:, 1));
        figure(1), imagesc(im_box)
        inds = sub2ind(imSize, Y, X);
        im_rois(inds) = im_bound(inds) | im_box;
    end
    
    fnout = [dataPath, fsn{i}(1 : end - 6), 'CortexGT.tif'];
    imwrite(uint8(im_rois > 0), fnout);
    toc
end


