%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @article{Canh2018_MSCSNet,
%   title={Multi-Scale Deep Compressive Sensing Network},
%   author={Thuong, Nguyen Canh and Byeungwoo, Jeon},
%   conference={IEEE International Conference on Visual Comunication and Image Processing},
%   year={2018}
% }
% by Thuong Nguyen Canh (9/2018)
% ngcthuong@gmail.com
% https://github.com/AtenaKid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You need to install Matconvnet in order to run this code 

warning('off','all')
addpath('D:\matconvnet-1.0-beta25\matlab\mex');     % Link to your Matconvnet Mex file 
% addpath('matconvnet-1.0-beta25\matlab');

addpath('.\utilities');
folderTest  = 'Classic13_512';
networkTest = {'CSNet' 'MS-CSNet1' 'MS-CSNet2' 'MSCSNet-3'};      % 10

showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32; 
isLearnMtx  = [1, 0];
network     = networkTest{3}; 

for samplingRate = [0.1:0.1:0.3]
    modelName   = [network '_r' num2str(samplingRate)]; %%% model name
        
    data = load(fullfile('models', network ,[modelName,'.mat']));
    net  = dagnn.DagNN.loadobj(data.net);
    if strcmp(network,'CSNet')
        net.renameVar('x0', 'input'); 
        net.renameVar('x12', 'prediction'); 
    else
        net.removeLayer(net.layers(end).name) ;
    end
        
    net.mode = 'test';
    net.move('gpu');
        
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})) );
    end
    
    PSNRs_CSNet = zeros(1,length(filePaths));
    SSIMs_CSNet = zeros(1,length(filePaths));
    
    count = 1;
    allName = cell(1);
    
    for i = 1:length(filePaths)
        
        %%% read images
        image = imread(fullfile('testsets', folderTest, filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        allName{count} = nameCur;
        if size(image,3) == 3
            image = modcrop(image,32);
            image = rgb2ycbcr(image);
            image = image(:,:,1);
        end
        label = im2single(image);
        if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
            continue
        end
        
        input = label;
        input = gpuArray(input);
        %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        tic
        net.eval({'input', input}) ;
        time(i) = toc; 
        out1 = net.getVarIndex('prediction') ;
        output = gather(squeeze(gather(net.vars(out1).value)));
        
        %output = res(end).x;
        output = gather(output);
        input  = gather(input);
        %%% calculate PSNR and SSIM
        [PSNRCur_CSNet, SSIMCur_CSNet] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        if showResult
            %imshow(cat(2,im2uint8(label),im2uint8(output)));
            %title([filePaths(i).name,'    ',num2str(PSNRCur_CSNet,'%2.2f'),'dB','    ',num2str(SSIMCur_CSNet,'%2.4f')])
            %drawnow;
            display(['        ' filePaths(i).name,'        ',num2str(PSNRCur_CSNet,'%2.2f'),'dB','    ',num2str(SSIMCur_CSNet,'%2.3f')])
        end
        
        PSNRs_CSNet(i) = PSNRCur_CSNet;
        SSIMs_CSNet(i) = SSIMCur_CSNet;
        
        % save results for current image
        if writeRecon
            folder  = ['Results\2Image_' network ];
            if ~exist(folder), mkdir(folder); end
            fileName = [folder '\' folderTest '_' allName{count} '_subrate' num2str(samplingRate) '.png'];
            imwrite(im2uint8(output), fileName );
            
            count = count + 1;
        end
        
    end
    % save results for current image
    folder  = ['Results\1Text_' network ];
    if ~exist(folder), mkdir(folder); end
    imgName = [folderTest ];
    fileName = [folder '\' imgName '_subrate' num2str(samplingRate) '.txt'];
    write_txt(fileName, allName, samplingRate, PSNRs_CSNet, SSIMs_CSNet, time);
    
    disp(['Average, subrate ' num2str(samplingRate) ': ' num2str(mean(PSNRs_CSNet), ...
           '%2.3f') 'dB, SSIM: ', num2str(mean(SSIMs_CSNet), '%2.4f'), ', time: ', num2str(mean(time), '%2.4f')]);
end
