function write_txt(fileName, imgName, subRate, PSNRCur, SSIMCur, time )

fileID = fopen(fileName,'w');
fprintf(fileID,'		Img    	   rate    	 PSNR         SSIM\n');
for i = 1:1:length(PSNRCur)
    fprintf(fileID,'%13s \t %6.3f \t %6.3f \t %6.4f \t %6.4f \n', imgName{i}, subRate, PSNRCur(i), SSIMCur(i),time(i));
end

fprintf(fileID,'		Avg      %6.3f \t %6.3f \t %6.4f \t %6.4f \n', subRate, mean(PSNRCur), mean(SSIMCur),mean(time));

fclose(fileID);