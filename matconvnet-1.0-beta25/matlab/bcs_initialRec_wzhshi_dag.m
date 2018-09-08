function y = bcs_initialRec_wzhshi_dag( x, dims, dzdy )
% global dims
%BCS_INITIALREC_WZHSHI Summary of this function goes here
%   Detailed explanation goes here
%Author: Wuzhen Shi
%Email: wzhshi@hit.edu.cn
%School of Computer Science of Technology, Harbin Institute of Technology
x  = x{1, 1};
sz = size(x) ;

if nargin <= 2 || isempty(dzdy)
    dims = horzcat(dims, 1) ;
%     dims = horzcat(dims, size(x,4)) ;
    y = [];
    for i=1:sz(1)
        temp = [];
        for j=1:sz(2)
          temp = [temp vl_nnreshape_wzhshi(x(i,j,:,:),dims)];         
        end
        y = [y;temp];
    end
else
    y0 = x;
    for i=1:sz(1)
        for j = 1:sz(2)
            y0(i,j,:,:) = vl_nnreshape_wzhshi(dzdy((i-1)*dims(1)+1 : i*dims(1),...
                (j-1)*dims(2)+1 : j*dims(2) , 1 , :), [1 1 dims(1)*dims(2)]);
                        
        end
    end
    y = cell(1, 1); 
    y{1, 1} = y0; 
end



