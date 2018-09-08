function net = addReshapeConcat(net)
%ADDCUSTOMLOSSLAYER Add a custom loss layer to a network
%   NET = ADDCUSTOMLOSSLAYER(NET, FWDFUN, BWDFUN) adds a custom loss
%   layer to the network NET using FWDFUN for forward pass and BWDFUN for
%   a backward pass.

layer.name = 'ReshapeConcat' ;
layer.type = 'custom' ;
layer.forward = @bcs_initialRec_wzhshi_dag ;
layer.backward = @bcs_initialRec_wzhshi_dag ;
layer.class = [] ;

% Make sure that the loss layer is not added multiple times
if strcmp(net.layers{end}.name, layer.name)
  net.layers{end} = layer ;
else
  net.layers{end+1} = layer ;
end

  function res_ =  forward(layer, res, res_)
    res_.x = fwfun(res.x, layer.class) ;
  end

  function res = backward(layer, res, res_)
    res.dzdx = bwfun(res.x, layer.class, res_.dzdx) ;
  end
end