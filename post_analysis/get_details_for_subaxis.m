function [ verti_length, verti_dis, hori_dis, MT, MB, ML, MR ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v,varargin )
% auto calculate the inter subaxises distance "verti_dis, hori_dis" and
% vertical length "verti_length" based on total number of rows and columns,
% horizontal length and edge space

syms x % x is the length of a subaxis
if nargin <= 9
    if nargin == 7 % if MR ML use same ratio, vargin is empty.
        MR = edge_multiplyer_h*inter_multiplyer_h*x;
        ML = edge_multiplyer_h*inter_multiplyer_h*x;
    elseif nargin > 7 %varargin{1} is the ratio for ML
        MR = edge_multiplyer_h*inter_multiplyer_h*x;
        ML = varargin{1}*inter_multiplyer_h*x;
    end
    
    
    % solve equations
    eqn1 = total_column*x + (total_column - 1)*inter_multiplyer_h*x +MR + ML == hori_length;
    solx = solve(eqn1,x);
    
    if nargin <= 8
        MT = edge_multiplyer_v*inter_multiplyer_v*solx;
        MB = edge_multiplyer_v*inter_multiplyer_v*solx;
    elseif nargin == 9 %varargin{2} is the ratio for MB
        MT = edge_multiplyer_v*inter_multiplyer_v*solx;
        MB = varargin{2}*inter_multiplyer_v*solx;
    end
    
    verti_length = double(total_row*solx + (total_row - 1)*inter_multiplyer_v*solx + MT + MB);
    verti_dis = inter_multiplyer_v*solx;
    verti_dis = double(verti_dis/verti_length);
    
    hori_dis = inter_multiplyer_h*solx;
    hori_dis = double(hori_dis/hori_length);
    
    MT = double(MT/verti_length);
    MB = double(MB/verti_length);
    if nargin == 7 % if MR ML use same ratio, vargin is empty.
        MR = double(edge_multiplyer_h*inter_multiplyer_h*solx/hori_length);
        ML = double(edge_multiplyer_h*inter_multiplyer_h*solx/hori_length);
    elseif nargin > 7 %varargin{1} is the ratio for ML
        MR = double(edge_multiplyer_h*inter_multiplyer_h*solx/hori_length);
        ML = double(varargin{1}*inter_multiplyer_h*solx/hori_length);
    end
else
    % if nargin > 9, the subaxis is not square. The ratio is
    % width:height=varargin{3}:varargin{4}. x is set as width
    ratio = varargin{4}/varargin{3};
    %varargin{1} is the ratio for ML
    MR = edge_multiplyer_h*inter_multiplyer_h*x;
    ML = varargin{1}*inter_multiplyer_h*x;
    
    % solve equations
    eqn1 = total_column*x + (total_column - 1)*inter_multiplyer_h*x + MR + ML == hori_length;
    solx = solve(eqn1,x);
    
    
    MT = edge_multiplyer_v*inter_multiplyer_v*solx*ratio;
    MB = varargin{2}*inter_multiplyer_v*solx*ratio;
    
    
    verti_length = double(total_row*solx*ratio + (total_row - 1)*inter_multiplyer_v*solx*ratio + MT + MB);
    verti_dis = inter_multiplyer_v*solx*ratio;
    verti_dis = double(verti_dis/verti_length);
    
    hori_dis = inter_multiplyer_h*solx;
    hori_dis = double(hori_dis/hori_length);
    
    MT = double(MT/verti_length);
    MB = double(MB/verti_length);
    
    MR = double(edge_multiplyer_h*inter_multiplyer_h*solx/hori_length);
    ML = double(varargin{1}*inter_multiplyer_h*solx/hori_length);
    
end
end

