function lambda = Intensity_HP(t, History, para) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compute the intensity functions of Hawkes processes
%
% Parameters of Hawkes processes
% para.mu: base exogenous intensity
% para.A: coefficients of impact function
% para.kernel: 'exp', 'gauss'
% para.w: bandwith of kernel
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = para.mu(:);

if ~isempty(History)
    
    Time = History(1, :);
    index = Time<=t;
    Time = Time(index);
    Event = History(2, index);
    
    A = para.A(Event, :, :);
    basis = Kernel(t- Time(:), para);   
    
    if para.Sentiment
        Sentiments = History(3,index);
    end
    
    for c = 1:size(para.A, 3)
        if para.Sentiment
            s = exp(-mean(abs(Sentiments(end) - Sentiments(1:end-1))));
            lambda(c) = lambda(c) + sum(sum(basis.*(A(:,:,c))*s));
        else
            lambda(c) = lambda(c) + sum(sum(basis.*(A(:,:,c))));
        end
    end
end

lambda = lambda.*double(lambda>0);
end




