function [ gama ] = gma( nstates,obsseq,alp,bet )
% GAMMA is the P of being is state Si at time t, given observation sequence and the model
sum = 0;
for t = 1:length(obsseq);
    sum = 0;
    for i = 1:nstates;
        gama(t,i) = alp(t,i)*bet(t,i);
        sum = sum + gama(t,i);
    end
    
    if (sum~=0);
        for k = 1:nstates;
            gama(t,k) = gama(t,k)/sum;
        end
    end
end

end
