function [ beta ] = backward( nstates,A,B,Pi,obsseq )
% backward parameter

% initialization:
beta=[]; % backward parameter
for i = 1:nstates;
    beta(i,length(obsseq)) = 1;
end
% induction
for t = (length(obsseq)-1):-1:1;
    for i = 1:nstates;
        beta(i,t) = 0;
        for j = 1:nstates;
            beta(i,t) = beta(i,t)+(beta(j,t+1)*A(i,j)*B(j,obsseq(t+1)));
        end
    end
end
beta=beta';
end

