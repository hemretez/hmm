function [ alpha ] = forw( nstates,A,B,Pi,obsseq )
%calculates forwards variable
scale = []; scale(1) = 0;

% initialization:
alpha = []; % forward variable
for i = 1:nstates;
    alpha(i,1) = Pi(i)*B(i,obsseq(1));
    scale(1) = scale(1)+alpha(i,1);
end

% to scale:
for i = 1:nstates;
    alpha(i,1) = alpha(i,1)/scale(1);
end

% induction:

for t = 2:(length(obsseq));
    scale(t) = 0;
    for j = 1:nstates;
        par = 0;
        alpha(j,t) = 0;
        for i = 1:nstates;
            alpha(j,t) = alpha(j,t) + alpha(i,t-1)*A(i,j);
        end
        alpha(j,t) = alpha(j,t)*B(obsseq(t),j);
        scale(t) = scale(t)+alpha(j,t);
    end
    
   % end scaling
   for j = 1:nstates;
       alpha(j,t) = alpha(j,t)/scale(t);
   end
end
alpha=alpha';
end


% 
% 
% function [ alpha ] = forw( nstates,A,B,Pi,obsseq )
% %calculates forwards variable
% scale = []; scale(1) = 0;
% 
% % initialization:
% alpha = []; % forward variable
% for i = 1:nstates;
%     alpha(i,1) = Pi(i)*B(i,obsseq(1));
%     scale(1) = scale(1)+alpha(i,1);
% end
% 
% % to scale:
% for i = 1:nstates;
%     alpha(i,1) = alpha(i,1)/scale(1);
% end
% 
% % induction:
% 
% for t = 1:(length(obsseq)-1);
%     scale(t+1) = 0;
%     for j = 1:nstates;
%         par = 0;
%         %alpha(j,t+1) = 0;
%         for i = 1:nstates;
%             par = par + alpha(i,t)*A(i,j);
%         end
%         alpha(j,t+1) = par*B(obsseq(t+1),j);
%         scale(t+1) = scale(t+1)+alpha(j,t+1);
%     end
%     
%    % end scaling
%    for j = 1:nstates;
%        alpha(j,t+1) = alpha(j,t+1)/scale(t+1);
%    end
% end
% alpha=alpha';
% end

