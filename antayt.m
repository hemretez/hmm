% COMP 437 Assignment 4
%-------------------------------------------------Hazar Emre Tez

%% Variables, data and model

clc; clear all; close;

numstates = 2; % number of the states
numsym = 2; % number of distinct observation symbols per state

load('data.txt');
observations = data + 1;
iteration = 1;
N = length(data);
T = length(data(1,:));

%% model
% load('mymodel.txt'); % load created model [a][b][pi]
% a = mymodel(1:2,:);  % state transition matrix (2x2)
% b = mymodel(3:4,:);  % observation probability distribution (2x2)
% pi = mymodel(5,:); % initial probabilities (1x2)

a = [0.6000 0.4000; 0.1000 0.9000]
b = [0.2000 0.8000; 0.5500 0.4500]
pi = [0.4000 0.6000]
% a=[0.950 0.050 ; 0.100 0.900];
% b=[0.950 0.050 ; 0.200 0.800];
% pi=[0.950 0.050];

% x=rand(1); x1=rand(1);
% y=rand(1); y1=rand(1);
% z=rand(1);
% a = [x 1-x;x1 1-x1] % state transition matrix (2x2)
% b = [y 1-y;y1 1-y1] % observation probability distribution (2x2)
% pi = [z 1-z] % initial probabilities (1x2)

%epsilon and gamma initialized.
%eps = zeros(N,T,numstates,numstates);
gamma = zeros(N,T,numstates);

scaling = zeros;
newLikelihood = [];
num = 0;
denom = 0;

for i = 1:N;
    %Calculating the forward probability and the backward probability for
    %each sequence
    sequence = observations(i,:);
    alpha = forw(numstates,a,b,pi,sequence);
    beta = backward(numstates,a,b,pi,sequence);
    
    gamma(i,:,:) = gmm2(numstates,sequence,alpha,beta);
    
    for k = 1:numstates;
        pi(k)=0;
    end
    for k = 1:numstates;
        summ = 0;
        for l = 1:i;
            summ = summ + gamma(l,1,k);
        end
        pi(k) = summ/N;
    end
    
    % Estimation of transition probabilities:
    for k = 1:numstates;
        for l = 1:numsym;
            num = 0; den = 0;
            for n = 1:i;
                snum = 0; sden = 0;
                sequ = observations(n,:);
                for m = 1:(T-1);
                    snum = snum + alpha(m,k)*a(k,l)*b(l,sequ(m+1))*beta((m+1),l);
                    sden = sden + alpha(m,k)*beta(m,k);
                end
                num = num + snum;
                den = den + sden;
                
            end
            if (den ~= 0 || num ~=0);
                a(k,l) = num/den;
            else
                a(k,l) = a(k,l);
            end
        end
    end
    
    
    %  Estimation of emission probabilities:
    for k = 1:numstates;
        for l = 1:numsym;
            den = 0; num = 0;
            for m = 1:i;
                snum = 0; sden = 0;
                sequ = observations(m,:);
                for n = 1:T;
                    if sequence(n) == l;
                        num = num + alpha(n,k)*beta(n,k);
                    end
                    den = den + alpha(n,k)*beta(n,k);
                end
                num = num + snum;
                den = den + sden;
                
            end
            
            if den == 0;
                b(k,l) = 0;
            else
                b(k,l) = num/den;
            end
            
        end
    end
%     % Estimation of emission probabilities:
%     for n = 1:numstates;
%         for j = 1:numsym;
%             den = 0; num = 0;
%             for k = 1:i;
%                 
%                 for m = 1:T;
%                     if observations(k,m) == j;
%                         num = num + gamma(k,m,n);
%                     end
%                 end
%                 for m = 1:T;
%                     den = den + gamma(k,m,n);
%                 end
%             end
%             
%             if num == 0;
%                 b(n,j) = 0;
%             else
%                 b(n,j) = num/den;
%             end
%             
%         end
%     end
    
end




%% Check if the model has converged or we should stop. if not converged:
% Parameter re-estimation






a
b
pi



