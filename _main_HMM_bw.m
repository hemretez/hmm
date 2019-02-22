
%-------------------------------------------------Hazar Emre Tez

%% Variables, data and model

clc; clear all; close;

numstates = 2; % number of the states
numsym = 2; % number of distinct observation symbols per state

load('data.txt');
observations = data + 1;
N = length(data);
T = length(data(1,:));

%% model
expln = ['A random initial model:\n\n'];
fprintf(expln);
a = [ 0.6311 0.3689 ;  0.4607    0.5393]
b = [ 0.8816    0.1184; 0.3564    0.6436]
pi = [ 0.7555    0.2445 ]

% Random model
% x=rand(1); x1=rand(1);
% y=rand(1); y1=rand(1);
% z=rand(1);
% a = [x 1-x;x1 1-x1] % state transition matrix (2x2)
% b = [y 1-y;y1 1-y1] % observation probability distribution (2x2)
% pi = [z 1-z] % initial probabilities (1x2)

%epsilon and gamma initialized.
eps = zeros(N,T,numstates,numstates);
gamma = zeros(N,T,numstates);

for i = 1:N;
    %Calculating the forward probability and the backward probability for
    %each sequence
    num = 0;
    denom = 0;
    sequence = observations(i,:);
    alpha = forw(numstates,a,b,pi,sequence);
    beta = backward(numstates,a,b,pi,sequence);
    gamma(i,:,:) = gma(numstates,sequence,alpha,beta);
    
    for t = 1:T;
        s = 0;
        for k = 1:numstates;
            for l = 1:numstates;
                if t == length(sequence);
                    num = alpha(t,k)*a(k, l);
                else
                    num = alpha(t,k) * a(k, l) * beta(t + 1,l) * b(l, sequence(t + 1));
                end
                eps(i,t,k,l)= num;
                s = s + num;
                denom = denom + alpha(t,k)*beta(t,k);
            end
        end
        
        if s ~= 0; % Scaling
            for k = 1:numstates;
                for l = 1:numstates;
                    eps(i,t,k,l) = eps(i,t,k,l)/s;
                end
            end
        end
    end
    
    %% Parameter re-estimation
    
    % Estimation of initial state probabilities:
    for k = 1:numstates;
        summ = 0;
        for m = 1:i;
            summ = summ + gamma(m,1,k);
        end
        pi(k) = summ/i;
    end
    
    % Estimation of transition probabilities:
    for n = 1:numstates;
        for j = 1:numsym;
            den = 0; num = 0;
            for k = 1:i;
                for m = 1:(T-1);
                    num = num + eps(k,l,n,j);
                end
                for m = 1:(T-1);
                    den = den + gamma(k,l,n);
                end
            end
            if den ~= 0;
                a(n,j) = num/den;
            else
                a(n,j) = 0;
            end
        end
    end
    
    % Estimation of emission probabilities:
    for n = 1:numstates;
        for j = 1:numsym;
            den = 0; num = 0;
            for k = 1:i;
                for m = 1:T;
                    if observations(k,m) == j;
                        num = num + gamma(k,m,n);
                    end
                end
                for m = 1:T;
                    den = den + gamma(k,m,n);
                end
            end
            
            if (den == 0 || num ==0);
                b(n,j) = 0;
            else
                b(n,j) = num/den;
            end
            
        end
    end
    
end
expln = ['Final model\n\n'];
fprintf(expln);
a
b
pi



