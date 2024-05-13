function [A,S,iter,SSE] = mcr_pca(T,P,A,S,est1st,thresh,maxit,afix,sfix,...
    anls,snls,snorm)
% Multivariate curve resolution (MCR) using PCA scores & loadings of data
% as inputs. Employs rigorous least squares equality and inequality 
% constraints for all elements in the solution factor matrices.
%
% I/O:  [A,S,iter,SSE] = mcr_pca(T,P,A,S,est1st,thresh,maxit,afix,sfix,anls,snls,snorm)
%       [A,S] = mcr_pca(T,P,A,S);
%
%       INPUTS:
%       T:   m x q matrix of abundance scores
%       P:   q x n matrix of spectral loadings
%       A:   m x p matrix of initial estimate for abundance factors
%       S:   p x n matrix of initial estimate for spectral factors
%            One may enter an empty matrix for either A or S if desired.
%
%       Optional Inputs:
%       est1st: Choice of which mode, spectral (1) or abundance (2) to 
%            estimate first.  If you have a better initial estimate for 
%            spectra, say, you may want to estimate abundances as a first 
%            step in MCR. Default is spectra. If abundance is selected, the
%            first iteration will skip spectral mode.
%
%       thresh: Termination SSE difference between consecutive iterations;
%            e.g. 1e-6, no default value.
%
%       maxit: Termination maximum number of iterations. Default 1e3. Max 1e6.
%
%       afix: the indices of abundance matrix, A, that remain fixed during
%            mcr using method of direct elimination.
%         There are two ways to enter afix.
%         Method 1: Enter afix as a column of logicals (T/F) of length p.
%                 Use true where there is an equality constraint on the 
%                 input abundance factors (A) and false when there is none.
%                 Use this method only when entire abundance factors are 
%                 equality constrained, that is do not change during mcr. 
%                 These are multiplied by the corresponding spectra and 
%                 subtracted from the data after each iteration.
%         Method 2: Enter afix as a matrix of logicals (T/F) of size A.
%                 Use true where there is an equality constraint on the 
%                 input abundance factors (A) and false when there is none.
%                 This method may be use when entire factors (all variables) 
%                 are equality constrained or when abundances in some but 
%                 not all variables are constrained. These are multiplied by 
%                 the corresponding spectra and subtracted from the data 
%                 after each iteration.
%
%       sfix: the indices of spectral matrix, S, that remain fixed during 
%            mcr using method of direct elimination.
%         There are two ways to enter sfix.
%         Method 1: Enter sfix as a column of logicals (T/F) of length p.
%                 Use true where there is an equality constraint on the 
%                 input spectral factors (S) and false when there is none.
%                 Use this method only when entire spectral factors are 
%                 equality constrained, that is do not change during mcr. 
%                 These are multiplied by the corresponding abundances and 
%                 subtracted from the data after each iteration.
%         Method 2: Enter sfix as a matrix of logicals (T/F) of size S.
%                 Use true where there is an equality constraint on the 
%                 input spectral factors (S) and false when there is none.
%                 This method may be use when entire factors (all wavelengths)
%                 are equality constrained or when intensities in some but 
%                 not all wavelengths are constrained. These are multiplied
%                 by the corresponding abundances and subtracted from 
%                 the data after each iteration.
%
%       anls: the indices of abundance matrix (A) that are nonnegativity 
%            constrained during mcr using fast combinatorial NNLS.
%         There are two ways to enter anls.
%         Method 1: Enter anls as a column of logicals (T/F) of length p.
%                 Use true where there is a nonnegativity constraint and 
%                 false when there is none.
%                 Use this method only when entire abundance factors are 
%                 nonnegativity constrained.
%         Method 2: Enter anls as a matrix of logicals (T/F) of size A.
%                 Use true where there is a nonnegativity constraint and 
%                 false when there is none.
%                 This method may be use when entire factors (all variables) 
%                 are nonnegativity constrained or when abundances in 
%                 some but not all variables are nonnegativity constrained.
%
%       snls: the indices of spectral matrix (S) that are nonnegativity 
%            constrained during mcr using fast combinatorial NNLS.
%         There are two ways to enter snls.
%         Method 1: Enter snls as a column of logicals (T/F) of length p.
%                 Use true where there is a nonnegativity constraint and 
%                 false when there is none.
%                 Use this method only when entire spectral factors are 
%                 nonnegativity constrained.
%         Method 2: Enter snls as a matrix of logicals (T/F) of size S.
%                 Use true where there is a nonnegativity constraint and 
%                 false when there is none.
%                 This method may be use when entire factors (all wavelengths) 
%                 are nonnegativity constrained or when intensities in some 
%                 but not all wavelengths are constrained.
%
%       snorm: normalize spectral factors to unit length during MCR (T/F)
%            Default: true. Overridden if factors are equality constrained.
%
%       OUTPUTS:
%       A:    Abundance MCR factors
%       S:    Spectral MCR factors
%       iter: Number of iterations performed
%       SSE:  Sum of squared residuals at each iteration
%
% Copyright 2015 National Technology & Engineering Solutions of Sandia, 
% LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the 
% U.S. Government retains certain rights in this software.

% Mark H. Van Benthem, Sandia National Laboratories, 9/12/2002
% Revised: 7/31/2008, 12/02/2008, 02/22/2015, 05/14/2019, 08/27/2019


% check input arguments
narginchk(4,12)
[rT,qT] = size(T);
[qP,cP] = size(P);
if nargin<5 || isempty(est1st)
    est1st = 1;
end
if isempty(A) && isempty(S)
    error('Need initial estimates for A or S')
elseif isempty(S)
    S = zeros(size(A,2),cP);
    est1st = 1;
elseif isempty(A)
    A = zeros(rT,size(S,1));
    est1st = 2;
end
rmcr = size(A,2); % number of MCR factors
bitvec = 2.^(rmcr-1:-1:0);
% All of the preliminary stuff
if qP ~= qT % number of PCA factors
    error('Number of columns in T must be the same as rows in P')
elseif rmcr ~= size(S,1)
    error('Number of columns in A must be the same as rows in S')
elseif rT ~= size(A,1)
    error('Number of rows in A must be the same as rows in T')
elseif cP ~= size(S,2)
    error('Number of columns in S must be the same as columns in P')
end
if nargin<6 || isempty(thresh)
    thresh = NaN;
end
if (nargin<7 || isempty(maxit)) && isnan(thresh)
    maxit = 1e3;
elseif nargin<7 || isempty(maxit) || maxit>1e6
    maxit = 1e6;
end
maxit = max(round(maxit,0),1); % ensure whole number and 1 iteration
% abundance factors that remain fixed (augmented variables)
if nargin<8 || isempty(afix) || all(all(~afix))
    aff = false;
    afix = false(rmcr,1);
    nafix = 1;
    afixidx = 1:rT;
else
    Sizafix = size(afix);
    % determine how to break up the constrained data
    aff = true;
    afix = sparse(logical(afix));
    if min(Sizafix) == 1
        if max(Sizafix) ~= rmcr
            error('Length of vector afix must be equal to columns in A')
        end
        afix = afix(:);
        nafix = 1;
        afixidx = 1:rT;
        afixbps = [0 rT];
    else
        if any(Sizafix ~= size(A))
            error('Size of afix must be equal to the size of A')
        end
        afix = transpose(afix);
        [cfixbns,afixidx] = sort(bitvec*afix);
        afixbps = [find(diff(cfixbns)) rT];
        afix = afix(:,afixidx(afixbps));
        nafix = length(afixbps);
        afixbps = [0 afixbps];
    end
end
afix = logical(full(afix));
afree = ~afix;
% spectral factors that remain fixed (augmented variables)
if nargin<9 || isempty(sfix) || all(all(~sfix))
    sff = false;
    sfix = false(rmcr,1);
    nsfix = 1;
    sfixidx = 1:cP;
else
    Sizsfix = size(sfix);
    sff = true;
    sfix = sparse(logical(sfix));
    if min(Sizsfix) == 1
        if max(Sizsfix) ~= rmcr
            error('Length of vector sfix must be equal to columns in S')
        end
        sfix = sfix(:);
        nsfix = 1;
        sfixidx = 1:cP;
        sfixbps = [0 cP];
    else
        if any(Sizsfix ~= size(S))
            error('Size of sfix must be equal to the size of S')
        end
        [sfixbns,sfixidx] = sort(bitvec*sfix);
        sfixbps = [find(diff(sfixbns)) cP];
        sfix = sfix(:,sfixidx(sfixbps));
        nsfix = length(sfixbps);
        sfixbps = [0 sfixbps];
    end
end
sfix = logical(full(sfix));
sfree = ~sfix;
if nargin<10 || isempty(anls)
    ann = false;
    anls  = false(rmcr,rT);
else
    ann = true;
    Sizanls = size(anls);
    anls = transpose(~anls);
    if min(Sizanls) == 1
        if max(Sizanls) ~= rmcr
            error('Length of vector anls must be equal to columns in A')
        end
        anls = repmat(anls(:),1,rT);
    elseif any(Sizanls ~= size(A))
        error('Size of anls must be equal to the size of A')
    end
end
if nargin<11 || isempty(snls)
    snn = false;
    snls  = false(rmcr,cP);
else
    snn = true;
    Sizsals = size(snls);
    if min(Sizsals) == 1
        if max(Sizsals) ~= rmcr
            error('Length of vector snls must be equal to columns in S')
        end
        snls = repmat(snls(:),1,cP);
    elseif any(Sizsals ~= size(S))
        error('Size of snls must be equal to the size of S')
    end
    snls = ~snls;
end
if nargin < 12
    snorm = true;
end
% set up the starting conditions for the fast implemenation of NNLS
Pc   = zeros(rmcr,rT);
Ps   = zeros(rmcr,cP);
normind = all(sfree,2)&all(afree,2);
if snorm
    S(normind,:) = factor_norm(S(normind,:),2);
end
SST = sum(sum(T.^2).*sum(P'.^2)); % total sum of squares (for error calc.)
SSF = 0;
SSE = zeros(1,maxit); % sum of squared errors for each iteration
% redesign the display text string add next three lines 20150222 mvb
nitstr = floor(log10(maxit))+1;
outstr = sprintf('%%%dd, RMSE:%%11.4e\\n',nitstr);
backstr = repmat('\b',1,nitstr+19);
for iter = 1:maxit
    % least squares solution for spectral factors
    if iter==1 && est1st~=1 % skip spectral estimate on first iteration
        disp('Estimate Abundance Factors First')
    else % perform spectral estimation on first iteration
    if iter==1
        disp('Estimate Spectral Factors First')
        AtA = A'*A;
        ATP = (A'*T)*P;
    end
    for ii = 1:nsfix
        if sff
            idx = sfixidx(sfixbps(ii)+1:sfixbps(ii+1));
            Zidx = ATP(sfree(:,ii),idx) - AtA(sfree(:,ii),sfix(:,ii))*...
                S(sfix(:,ii),idx);
        else
            idx = sfixidx;
            Zidx = ATP(sfree(:,ii),:);
        end
        if snn
            Sidx = S(sfree(:,ii),idx);
            [Sidx,Ps(sfree(:,ii),idx)] = ...
                fcnnls(AtA(sfree(:,ii),sfree(:,ii)),Zidx,...
                Sidx,Ps(sfree(:,ii),idx),snls(sfree(:,ii),idx),0);
        else
            Sidx = AtA(sfree(:,ii),sfree(:,ii))\Zidx;
        end
        S(sfree(:,ii),idx) = Sidx;
    end
    end
    if snorm
        S(normind,:) = factor_norm(S(normind,:),2);
    end
    % least squares solution for abundance factors
    StS = S*S';
    TPS = T*(P*S');
    for ii = 1:nafix
        if aff
            idx = afixidx(afixbps(ii)+1:afixbps(ii+1));
            Zidx = TPS(idx,afree(:,ii)) - A(idx,afix(:,ii))*...
                StS(afix(:,ii),afree(:,ii));
            Zidx = Zidx';
            SSF = SSF + trace((A(idx,afix(:,ii))'*A(idx,afix(:,ii)))*...
                StS(afix(:,ii),afix(:,ii)))-...
                2.*trace(A(idx,afix(:,ii))'*TPS(idx,afix(:,ii)));
       else
            idx = afixidx;
            Zidx = TPS(:,afree(:,ii))';
        end
        if ann
            Aidx = A(idx,afree(:,ii))';
            [Aidx,Pc(afree(:,ii),idx),SSR] = ...
                fcnnls(StS(afree(:,ii),afree(:,ii)),Zidx,...
                Aidx,Pc(afree(:,ii),idx),anls(afree(:,ii),idx),0);
            A(idx,afree(:,ii)) = Aidx';
            SSF = SSF - SSR;
        else
            A(idx,afree(:,ii)) = (StS(afree(:,ii),afree(:,ii))\Zidx)';
        end
    end
    AtA = A'*A;
    ATP = (A'*T)*P;
    % Compute errors and check convergence
    if ann
        SSE(iter) = SST + SSF;
        SSF = 0;
    else
        SSE(iter) = SST - 2.*trace(ATP*S') + trace(AtA*StS);
    end
    if isnan(SSE(iter))
        disp('No feasible solution found, change starting points.')
        break % exit gracefully if no feasible solution will be found
    end
    % reformat the display text next two lines 20150222 mvb
    if iter==1, fprintf(['Iteration: ',outstr],0,0); end
    fprintf(backstr);fprintf(outstr,iter,SSE(iter));
    % if using SSE difference, make that computation
    if ~isnan(thresh) && (iter>1)
        itdiff = abs(SSE(iter)-SSE(iter-1));
        if itdiff<thresh
            break
        end
    end
end
SSE = SSE(1:iter);

function factors = factor_norm(factors,dim)
% FACTOR_NORM  Column or row normalization of matrix or vector

if min(size(factors))==1
    normf = sqrt(sum(factors.^2));
    factors = factors./normf;
elseif dim==2
    normf = sqrt(sum(factors.^2,2));
    factors = diag(sparse(1./normf)) * factors;
else
    normf = sqrt(sum(factors.^2,1));
    factors = factors * diag(sparse(1./normf));
end

