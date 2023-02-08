% Solves the 2 point boundary value problem y''-k^2y=f using the Chebyshev
% integration matrix. 
% BCs(1,:) = +y'(1)
% BCs(2,:) = +y(1)
% BCs(3,:) = +y'(-1)
% BCs(4,:) = -y(-1) (NOTE the change in sign)
function [secD, varargout] = BVPChebInt(k,N,SIMat,BCs,H,fhat,rbc,lbc)
    % Get the correct BCs from the precomputed values
    BCs=[BCs(2,:); -BCs(4,:)]+...
    (k~=0)*[H*BCs(1,:)+(H^2*k-1)*BCs(2,:); H*BCs(3,:)+(H^2*k+1)*BCs(4,:)];
    % Schur complement approach
    A=speye(N)-k^2*SIMat(1:N,1:N);  % sparse matrix 
    [L,U]=lu(A);     % LU factorization should be done outside time loop
    B=-k^2*SIMat(1:N,N+1:N+2);
    C=BCs(:,1:N);
    D=BCs(:,N+1:N+2);
    y=(C*(U\(L\B))-D) \ (C*(U\(L\fhat))-[rbc;lbc]); % 2 x 2 solve
    x=U\(L\(fhat-B*y)); % back substitute
    secD=[x;y];
    if nargout >= 2
        varargout{1} = A;
    end
    if nargout >= 3
        varargout{2} = B;
    end
    if nargout >= 4
        varargout{3} = C;
    end
    if nargout >= 5
        varargout{4} = D;
    end
    if nargout >= 6
        varargout{5} = inv(C*(U\(L\B))-D);
    end
end
    