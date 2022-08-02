% THE MATRIX FOR THE SECOND INTEGRAL
function SIMat = secondIntegralMatrix(N)
    jj=(3:N-1)';
    colm2=[0; 0; 0.25; 1./((2*jj).*(2*jj-2))];
    colp2=[0; 0.125; 1/24; 1./((2*jj).*(2*jj+2)).*(jj<N-2)];
    col0=[0; -0.125; -1/8-1/24; -1./((2*jj).*(2*jj-2))-1./((2*jj).*(2*jj+2)).*(jj<N-1)];
    SIMat=spdiags([colm2 col0 colp2],[-2 0 2],N,N+2);
    SIMat(1,N+1)=1; SIMat(2,N+2)=1;
end
