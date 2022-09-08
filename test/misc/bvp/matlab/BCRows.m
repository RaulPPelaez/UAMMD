% This function gives you the following BCs:
% BCR1 = first derivative (first integral) evaluated at x=1
% BCR2 = function (second integral) evaluated at x=1
% BCL1 = first derivative (first integral) evaluated at x=-1
% BCL2 = NEGATIVE OF function (second integral) evaluated at x=-1
function BCs = BCRows(N)
    BCR1=zeros(1,N+2);
    BCR2=zeros(1,N+2);
    BCL1=zeros(1,N+2);
    BCL2=zeros(1,N+2);
    % Special cases - right
    BCR1(N+2)=1; BCR2(N+1)=1; BCR1(1)=1; BCR1(3)=-1/2;
    BCR2(N+2)=BCR2(N+2)+1; BCR2(2)=-1/8; BCR2(4)=1/8;
    BCR1(2)=BCR1(2)+1/4; BCR1(4)=BCR1(4)-1/4; BCR2(1)=BCR2(1)+1/4; BCR2(3)=BCR2(3)-1/8-1/24;
    BCR2(5)=BCR2(5)+1/24;
    % Special cases - left
    BCL1(N+2)=1; BCL2(N+1)=-1; BCL1(1)=-1; BCL1(3)=1/2;
    BCL2(N+2)=BCL2(N+2)+1; BCL2(2)=-1/8; BCL2(4)=1/8;
    BCL1(2)=BCL1(2)+1/4; BCL1(4)=BCL1(4)-1/4; BCL2(1)=BCL2(1)-1/4; BCL2(3)=BCL2(3)+1/8+1/24;
    BCL2(5)=BCL2(5)-1/24;
    % Easy cases
    jj=(3:N-1);
    BCR1(jj)=BCR1(jj)+1./(2*jj); 
    BCL1(jj)=BCL1(jj)+(-1).^jj./(2*jj);
    BCR1(jj+2)=BCR1(jj+2)-1./(2*jj).*(jj<N-1); 
    BCL1(jj+2)=BCL1(jj+2)-(-1).^jj./(2*jj).*(jj<N-1);
    BCR2(jj-1)=BCR2(jj-1)+1./(2*jj).*1./(2*jj-2); 
    BCL2(jj-1)=BCL2(jj-1)-1./(2*jj).*1./(2*jj-2).*(-1).^jj;
    BCR2(jj+3)=BCR2(jj+3)+1./(2*jj).*1./(2*jj+2).*(jj<N-2); 
    BCL2(jj+3)=BCL2(jj+3)-1./(2*jj).*1./(2*jj+2).*(-1).^jj.*(jj<N-2);
    BCR2(jj+1)=BCR2(jj+1)-1./(2*jj).*1./(2*jj-2)-1./(2*jj).*1./(2*jj+2).*(jj<N-1);
    BCL2(jj+1)=BCL2(jj+1)+(1./(2*jj).*1./(2*jj-2)+1./(2*jj).*1./(2*jj+2).*(jj<N-1)).*(-1).^jj;
    BCs=[BCR1; BCR2; BCL1; BCL2];
end