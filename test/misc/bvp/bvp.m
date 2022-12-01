clear all;
addpath('matlab')

f=importdata('data.main'); s=[f{:}];
look="k";
k=sscanf(s((regexp(s, [look," "])):end),[look," %g"])
look="H";
H=sscanf(s((regexp(s, [look," "])):end),[look," %g"])
look="nz";
Nz=sscanf(s((regexp(s, [look," "])):end),[look," %g"])
look="alpha";
alpha=sscanf(s((regexp(s, [look," "])):end),[look," %g"])
look="beta";
beta=sscanf(s((regexp(s, [look," "])):end),[look," %g"])
SIMat = H^2*secondIntegralMatrix(Nz);
BCs = BCRows(Nz);
fhat=load('fn.dat');
res = SIMat(1:Nz,:)*BVPChebInt(k,Nz,SIMat,BCs,H,fhat,alpha,beta);
disp('OUTPUT BEGIN')
format longG
disp(res)
