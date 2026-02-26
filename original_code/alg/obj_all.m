function [ obj ] = obj_all( A,X,G,F,W,V,alpha,Lambda1,Lambda2,lambda,mu,k,m )
%OBJ_ALL 此处显示有关此函数的摘要
%   此处显示详细说明

[n,~]=size(X);
AX=X;
for i=1:k
    AX=A*AX;
end

obj1=0;
for j=1:m
    obj1=obj1+sum(sum((AX-G{j}*F{j}).^2));
end

GG=zeros(n,n);
for j=1:m
    GG=GG+alpha(j).*G{j}*G{j}';
end
obj2=lambda.*sum(sum((W-GG).^2));

dd=1./sqrt(max(sum(W,2),eps));
DWD=bsxfun(@times,W,dd);
DWD=bsxfun(@times,DWD,dd');
obj3=sum(sum(Lambda1'.*(A-0.5.*eye(n)-0.5.*DWD)));
obj4=sum(sum(Lambda2'.*(W-V)));

obj5=mu./2.*sum(sum((A-0.5.*eye(n)-0.5.*DWD).^2));
obj6=mu./2.*sum(sum((W-V).^2));

obj=obj1+obj2+obj3+obj4+obj5+obj6;

