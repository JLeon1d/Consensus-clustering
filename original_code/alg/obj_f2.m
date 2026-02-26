function [ obj,grad ] = obj_f2( A,G,V,Lambda1,Lambda2,mu,alpha,lambda,m,n,x )
%OBJ_F2 此处显示有关此函数的摘要
%   此处显示详细说明
W=reshape(x,n,n);
GG=zeros(n,n);
for v=1:m
    GG=GG+alpha(v).*G{v}*G{v}';
end
obj1=lambda.*sum(sum((W-GG).^2));
dd=1./sqrt(max(sum(W,2),eps));
DWD1=bsxfun(@times,W,dd);
DWD2=bsxfun(@times,W,dd');
DWD=bsxfun(@times,DWD1,dd');
obj2=-sum(sum(Lambda1.*DWD))./2;
obj3=sum(sum(Lambda2.*W));
obj4=mu./2.*sum(sum((A-0.5.*(eye(n)+DWD)).^2));
obj5=mu./2.*sum(sum((W-V).^2));

obj=obj1+obj2+obj3+obj4+obj5;

grad1=2.*lambda.*(W-GG);
grad21=bsxfun(@times,Lambda1,dd);
grad21=bsxfun(@times,grad21,dd');
grad21=-0.5.*grad21;

tmp1=sum(Lambda1.*DWD1)'+ sum(Lambda1.*DWD2,2);
tmp1=tmp1.*dd.^3;
grad2=bsxfun(@plus,grad21,0.25.*tmp1);

grad5=mu.*(W-V);

dd2=1./max(sum(W,2),eps);
DWD21=bsxfun(@times,W,dd2);
DWD22=bsxfun(@times,W,dd2');
DWD23=bsxfun(@times,DWD21,dd2');
tmp2=(sum(DWD21.*W)'+sum(DWD22.*W,2)).*(dd2.^2);
grad41=bsxfun(@minus,2.*DWD23,tmp2);

B=(A-eye(n)./2);
grad421=bsxfun(@times,B,dd);
grad421=bsxfun(@times,grad421,dd');

tmp3=sum(B.*DWD1)'+ sum(B.*DWD2,2);
tmp3=tmp3.*dd.^3;
grad42=bsxfun(@minus,grad421,tmp3.*0.5);
grad4=mu./8.*grad41-mu./2.*grad42;
grad=grad1+grad2+Lambda2+grad4+grad5;

grad=reshape(grad,n*n,1);
end

