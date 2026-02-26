function [ label1,label2,W,G,F,alpha,obj ] = ACMK( X,m,c,lambda,k,W,G,F )
%OPTIMIZE 此处显示有关此函数的摘要
%   此处显示详细说明

[n,d]=size(X);
alpha=ones(1,m)./m;
maxiter=20;
V=W;
dd=1./sqrt(max(sum(W,2),eps));
DWD=bsxfun(@times,W,dd);
DWD=bsxfun(@times,DWD,dd');
A=(eye(n)+DWD)./2;
Lambda1=zeros(n,n);
Lambda2=zeros(n,n);
mu=1;
rho=1.05;
opt=[];
opt.Display='off';
obj=[];
GF=cell(1,m);

for i=1:maxiter
    GF_all=zeros(d,n);
    for j=1:m
        GF{j}=G{j}*F{j};
        GF_all=GF_all+GF{j}';
    end

    A=LBFGSB1(A,DWD,GF,GF_all,X,Lambda1,mu,k,m );

    W=LBFGSB2(W,A,G,V,Lambda1,Lambda2,mu,alpha,lambda,m );

    dd=1./sqrt(max(sum(W,2),eps));
    DWD=bsxfun(@times,W,dd);
    DWD=bsxfun(@times,DWD,dd');
    AX=X;
    for j=1:k
        AX=A*AX;
    end

    G=optimize_G(AX,G,F,W,alpha,lambda);

    for j=1:m
        GG=G{j}'*G{j};
        gg=1./max(diag(GG),eps);
        GX=G{j}'*AX;
        F{j}=bsxfun(@times,GX,gg);
    end

    V=W+Lambda2./mu;
    V=(V+V')./2;

    H=zeros(m,m);
    for j=1:m
        for p=1:m
            tmp=G{j}'*G{p};
            tmp=tmp*G{p}';
            H(j,p)=sum(sum(G{j}'.*tmp));
        end
    end
    H=H+H';
    
    f=zeros(m,1);
    for j=1:m
        tmp2=W'*G{j};
        f(j)=sum(sum(tmp2.*G{j}));
    end
    f=-2.*f;
    alpha=quadprog(H,f,[],[],ones(1,m),1,zeros(m,1),ones(m,1),alpha,opt);

    
    Lambda1=Lambda1+mu.*(A-0.5.*(eye(n)+DWD));
    Lambda2=Lambda2+mu.*(W-V);
    mu=mu.*rho;
end
    
L=(eye(n)-A).*2;
L=(L+L')./2;
eigvec = eig1(L, c+1, 0);
eigvec(:,1)=[];
Y=discretisation(eigvec);
[label1,~]=find(Y');

label2=litekmeans(AX,c,'Replicates',20);
    


end

