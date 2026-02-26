clear;
addpath('./alg/')
addpath('./alg/L-BFGS-B-C-master/Matlab/')
addpath('./alg/L-BFGS-B-C-master/src/')
addpath('./util/')
addpath('./util2/')
name='orlraws10P_uni';
m=10;

load(['./' name '.mat']);
c=length(unique(y));
result_sc=[];
result_km=[];
k=3;    
for t=1:10
    load(['./base_result/' name '/' name '_' num2str(t) '.mat']);
    for p1=1:7
        lambda=10^(p1-4);
        [p1,t]
        [label1,label2]=ACMK( X,m,c,lambda,k,W,G,F );

        res=ClusteringMeasure(y,label1)
        result_sc.acc(p1,t)=res(1);
        result_sc.nmi(p1,t)=res(2);
            

        res=ClusteringMeasure(y,label2)
        result_km.acc(p1,t)=res(1);
        result_km.nmi(p1,t)=res(2);
            
        save(['./result_our/' name],'result_km','result_sc');
    end
%    
end