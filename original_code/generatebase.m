clear;
rand('state',0);
name='orlraws10P_uni'
m=10;
load(['./'  name '.mat']);
c=length(unique(y));
[n,~]=size(X);
mkdir(['./base_result/' name]);
for t=1:10
    W=zeros(n,n);
    label_all=zeros(n,m);
    km_result=[];
    F=cell(1,m);
    G=cell(1,m);
    for iter=1:m
        [label_ini,center]=litekmeans(X,c);
        F{iter}=center;
        GG=sparse(n,c);
        for j=1:n
            GG(j,label_ini(j))=1;
        end
        G{iter}=GG;
        W=W+GG*GG';
        res=ClusteringMeasure(y,label_ini)
        km_result.acc(iter)=res(1);
        km_result.nmi(iter)=res(2);
        label_all(:,iter)=label_ini;
    end
    W=W./m;
    save(['./base_result/' name '/' name '_' num2str(t) '.mat'],'G','F','W','km_result','label_all');
    
end