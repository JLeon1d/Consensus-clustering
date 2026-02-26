clear
rand('state',0);

path='./report_datasets/';
DIRS=dir([path,'*.mat']); 
nn=length(DIRS);
for i=1:nn
    name=DIRS(i).name
    if ~exist(['./result_our_nosp/' name])
    load([path  name]);
    c=length(unique(y));
    m=20;
    n=length(y);
    result_acc=zeros(10,10);
    result_nmi=result_acc;
    result_pur=result_acc;
    cluster_num=zeros(10,10);
    load(['./best_para2/' name]);
    for iter=1:10
        Ai=zeros(n,n,m);
        for j=1:m
            idxi=(iter-1)*20+j;
            YY=sparse(Yi{idxi});
            Ai(:,:,j)=full(YY*YY');
%        Ai{j}=YY*YY';
        end
        
        
 %       for gam=1:10
            gam=idx(iter);
            gamma=(gam-1).*0.1;
            gamma=gamma.^2.*m.^2
            S  = Optimize( Ai,c,gamma );
            [clusternum, ypred]=graphconncomp(sparse(S)); 
            ypred = ypred';
            res=ClusteringMeasure(y,ypred)
            result_acc(iter,gam)=res(1);
            result_nmi(iter,gam)=res(2);
            result_pur(iter,gam)=res(3);
            cluster_num(iter,gam)=clusternum;
 %       end
    end
    save(['./result_our_nosp/' name],'result_acc','result_nmi','result_pur','cluster_num');
    end
end