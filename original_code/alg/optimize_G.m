function [ G ] = optimize_G( AX,G,F,W,alpha,lambda )
%OPTIMIZE_G 此处显示有关此函数的摘要
%   此处显示详细说明

[n,~]=size(AX);
[c,~]=size(F{1});
m=length(F);
maxiter=5;

for i=1:m
    [idx,~]=find(G{i}');
    alphaW=alpha(i).*W';
    GG_i=sparse(n,n);
    for j=1:m
        if j~=i
            GG_i=GG_i+alpha(j).*G{j}*G{j}';
        end
    end
    for iter=1:maxiter
        flag=0;
        for j=1:n
            AXj=AX(j,:);
            AXj_all=repmat(AXj,c,1);
            obj1=sum((AXj_all-F{i}).^2,2);
            obj1=obj1';
            
            G_j=G{i};
            G_j(j,:)=[];
            W_j1=alphaW(j,:);
            W_j1(j)=[];
            W_j2=alphaW(:,j);
            W_j2(j)=[];
%             W_j2_all=repmat(W_j2,1,c);
%             W_j1_all=repmat(W_j1',1,c);
%             obj2=sum(G_j.*(W_j2_all+W_j1_all)).*lambda.*(-2);
            obj21=sum(bsxfun(@times,G_j,W_j2+W_j1'));
%            obj22=sum(bsxfun(@times,G_j,W_j1'));
            obj2=(obj21).*lambda.*(-2);
            
            GG_i_j=GG_i(:,j);
            GG_i_j(j)=[];
            obj3=sum(bsxfun(@times,G_j,GG_i_j));
            obj3=obj3.*4.*lambda.*alpha(i);
            
            obj4=sum(G_j).*2.*alpha(i).*alpha(i).*lambda;
            
            obj=obj1+obj2+obj3+obj4;
            [~,minidx]=min(obj);
            
%             for k=1:c
%                 GGG=G{i};
%                 GGG(j,:)=zeros(1,c);
%                 GGG(j,k)=1;
%                 obj21=sum(sum((AX-GGG*F{i}).^2));
%                 GGGG=zeros(n,n);
%                 for kk=1:m
%                     if kk==i
%                         GGGG=GGGG+GGG*GGG'.*alpha(i);
%                     else
%                         GGGG=GGGG+G{kk}*G{kk}'.*alpha(i);
%                     end
%                 end
%                 obj22(k)=obj21+lambda.*sum(sum((W-GGGG).^2));
%             end
            
            if idx(j)~=minidx
                G{i}(j,idx(j))=0;
                G{i}(j,minidx)=1;
                idx(j)=minidx;
                flag=1;
            end
        end
        if flag==0
            break;
        end
    end


end




end

