function [ obj,grad ] = obj_f1_d2( DWD,GF,GF_all,X,Lambda1,mu,k,m,n,x )
%OBJ_F2 此处显示有关此函数的摘要
%   此处显示详细说明
%tic

A=reshape(x,n,n);
clear x;
%AA{1}=A;
AX=cell(1,k+1);
AX{1}=X;
clear X;

for i=1:k
    AX{i+1}=A*AX{i};
end
AAX=AX{k+1};
XAA=cell(1,k-1);
XAA{1}=AAX'*A;
for i=2:k-1
    XAA{i}=XAA{i-1}*A;
end
GFA=cell(1,k);
GFA{1}=GF_all;
clear GF_all;
    for r=2:k
        GFA{r}=GFA{r-1}*A;
    end



%for i=2:k
%    AA{i}=AA{i-1}*AA{1};
%end

%AAX=AA{k}*X;

obj1=0;
%dd=1./sqrt(max(sum(W,2),eps));
%W=bsxfun(@times,W,dd);
%W=bsxfun(@times,W,dd');
DWD=(eye(n)+DWD)./2;
for i=1:m
    obj1=obj1+sum(sum((AAX-GF{i}).^2));
    
end
clear GF;
obj2=sum(sum(Lambda1.*A));
obj3=sum(sum((A-DWD).^2));
obj=obj1+obj2+mu./2.*obj3;

% tic
% U=2.*AAX*X';
% grad1=zeros(n,n);
% for i=1:n
%     for j=1:n
%         Ii=zeros(n,1);
%         Ii(i)=1;
%         AJA=Ii*AA{k-1}(j,:);
%         for r=1:k-2
%             AJA=AJA+AA{r}(:,i)*AA{k-r-1}(j,:);
%         end
%         Ij=zeros(1,n);
%         Ij(j)=1;
%         AJA=AJA+AA{k-1}(:,i)*Ij;
%         grad1(i,j)=sum(sum(U.*AJA));
%     end
% end
% grad1=grad1.*m;
% toc

%tic
%XAAk=X'*AA{k}';
grad1=AX{k}*AAX';
for r=1:k-1
    grad1=grad1+AX{k-r}*XAA{r};
end
clear XAA AAX;
%grad1=grad1+X*XAA{k-1};
grad1=2.*grad1';
grad1=grad1.*m;
%toc



        
%grad2=zeros(n,n);
grad2=AX{1}*GFA{k};
for r=1:k-1
    grad2=grad2+AX{r+1}*GFA{k-r};
end
%grad2=grad2+AX{k}*FGA{1};
% for i=1:m
%     grad2=grad2+XF{i}*GA{i,k};
% end
% for r=1:k-2
%     for i=1:m
%         grad2=grad2+(AX{r+1}*F{i}')*(GA{i,k-r});
%     end
% end
% for i=1:m
%     grad2=grad2+AX{k}*F{i}'*G{i}';
% %    grad2=grad2+AX{k}*GF{i}';
% end
grad2=grad2';

clear AX   GFA 
grad12=grad1-2.*grad2;
clear grad1 grad2;
%grad3=Lambda1;

%grad4=mu.*A;

%grad5=mu.*DWD;

grad=grad12+Lambda1+mu.*A-mu.*DWD;
%grad=grad1-2.*grad2+Lambda1+mu.*A-mu.*DWD;

grad=reshape(grad,n*n,1);
  
%toc
    
    

end

