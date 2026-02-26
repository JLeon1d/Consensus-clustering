function [ W_result ] = LBFGSB2(W_init,A,G,V,Lambda1,Lambda2,mu,alpha,lambda,m )
%QUASINEWTON1 Summary of this function goes here
%   Detailed explanation goes here
    [n,~]=size(W_init);
    funObj=@(x)obj_f2( A,G,V,Lambda1,Lambda2,mu,alpha,lambda,m,n,x);
    
    x_init=reshape(W_init,n*n,1);
%    [row,~]=size(x_init);
    LB=zeros(n*n,1);
%    LB=zeros(row,1);
    UB=ones(n*n,1);
    options=[];
    options.x0=x_init;
%    options.verbose=0;
    options.pgtol=1e-3;
    options.maxIts=20;
    options.printEvery=1;
%    options.numDiff=1;
%    x = minConf_TMP(funObj,x_init,LB,UB,options);
     x = lbfgsb(funObj,LB,UB,options);
    W_result=reshape(x,n,n);
%       opts.x0     The starting value (default: all zeros)
%       opts.m      Number of limited-memory vectors to use in the algorithm
%                       Try 3 <= m <= 20. (default: 5 )
%       opts.factr  Tolerance setting (see this source code for more info)
%                       (default: 1e7 ). This is later multiplied by machine epsilon
%       opts.pgtol  Another tolerance setting, relating to norm(gradient,Inf)
%                       (default: 1e-5)
%       opts.maxIts         How many iterations to allow (default: 100)
%       opts.maxTotalIts    How many iterations to allow, including linesearch iterations
%                       (default: 5000)
%       opts.printEvery     How often to display information (default: 1)
%       opts.errFcn         A function handle (or cell array of several function handles)
%                       that computes whatever you want. The output will be printed
%                       to the screen every 'printEvery' iterations. (default: [] )
%                       Results saved in columns 3 and higher of info.err variable
    

end

