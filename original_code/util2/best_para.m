clear;
data='./report_datasets/';
%mydir='./result4/';
DIRS=dir([data,'*.mat']); 
nn=length(DIRS);
dest_dir = './result_our_new/';
for id = 1:nn
    name=DIRS(id).name
	load([dest_dir name]);
	[~,idx]=max(result_acc,[],2);
	save(['./best_para2/' name],'idx');
end