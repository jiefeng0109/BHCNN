function   result = docolor1 (SP)


%% 对于 Pavia 图 着色成彩色
%load('F:\code\python\对比算法\CNN\plot\Indian_pines\plot_label2.mat')
load('PaviaU_plot.mat')
SP = plot_max;

nrow=610; ncol=340; 


result=zeros(nrow,ncol,3);
color=zeros(1,10,3);

color(1,1,:)=[0 0 0];
color(1,2,:)=[1 1 1];
color(1,3,:)=[0 1 0];
color(1,4,:)=[0 1 1];
color(1,5,:)=[0 0 1];
color(1,6,:)=[1 0 1];
color(1,7,:)=[0.7412 0.4196 0.0353];
color(1,8,:)=[0.3647 0.0471 0.4824];
color(1,9,:)=[1 0 0];
color(1,10,:)=[1 1 0];



[c1,v1]=find(SP==0);
for i1=1:length(c1)
result(c1(i1),v1(i1),:)=color(1,1,:);
end
[c2,v2]=find(SP==1);
for i1=1:length(c2)
result(c2(i1),v2(i1),:)=color(1,2,:);
end

[c3,v3]=find(SP==2);
for i1=1:length(c3)
result(c3(i1),v3(i1),:)=color(1,3,:);
end

[c4,v4]=find(SP==3);
for i1=1:length(c4)
result(c4(i1),v4(i1),:)=color(1,4,:);
end

[c5,v5]=find(SP==4);
for i1=1:length(c5)
result(c5(i1),v5(i1),:)=color(1,5,:);
end

[c6,v6]=find(SP==5);
for i1=1:length(c6)
result(c6(i1),v6(i1),:)=color(1,6,:);
end

[c7,v7]=find(SP==6);
for i1=1:length(c7)
result(c7(i1),v7(i1),:)=color(1,7,:);
end

[c8,v8]=find(SP==7);
for i1=1:length(c8)
result(c8(i1),v8(i1),:)=color(1,8,:);
end

[c9,v9]=find(SP==8);
for i1=1:length(c9)
result(c9(i1),v9(i1),:)=color(1,9,:);
end

[c10,v10]=find(SP==9);
for i1=1:length(c10)
result(c10(i1),v10(i1),:)=color(1,10,:);
end

figure,imshow(result,[]);
imwrite(result,'PaviaU_hdcnn.tif');


%% http://wenku.baidu.com/view/d4d0ec7f31b765ce050814c2.html
%% 1    17    33    49    65    81    97   113   129   145   161 
%% 177   193   209  225   241   256