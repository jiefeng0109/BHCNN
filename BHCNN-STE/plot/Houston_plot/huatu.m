function   result = docolor1 (SP)


%% 对于 Indiana Pines and  Salinas 图 着色成彩色
%load('F:\code\python\对比算法\CNN\plot\Indian_pines\plot_label2.mat')
load('Houston_plot.mat')
SP = plot_max;

nrow=1905; ncol=349; 

result=zeros(nrow,ncol,3);
color=zeros(1,16,3);

color(1,1,:)=[0         0    0.5156];
color(1,2,:)=[0         0    0.7656];
color(1,3,:)=[0    0.0156    1.0000];
color(1,4,:)=[0    0.2656    1.0000];
color(1,5,:)=[ 0    0.5156    1.0000];
color(1,6,:)=[ 0    0.7656    1.0000];
color(1,7,:)=[ 0.0156    1.0000    0.9844];
color(1,8,:)=[0.2656    1.0000    0.7344];
color(1,9,:)=[0.5156    1.0000    0.4844];
color(1,10,:)=[0.7656    1.0000    0.2344];
color(1,11,:)=[1.0000    0.9844         0];
color(1,12,:)=[1.0000    0.7344         0];
color(1,13,:)=[1.0000    0.4844         0];
color(1,14,:)=[1.0000    0.2344         0];
color(1,15,:)=[0.5451        0.2706        0.0745];
color(1,16,:)=[0.7344         0         0];


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

[c11,v11]=find(SP==10);
for i1=1:length(c11)
result(c11(i1),v11(i1),:)=color(1,11,:);
end

[c12,v12]=find(SP==11);
for i1=1:length(c12)
result(c12(i1),v12(i1),:)=color(1,12,:);
end

[c13,v13]=find(SP==12);
for i1=1:length(c13)
result(c13(i1),v13(i1),:)=color(1,13,:);
end

[c14,v14]=find(SP==13);
for i1=1:length(c14)
result(c14(i1),v14(i1),:)=color(1,14,:);
end

[c15,v15]=find(SP==14);
for i1=1:length(c15)
result(c15(i1),v15(i1),:)=color(1,15,:);
end

[c16,v16]=find(SP==15);
for i1=1:length(c16)
result(c16(i1),v16(i1),:)=color(1,16,:);
end


figure,imshow(result,[]);
imwrite(result,'houston_hdcnn.tif');


%% http://wenku.baidu.com/view/d4d0ec7f31b765ce050814c2.html
%% 1    17    33    49    65    81    97   113   129   145   161 
%% 177   193   209  225   241   256