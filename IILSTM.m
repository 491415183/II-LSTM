%预处理电池充电数据
for i=1:378
    CC{1,i}(:,1)=cell_11.cycle(i).time;
    CC{1,i}(:,2)=cell_11.cycle(i).current;
    CC{1,i}(:,3)=cell_11.cycle(i).voltage;
    CC{1,i}(:,4)=cell_11.cycle(i).temperature;
    CC{1,i}(:,5)=cell_11.cycle(i).C;
end

for t=1:378
    
        index{1,t}(:,1)=find(CC{1,t}(:,2)>0);
    
end
%提取健康状态80以上的充电数据
for i=1:210
    soh{1,i}(:,1)=CC{1,i}(index{1,i}(:,1),1);
    soh{1,i}(:,2)=CC{1,i}(index{1,i}(:,1),2);
    soh{1,i}(:,3)=CC{1,i}(index{1,i}(:,1),3);
    soh{1,i}(:,4)=CC{1,i}(index{1,i}(:,1),4);
    soh{1,i}(:,5)=CC{1,i}(index{1,i}(:,1),5);
end %提取时间及电流>0数据

for i=1:200
    capacity(i,1)=max(soh{1,i}(:,5));
end
capacity(:,2)=capacity(:,1)/1500;
c=find(capacity(:,2)<0.8,1);
capacity(292:end,:)=[];
capacity(:,1)=[];

%提取特征
%IC曲线峰值
%IC曲线峰值位置
%cc部分充电时间3.9-4.1
%最高温度
%% 50%额定容量的电压
for i=1:291
    [dianya(i,1),dianya(i,2)]=min(abs(CC{1,i}(:,5)-0.75));
    dianya(i,3)=CC{1,i}(dianya(i,2),3)
end
%% CC部分充电时间3.9-4.1
for i=1:291
    [a1(i,1),b1(i,1)]=min(abs(soh{1,i}(:,3)-3.9));
    [a1(i,2),b1(i,2)]=min(abs(soh{1,i}(:,3)-4.1));
    A1(i,1)=soh{1,i}(b1(i,1),5);
    A1(i,2)=soh{1,i}(b1(i,2),5);
    A1(i,3)=A1(i,2)-A1(i,1);%第一个特征
end

%绘制IC曲线
soh1=soh;
for i=1:200 
    t=1
    while t<length(soh1{1,i})
        e=find(soh1{1,i}(:,3)>soh1{1,i}(t,3)+0.012,1);
        soh1{1,i}(e,7)=soh1{1,i}(e,3);
        t=e;
    end
    soh1{1,i}(soh1{1,i}(:,7)==0,:)=[];
end

for t=1:200
  for u=1:length(soh1{1,t})-1
     soh1{1,t}(u,8)=(soh1{1,t}(u+1,5)-soh1{1,t}(u,5))/(soh1{1,t}(u+1,3)-soh1{1,t}(u,3));
  end
  soh1{1,t}(soh1{1,t}(:,8)==0,:)=[];
end
%线性差值光滑曲线
for t=1:200
     x{1,t}=soh1{1,t}(:,3);
     y{1,t}=soh1{1,t}(:,8);
end 
for t=1:200
   x{2,t}=linspace(x{1,t}(1,1),x{1,t}(length(x{1,t}),1),1000);
   y{2,t}=interp1(x{1,t},y{1,t},x{2,t},'spline');
   
end
%set(gca,'xlim',[3.4 4.1]);
clear all
clc
hold on
x1 = x{2,1}(1,400:443);

area( x1, y{2,1}(1,400:443), 'FaceColor', [ 0, 1, 0 ] )
axis( [ 3.3, 4.2, 0, 3600 ] )
hold on
plot(x{2,1},y{2,1},LineWidth=1.5);
xlabel('Voltage(V)')
ylabel('IC(Ah/V)')
% clear all
% close all
%% 绘制渐变IC曲线
for i=1:291
    hold on
    set(gca,'xlim',[3.4 4.17]);
    plot(x{2,i},y{2,i},'color',[0.17 + i*0.0026897 0 0.89 - i*0.001552],LineWidth=1.2);
    %legend('真实值', '预测值')
    xlabel('Voltage(V)')
    ylabel('IC(Ah/V)')
    %colorbar
end



%% 减少渐变曲线数目
for i=1:4:291
    x1{1,i}=x{2,i};
    y1{1,i}=y{2,i};
end
for i=1:length(x1)
        k=isequal(x1(i),{[]});
        if(k==1)
                x1(i)=[];
        end
end

for i=1:length(y1)
        k=isequal(y1(i),{[]});
        if(k==1)
                y1(i)=[];
        end
end

% 创建数据

num_lines = 73;

% 定义蓝色和黄色的 RGB 值
blue = [0, 0, 1];
green = [0, 1, 0];

% 为红、绿、蓝色生成渐变值
r_values = linspace(blue(1), green(1), num_lines);
g_values = linspace(blue(2), green(2), num_lines);
b_values = linspace(blue(3), green(3), num_lines);

% 绘制渐变颜色的曲线
figure;
hold on;
for i = 1:70
    %set(gca,'xlim',[3.4 4.17]);
    
    plot(x1{1,i}, y1{1,i}, 'Color', [r_values(i), g_values(i), b_values(i)], 'LineWidth', 2);
    xlabel('Voltage(V)')
    ylabel('IC(Ah/V)')
end
colorbar
hold off;

x = linspace(0, 2*pi, 100);
% 为示例绘制5条曲线
for i = 1:5
    y = sin(i*x);
    if i == 1
        plot(x, y, 'DisplayName', 'First Curve');
    elseif i == 5
        plot(x, y, 'DisplayName', 'Last Curve');
    else
        plot(x, y);
    end
    hold on;
end

% 显示图例
legend;
%% 寻找IC曲线峰值 F1
for i=1:200
    f1(i,1)=max(y{2,i}(1,:));
end
%% 寻找IC曲线峰值对应位置F2
for i=1:200
    [a(i,1),b(i,1)]=max(y{2,i}(1,:));
    f1(i,2)=x{2,i}(1,b(i,1))
end

%% IC曲线峰值左端斜率 F3
for i=1:200
    [fengzhi(i,1),weizhi(i,1)]=min(abs(soh{1,i}(:,3)-3.9));
    Voltage_fengzhi(i,1)=x{1,i}(weizhi(i,1),1)
end
for i=1:370
    time_voltage(i,1)=find(soh{1,i}(:,3)==Voltage_fengzhi(i,1));
end
time_voltage(:,2)=time_voltage(:,1)+100;
for i=1:200
    Voltage_fengzhi(i,2)=x{1,i}(weizhi(i,1)-3,1);
    fengzhi(i,2)=y{1,i}(weizhi(i,1)-3,1);
end
CZ(:,1)=(fengzhi(:,2)-fengzhi(:,1));
CZ(:,2)=(Voltage_fengzhi(:,2)-Voltage_fengzhi(:,1))
CZ(:,3)=CZ(:,1)./CZ(:,2);
capacity286=capacity;
coeff = corr(CZ(:,3) , capacity(:,2),'type','pearson');
f1(:,3)=CZ(:,3);
%% 固定电压差容量增量差 F4
for i=1:291
    [a2(i,1),b2(i,1)]=min(abs(x{1,i}(:,1)-3.68));
    [a2(i,2),b2(i,2)]=min(abs(x{1,i}(:,1)-3.75));
    f4(i,1)=y{1,i}(b2(i,1),1)-y{1,i}(b2(i,2),1)
end

%%

for i=1:200
    a(i,1)=soh1{1,i}(end,5)
end
% capa_1=capacity./1500;
f1(:,5)=capacity(:,2);
f1(:,4)=a./1500;
%%
% for i=1:291
%     tem(i,1)=max(soh{1,i}(:,4));
% end
% F1(:,4)=tem;
% F1(:,5)=capacity;
% close all
% clear all
%% mae
LSTM0(:,3)=LSTM1(:,2);
LSTM0(:,4)=GPR0(:,2);
LSTM0(:,5)=SVR0(:,2);
MAE1=(sum((abs(LSTM0(:,1)-LSTM0(:,2))))/282);
MAE2=(sum((abs(LSTM0(:,1)-LSTM0(:,3))))/282);
MAE3=(sum((abs(LSTM0(:,1)-LSTM0(:,4))))/282);
MAE4=(sum((abs(LSTM0(:,1)-LSTM0(:,5))))/282);
%%  导入数据
%res = xlsread('数据集.xlsx');
res=Untitled;
%拆分数据

for i=1:8
    while i<=7
        F{1,i}=res(1:i*40,:);
    end
        F{1,i}=res(:,:)
end
F1(1:40,:)=res(1:40,:);
F2(41:80,:)=res(41:80,:);
F3(81:120,:)=res(81:120,:);
F4(121:160,:)=res(121:160,:);
F5(161:200,:)=res(161:200,:);
F6(201:240,:)=res(201:240,:);
F7(241:270,:)=res(241:270,:);
%感觉不需要拆分成这样
F7(find(F7(:,1)==0),:)=[];
F6(find(F6(:,1)==0),:)=[];
F5(find(F5(:,1)==0),:)=[];
F4(find(F4(:,1)==0),:)=[];
F3(find(F3(:,1)==0),:)=[];
F2(find(F2(:,1)==0),:)=[];
F1(find(F1(:,1)==0),:)=[];
%%  划分训练集和测试集
temp = randperm(210);
res=Untitled;
P_train = res(temp(1: 70), 1: 4)';
T_train = res(temp(1: 70), 5)';
M = size(P_train, 2);

P_test = res(temp(71: end), 1: 4)';
T_test = res(temp(71: end), 5)';
N = size(P_test, 2);
%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
P_train =  double(reshape(P_train, 4, 1, 1, M));
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers4 = [
    sequenceInputLayer(4)               % 建立输入层
    
    lstmLayer(10, 'OutputMode', 'last')  % LSTM层
    reluLayer                           % Relu激活层
    
    fullyConnectedLayer(1)              % 全连接层
    regressionLayer];                   % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MiniBatchSize', 5, ...               % 批大小
    'MaxEpochs', 300, ...                 % 最大迭代次数
    'InitialLearnRate', 1e-2, ...          % 初始学习率为
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子
    'LearnRateDropPeriod', 800, ...        % 经过 800 次训练后 学习率为 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型
neta = trainNetwork(p_train, t_train, layers4, options);

%%  仿真预测
t_sim1 = predict(neta, p_train);
t_sim2 = predict(neta, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
J(:,1)=res(:,5);
J(1:70,2)=T_sim1;
J(71:end,2)=T_sim2;
J(:,3)=temp';
bb=sortrows(J,3);
bb(:,1)=res(:,5);
bb(:,3)=bb(:,1)-bb(:,2);
RMSE1=(sum((bb(:,3)).^2)/200).^0.5;
close all
clear all

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
%% 冻结层数
numFrozenLayers = 2;
% 冻结LSTM层之前的层
freezeLayers = 'lstm';
net = freezeWeights(net, freezeLayers);

% 查看冻结后的模型结构
disp(net.Layers);
for i = 2:2:4
    layer = net.Layers(i);
    if i==2
        layer.InputWeightsLearnRateFactor = 0;
        layer.BiasLearnRateFactor = 0;
    elseif i==4
        layer.WeightLearnRateFactor = 0;
        layer.BiasLearnRateFactor = 0;
    end
end
%% net1
temp = randperm(40);

P1_train = F1(temp(1: 28), 1: 4)';
T1_train = F1(temp(1: 28), 5)';
M = size(P1_train, 2);

P1_test = F1(temp(29: end), 1: 4)';
T1_test = F1(temp(29: end), 5)';
N = size(P1_test, 2);

[P1_train, ps1_input] = mapminmax(P1_train, 0, 1);
P1_test = mapminmax('apply', P1_test, ps1_input);

[t1_train, ps1_output] = mapminmax(T1_train, 0, 1);
t1_test = mapminmax('apply', T1_test, ps1_output);

P1_train =  double(reshape(P1_train, 4, 1, 1, M));
P1_test  =  double(reshape(P1_test , 4, 1, 1, N));

t1_train = t1_train';
t1_test  = t1_test' ;

for i = 1 : M
    p1_train{i, 1} = P1_train(:, :, 1, i);
end

for i = 1 : N
    p1_test{i, 1}  = P1_test( :, :, 1, i);
end



%% 
layers = [
    sequenceInputLayer(4)               % 建立输入层
    
    lstmLayer(10, 'OutputMode', 'last')  % LSTM层
    reluLayer                           % Relu激活层
    
    fullyConnectedLayer(1)              % 全连接层
    regressionLayer];                   % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MiniBatchSize', 5, ...               % 批大小
    'MaxEpochs', 300, ...                 % 最大迭代次数
    'InitialLearnRate', 1e-2, ...          % 初始学习率为
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子
    'LearnRateDropPeriod', 800, ...        % 经过 800 次训练后 学习率为 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

net1 = trainNetwork(p1_train, t1_train, layers, options);


t1_sim1 = predict(net1, p1_train);
t1_sim2 = predict(net1, p1_test );

T1_sim1 = mapminmax('reverse', t1_sim1, ps1_output);
T1_sim2 = mapminmax('reverse', t1_sim2, ps1_output);

J(1:30,2)=T1_sim1;
J(31:40,2)=T1_sim2;
J(:,3)=temp';
net1 = freezeWeights(net1);
%% 恢复顺序并计算RMSE 

J1(:,1)=res(1:40,5);
J1(1:30,2)=T1_sim1;
J1(31:end,2)=T1_sim2;
J1(:,3)=temp';
bb1=sortrows(J1,3);
bb1(:,1)=res(1:40,5);
RMSE1=(sum(((T1_sim2-T1_test').^2))/10).^0.5;
%% 训练F2
% 少量数据训练冻结后的net
% 然后解冻 0-1
% 正常训练
net2 = unfreezeWeights(net1,3)

temp = randperm(10);

P2_pre = F2(temp(1: 7), 1: 4)';
T2_pre = F2(temp(1: 7), 5)';
M = size(P2_pre, 2);

P2_pre = F2(temp(8: end), 1: 4)';
T2_tpre = F2(temp(8: end), 5)';
N = size(P2_tpre, 2);


[P2_pre, ps2_input] = mapminmax(P2_pre, 0, 1);
P2_tpre = mapminmax('apply', P2_tpre, ps2_input);

[t2_pre, ps2_output] = mapminmax(T2_pre, 0, 1);
t2_tpre = mapminmax('apply', T2_tpre, ps2_output);

P2_pre =  double(reshape(P2_pre, 4, 1, 1, M));
P2_tpre  =  double(reshape(P2_tpre , 4, 1, 1, N));

t2_pre = t2_pre';
t2_tpre  = t2_tpre' ;

for i = 1 : M
    p2_pre{i, 1} = P2_pre(:, :, 1, i);
end

net = unfreezeWeights(net); % 解冻所有层
net2 = trainNetwork(p2_train, t2_train, layers, options);

temp = randperm(40);

P2_train = F2(temp(1: 28), 1: 4)';
T2_train = F2(temp(1: 28), 5)';
M = size(P2_train, 2);

P2_test = F2(temp(29: end), 1: 4)';
T2_test = F2(temp(29: end), 5)';
N = size(P2_test, 2);

[P2_train, ps2_input] = mapminmax(P2_train, 0, 1);
P2_test = mapminmax('apply', P2_test, ps2_input);

[t2_train, ps2_output] = mapminmax(T2_train, 0, 1);
t2_test = mapminmax('apply', T2_test, ps2_output);

P2_train =  double(reshape(P2_train, 4, 1, 1, M));
P2_test  =  double(reshape(P2_test , 4, 1, 1, N));

t2_train = t2_train';
t2_test  = t2_test' ;

for i = 1 : M
    p2_train{i, 1} = P2_train(:, :, 1, i);
end

for i = 1 : N
    p2_test{i, 1}  = P2_test( :, :, 1, i);
end

net2 = trainNetwork(p2_train, t2_train, layers, options);


t2_sim1 = predict(net2, p2_train);
t2_sim2 = predict(net2, p2_test );

T2_sim1 = mapminmax('reverse', t2_sim1, ps2_output);
T2_sim2 = mapminmax('reverse', t2_sim2, ps2_output);
J(41:70,2)=T2_sim1;
J(71:80,2)=T2_sim2;
J(41:80,3)=temp'+40;
RMSE2=(sum(((T2_sim2-T2_test').^2))/N).^0.5;
%% 训练 net3
net3 = unfreezeWeights(net2,3)

temp = randperm(10);

P3_pre = F3(temp(1: 7), 1: 4)';
T3_pre = F3(temp(1: 7), 5)';
M = size(P3_pre, 2);

P3_pre = F3(temp(8: end), 1: 4)';
T3_tpre = F3(temp(8: end), 5)';
N = size(P3_tpre, 2);


[P3_pre, ps3_input] = mapminmax(P3_pre, 0, 1);
P3_tpre = mapminmax('apply', P3_tpre, ps3_input);

[t3_pre, ps3_output] = mapminmax(T3_pre, 0, 1);
t3_tpre = mapminmax('apply', T3_tpre, ps3_output);

P3_pre =  double(reshape(P3_pre, 4, 1, 1, M));
P3_tpre  =  double(reshape(P3_tpre , 4, 1, 1, N));

t3_pre = t3_pre';
t3_tpre  = t3_tpre' ;

for i = 1 : M
    p3_pre{i, 1} = P3_pre(:, :, 1, i);
end

net3 = unfreezeWeights(net3); % 解冻所有层
net3 = trainNetwork(p3_train, t3_train, layers, options);
temp = randperm(40);

P3_train = F3(temp(1: 30), 1: 4)';
T3_train = F3(temp(1: 30), 5)';
M = size(P3_train, 2);

P3_test = F3(temp(31: end), 1: 4)';
T3_test = F3(temp(31: end), 5)';
N = size(P3_test, 2);

[P3_train, ps3_input] = mapminmax(P3_train, 0, 1);
P3_test = mapminmax('apply', P3_test, ps3_input);

[t3_train, ps3_output] = mapminmax(T3_train, 0, 1);
t3_test = mapminmax('apply', T3_test, ps3_output);

P3_train =  double(reshape(P3_train, 4, 1, 1, M));
P3_test  =  double(reshape(P3_test , 4, 1, 1, N));

t3_train = t3_train';
t3_test  = t3_test' ;

for i = 1 : M
    p3_train{i, 1} = P3_train(:, :, 1, i);
end

for i = 1 : N
    p3_test{i, 1}  = P3_test( :, :, 1, i);
end

net3 = trainNetwork(p3_train, t3_train, layers, options);


t3_sim1 = predict(net3, p3_train);
t3_sim2 = predict(net3, p3_test );

T3_sim1 = mapminmax('reverse', t3_sim1, ps3_output);
T3_sim2 = mapminmax('reverse', t3_sim2, ps3_output);

J(81:110,2)=T3_sim1;
J(111:120,2)=T3_sim2;
J(81:120,3)=temp'+80;
RMSE3=(sum(((T3_sim2-T3_test').^2))/N).^0.5;
%% 训练net4
net4 = unfreezeWeights(net3,3)

temp = randperm(10);

P4_pre = F4(temp(1: 7), 1: 4)';
T4_pre = F4(temp(1: 7), 5)';
M = size(P4_pre, 2);

P4_pre = F4(temp(8: end), 1: 4)';
T4_tpre = F4(temp(8: end), 5)';
N = size(P4_tpre, 2);


[P4_pre, ps4_input] = mapminmax(P4_pre, 0, 1);
P4_tpre = mapminmax('apply', P4_tpre, ps4_input);

[t4_pre, ps4_output] = mapminmax(T4_pre, 0, 1);
t4_tpre = mapminmax('apply', T4_tpre, ps4_output);

P4_pre =  double(reshape(P4_pre, 4, 1, 1, M));
P4_tpre  =  double(reshape(P4_tpre , 4, 1, 1, N));

t4_pre = t4_pre';
t4_tpre  = t4_tpre' ;

for i = 1 : M
    p4_pre{i, 1} = P4_pre(:, :, 1, i);
end

net4 = unfreezeWeights(net4); % 解冻所有层
net4 = trainNetwork(p4_train, t4_train, layers, options);

temp = randperm(40);

P4_train = F4(temp(1: 30), 1: 4)';
T4_train = F4(temp(1: 30), 5)';
M = size(P4_train, 2);

P4_test = F4(temp(31: end), 1: 4)';
T4_test = F4(temp(31: end), 5)';
N = size(P4_test, 2);

[P4_train, ps4_input] = mapminmax(P4_train, 0, 1);
P4_test = mapminmax('apply', P4_test, ps4_input);

[t4_train, ps4_output] = mapminmax(T4_train, 0, 1);
t4_test = mapminmax('apply', T4_test, ps4_output);

P4_train =  double(reshape(P4_train, 4, 1, 1, M));
P4_test  =  double(reshape(P4_test , 4, 1, 1, N));

t4_train = t4_train';
t4_test  = t4_test' ;

for i = 1 : M
    p4_train{i, 1} = P4_train(:, :, 1, i);
end

for i = 1 : N
    p4_test{i, 1}  = P4_test( :, :, 1, i);
end

net4 = trainNetwork(p4_train, t4_train, layers, options);


t4_sim1 = predict(net4, p4_train);
t4_sim2 = predict(net4, p4_test );

T4_sim1 = mapminmax('reverse', t4_sim1, ps4_output);
T4_sim2 = mapminmax('reverse', t4_sim2, ps4_output);
RMSE4=(sum(((T4_sim2-T4_test').^2))/N).^0.5;
J(121:150,2)=T4_sim1;
J(151:160,2)=T4_sim2;
J(121:160,3)=temp'+120;
%% 训练net5
net5 = unfreezeWeights(net4,3)

temp = randperm(10);

P5_pre = F5(temp(1: 7), 1: 4)';
T5_pre = F5(temp(1: 7), 5)';
M = size(P5_pre, 2);

P5_pre = F5(temp(8: end), 1: 4)';
T5_tpre = F5(temp(8: end), 5)';
N = size(P5_tpre, 2);


[P5_pre, ps5_input] = mapminmax(P5_pre, 0, 1);
P5_tpre = mapminmax('apply', P5_tpre, ps5_input);

[t5_pre, ps5_output] = mapminmax(T5_pre, 0, 1);
t5_tpre = mapminmax('apply', T5_tpre, ps5_output);

P5_pre =  double(reshape(P5_pre, 4, 1, 1, M));
P5_tpre  =  double(reshape(P5_tpre , 4, 1, 1, N));

t5_pre = t5_pre';
t5_tpre  = t5_tpre' ;

for i = 1 : M
    p5_pre{i, 1} = P5_pre(:, :, 1, i);
end

net5 = unfreezeWeights(net5); % 解冻所有层
net5 = trainNetwork(p5_train, t5_train, layers, options);

temp = randperm(40);

P5_train = F5(temp(1: 30), 1: 4)';
T5_train = F5(temp(1: 30), 5)';
M = size(P5_train, 2);

P5_test = F5(temp(31: end), 1: 4)';
T5_test = F5(temp(31: end), 5)';
N = size(P5_test, 2);

[P5_train, ps5_input] = mapminmax(P5_train, 0, 1);
P5_test = mapminmax('apply', P5_test, ps5_input);

[t5_train, ps5_output] = mapminmax(T5_train, 0, 1);
t5_test = mapminmax('apply', T5_test, ps5_output);

P5_train =  double(reshape(P5_train, 4, 1, 1, M));
P5_test  =  double(reshape(P5_test , 4, 1, 1, N));

t5_train = t5_train';
t5_test  = t5_test' ;

for i = 1 : M
    p5_train{i, 1} = P5_train(:, :, 1, i);
end

for i = 1 : N
    p5_test{i, 1}  = P5_test( :, :, 1, i);
end

net5 = trainNetwork(p5_train, t5_train, layers, options);


t5_sim1 = predict(net5, p5_train);
t5_sim2 = predict(net5, p5_test );

T5_sim1 = mapminmax('reverse', t5_sim1, ps5_output);
T5_sim2 = mapminmax('reverse', t5_sim2, ps5_output);

J(161:190,2)=T5_sim1;
J(191:200,2)=T5_sim2;
J(161:200,3)=temp'+160;

RMSE5=(sum(((T5_sim2-T5_test').^2))/N).^0.5;



%% 251-291为增量部分 先集成弱学习器到强学习器

error1=sqrt(sum((T1_sim2' - T1_test).^2) ./ 10);
error2=sqrt(sum((T2_sim2' - T2_test).^2) ./ 10);
error3=sqrt(sum((T3_sim2' - T3_test).^2) ./ 10);
error4=sqrt(sum((T4_sim2' - T4_test).^2) ./ 10);
error5=sqrt(sum((T5_sim2' - T5_test).^2) ./ 10);
E=error1+error2+error3+error4+error5;
%权重系数
w1=error1/E;
w2=error2/E;
w3=error3/E;
w4=error4/E;
w5=error5/E;
%估计201-241 并加入新的弱学习器
net6 = unfreezeWeights(net5,3)

temp = randperm(10);

P6_pre = F6(temp(1: 7), 1: 4)';
T6_pre = F6(temp(1: 7), 5)';
M = size(P6_pre, 2);

P6_pre = F6(temp(8: end), 1: 4)';
T6_tpre = F6(temp(8: end), 5)';
N = size(P6_tpre, 2);


[P6_pre, ps6_input] = mapminmax(P6_pre, 0, 1);
P6_tpre = mapminmax('apply', P6_tpre, ps6_input);

[t6_pre, ps6_output] = mapminmax(T6_pre, 0, 1);
t6_tpre = mapminmax('apply', T6_tpre, ps6_output);

P6_pre =  double(reshape(P6_pre, 4, 1, 1, M));
P6_tpre  =  double(reshape(P6_tpre , 4, 1, 1, N));

t6_pre = t6_pre';
t6_tpre  = t6_tpre' ;

for i = 1 : M
    p6_pre{i, 1} = P6_pre(:, :, 1, i);
end

net6 = unfreezeWeights(net6); % 解冻所有层
net6 = trainNetwork(p6_train, t6_train, layers, options);
temp = randperm(40);

P6_train = F6(temp(1: 30), 1: 4)';
T6_train = F6(temp(1: 30), 5)';
M = size(P6_train, 2);

P6_test = F6(temp(31: end), 1: 4)';
T6_test = F6(temp(31: end), 5)';
N = size(P6_test, 2);

[P6_train, ps6_input] = mapminmax(P6_train, 0, 1);
P6_test = mapminmax('apply', P6_test, ps6_input);

[t6_train, ps6_output] = mapminmax(T6_train, 0, 1);
t6_test = mapminmax('apply', T6_test, ps6_output);

P6_train =  double(reshape(P6_train, 4, 1, 1, M));
P6_test  =  double(reshape(P6_test , 4, 1, 1, N));

t6_train = t6_train';
t6_test  = t6_test' ;

for i = 1 : M
    p6_train{i, 1} = P6_train(:, :, 1, i);
end

for i = 1 : N
    p6_test{i, 1}  = P6_test( :, :, 1, i);
end

net6 = trainNetwork(p6_train, t6_train, layers, options);


t6_sim1 = predict(net6, p6_train);
t6_sim2 = predict(net6, p6_test );

T6_sim1 = mapminmax('reverse', t6_sim1, ps6_output);
T6_sim2 = mapminmax('reverse', t6_sim2, ps6_output);

J(201:230,2)=T6_sim1;
J(231:240,2)=T6_sim2;
J(201:240,3)=temp'+200;
RMSE6=(sum(((T6_sim2-T6_test').^2))/N).^0.5;

t66_sim1 = predict(net6, p6_train);
t66_sim2 = predict(net6, p6_test );

T66_sim1 = mapminmax('reverse', t66_sim1, ps6_output);
T66_sim2 = mapminmax('reverse', t66_sim2, ps6_output);

error61=sqrt(sum((T6_sim1(:,1)' - T6_train(:,1)).^2) ./ M);
error62=sqrt(sum((T6_sim1(:,2)' - T6_train).^2) ./ M);
error63=sqrt(sum((T6_sim1(:,3)' - T6_train).^2) ./ M);
error64=sqrt(sum((T6_sim1(:,4)' - T6_train).^2) ./ M);
error65=sqrt(sum((T6_sim1(:,5)' - T6_train).^2) ./ M);
error66=sqrt(sum((T66_sim1' - T6_train).^2) ./ M);
% 舍弃net3
E1=error61+error63+error64+error65+error66;
w1=error61/E1;
w2=error63/E1;
w3=error64/E1;
w4=error65/E1;
w5=error66/E1;
%% 241-286
error1=sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2=sqrt(sum((T2_sim1' - T2_train).^2) ./ M);
error3=sqrt(sum((T3_sim1' - T3_train).^2) ./ M);
error4=sqrt(sum((T4_sim1' - T4_train).^2) ./ M);
error5=sqrt(sum((T5_sim1' - T5_train).^2) ./ M);
E=error1+error2+error3+error4+error5;
%权重系数
w1=error1/E;
w2=error2/E;
w3=error3/E;
w4=error4/E;
w5=error5/E;
%估计241-291 并加入新的弱学习器
temp = randperm(10);

P7_pre = F7(temp(1: 7), 1: 4)';
T7_pre = F7(temp(1: 7), 5)';
M = size(P7_pre, 2);

P7_pre = F7(temp(8: end), 1: 4)';
T7_tpre = F7(temp(8: end), 5)';
N = size(P7_tpre, 2);


[P7_pre, ps7_input] = mapminmax(P7_pre, 0, 1);
P7_tpre = mapminmax('apply', P7_tpre, ps7_input);

[t7_pre, ps7_output] = mapminmax(T7_pre, 0, 1);
t7_tpre = mapminmax('apply', T7_tpre, ps7_output);

P7_pre =  double(reshape(P7_pre, 4, 1, 1, M));
P7_tpre  =  double(reshape(P7_tpre , 4, 1, 1, N));

t7_pre = t7_pre';
t7_tpre  = t7_tpre' ;

for i = 1 : M
    p7_pre{i, 1} = P7_pre(:, :, 1, i);
end

net7 = unfreezeWeights(net7); % 解冻所有层
net7 = trainNetwork(p7_train, t7_train, layers, options);

temp = randperm(30);

P7_train = F7(temp(1: 21), 1: 4)';
T7_train = F7(temp(1: 21), 5)';
M = size(P7_train, 2);

P7_test = F7(temp(22: end), 1: 4)';
T7_test = F7(temp(22: end), 5)';
N = size(P7_test, 2);

[P7_train, ps7_input] = mapminmax(P7_train, 0, 1);
P7_test = mapminmax('apply', P7_test, ps7_input);

[t7_train, ps7_output] = mapminmax(T7_train, 0, 1);
t7_test = mapminmax('apply', T7_test, ps7_output);

P7_train =  double(reshape(P7_train, 4, 1, 1, M));
P7_test  =  double(reshape(P7_test , 4, 1, 1, N));

t7_train = t7_train';
t7_test  = t7_test' ;

for i = 1 : M
    p7_train{i, 1} = P7_train(:, :, 1, i);
end

for i = 1 : N
    p7_test{i, 1}  = P7_test( :, :, 1, i);
end

net7 = trainNetwork(p7_train, t7_train, layers, options);


t7_sim1 = predict(net7, p7_train);
t7_sim2 = predict(net7, p7_test );

T7_sim1 = mapminmax('reverse', t7_sim1, ps7_output);
T7_sim2 = mapminmax('reverse', t7_sim2, ps7_output);
RMSE7=(sum(((T7_sim2-T7_test').^2))/N).^0.5;
J(241:261,2)=T7_sim1;
J(262:270,2)=T7_sim2;
J(241:270,3)=temp'+240;
bb=sortrows(J,3);
bb(:,1)=res(:,5);
rmse=(sum(((bb(:,1)-bb(:,2)).^2))/282).^0.5;
%% 整合估计结果
J(:,1)=res(:,5);
J(1:30,2)=T1_sim1;
J(31:40,2)=T1_sim2;
J(41:70,2)=T2_sim1;
J(71:80,2)=T2_sim2;
J(81:110,2)=T3_sim1;
J(111:120,2)=T3_sim2;
J(121:150,2)=T4_sim1;
J(151:160,2)=T4_sim2;
J(161:190,2)=T5_sim1;
J(191:200,2)=T5_sim2;
J(201:230,2)=T6_sim1;
J(231:240,2)=T6_sim2;
J(241:275,2)=T7_sim1;
J(276:end,2)=T7_sim2;
J(:,3)=temp';

bb=sortrows(J,3);
bb(:,1)=res(:,5);

t7_sim1 = predict(net7, p7_train);
t7_sim2(:,1) = predict(net7, p7_test );
t7=predict(net7, p7_test );
t7_sim2(:,2) = predict(net6, p7_test );
t7_sim2(:,3) = predict(net5, p7_test );
t7_sim2(:,4) = predict(net4, p7_test );
t7_sim2(:,5) = predict(net3, p7_test );
t7_sim2(:,6) = predict(net5, p7_test );
t7_sim2(:,7) = predict(net1, p7_test );

T7 = mapminmax('reverse', t7, ps7_output);



T7_sim1 = mapminmax('reverse', t7_sim1, ps7_output);
T7_sim2(:,1) = mapminmax('reverse', t7_sim2, ps7_output);
T7_sim2(:,2) = mapminmax('reverse', t7_sim2(:,2), ps7_output);

% t6_sim1(:,1) = predict(net1, p6_train);
% t6_sim2(:,1) = predict(net1, p6_test );
% 
% t6_sim1(:,2) = predict(net2, p6_train);
% t6_sim2(:,2) = predict(net2, p6_test );
% 
% t6_sim1(:,3) = predict(net3, p6_train);
% t6_sim2(:,3) = predict(net3, p6_test );
% 
% t6_sim1(:,4) = predict(net4, p6_train);
% t6_sim2(:,4) = predict(net4, p6_test );
% 
% t6_sim1(:,5) = predict(net5, p6_train);
% t6_sim2(:,5) = predict(net5, p6_test );
% 
% T7_sim1 = mapminmax('reverse', t7_sim1, ps7_output);
% T7_sim2 = mapminmax('reverse', t7_sim2, ps7_output);
% 
% T6_sim1(:,6)=w1*T6_sim1(:,1)+w2*T6_sim1(:,2)+w3*T6_sim1(:,3)+w4*T6_sim1(:,4)+w5*T6_sim1(:,5);
% T6_sim2(:,6)=w1*T6_sim2(:,1)+w2*T6_sim2(:,2)+w3*T6_sim2(:,3)+w4*T6_sim2(:,4)+w5*T6_sim2(:,5);
% net6 = trainNetwork(p6_train, t6_train, layers, options);
% 
% t66_sim1 = predict(net6, p6_train);
% t66_sim2 = predict(net6, p6_test );
% 
% T66_sim1 = mapminmax('reverse', t66_sim1, ps6_output);
% T66_sim2 = mapminmax('reverse', t66_sim2, ps6_output);
% 
% error61=sqrt(sum((T6_sim1(:,1)' - T6_train(:,1)).^2) ./ M);
% error62=sqrt(sum((T6_sim1(:,2)' - T6_train).^2) ./ M);
% error63=sqrt(sum((T6_sim1(:,3)' - T6_train).^2) ./ M);
% error64=sqrt(sum((T6_sim1(:,4)' - T6_train).^2) ./ M);
% error65=sqrt(sum((T6_sim1(:,5)' - T6_train).^2) ./ M);
% error66=sqrt(sum((T66_sim1' - T6_train).^2) ./ M);
%%  单独训练最后部分

temp = randperm(46);

P_train = F7(temp(1: 40), 1: 4)';
T_train = F7(temp(1: 40), 5)';
M = size(P_train, 2);

P_test = F7(temp(41: end), 1: 4)';
T_test = F7(temp(41: end), 5)';
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
P_train =  double(reshape(P_train, 4, 1, 1, M));
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers1 = [
    sequenceInputLayer(4)               % 建立输入层
    
    lstmLayer(10, 'OutputMode', 'last')  % LSTM层
    reluLayer                           % Relu激活层
    
    fullyConnectedLayer(1)              % 全连接层
    regressionLayer];                   % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MiniBatchSize', 5, ...               % 批大小
    'MaxEpochs', 300, ...                 % 最大迭代次数
    'InitialLearnRate', 1e-2, ...          % 初始学习率为
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子
    'LearnRateDropPeriod', 800, ...        % 经过 800 次训练后 学习率为 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers1, options);

%%  仿真预测
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
RMSE=(sum(((T_sim2-T_test').^2))/N).^0.5;
RMSEl=(sum(((LSTM(:,1)-LSTM(:,2)).^2))/286).^0.5;









%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
%%  查看网络结构
analyzeNetwork(net)
%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid
%% 温度电压图
hold on
% xlim([3.5 4.2])
xlabel('Voltage(V)')
ylabel('Temperature(℃)')
plot(soh{1,1}(:,1),soh{1,1}(:,4));
plot(soh{1,50}(:,1),soh{1,50}(:,4));
plot(soh{1,100}(:,1),soh{1,100}(:,4));
plot(soh{1,150}(:,1),soh{1,150}(:,4));
plot(soh{1,200}(:,1),soh{1,200}(:,4));
legend('SOH=0.9835','SOH=0.9468','SOH=0.9012','SOH=0.8665','SOH=0.8464');
%% 误差图
hold on
plot(LSTM(:,1),'LineWidth', 1.5);
plot(LSTM(:,3),'LineWidth', 1.5);
plot(GPR(:,2),'LineWidth', 1.5);
plot(SVR(:,2),'LineWidth', 1.5);
plot(LSTM1(:,2),'LineWidth', 1.5);
legend('SOH','集成LSTM','GPR','SVR','传统LSTM');
GPR(:,3)=GPR(:,1)-GPR(:,2);
SVR(:,3)=SVR(:,1)-SVR(:,2);
LSTM(:,4)=LSTM(:,1)-LSTM(:,3)
RMSE1=(sum((LSTM(:,4)).^2)/286).^0.5;
RMSE2=(sum((GPR(:,3)).^2)/286).^0.5;
RMSE3=(sum((SVR(:,3)).^2)/286).^0.5;
RMSE4=(sum((LSTM1(:,3)).^2)/286).^0.5;
%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%冻结层数
numFrozenLayers = 4;
for i = 2:2:4
    layer = net.Layers(i);
    layer.WeightLearnRateFactor = 0;
    layer.BiasLearnRateFactor = 0;
end
%% 绘制双坐标图
CC{1,3}(218:end,1)=CC{1,3}(218:end,1)+2155;
x1 = CC{1,3}(:,1);
y1 = CC{1,3}(:,2);
y2 = CC{1,3}(:,3);

figure;
yyaxis left;
plot(x1, y1, 'b');
ylabel('Current(mA)');
yyaxis right;
plot(x1, y2, 'r');
ylabel('Voltage(V)');
xlabel('x');
%% 绘图

LSTM0(:,3)=LSTM1(:,2);
LSTM0(:,4)=GPR0(:,2);
LSTM0(:,5)=SVR0(:,2);
RMSE1=(sum(((LSTM(:,1)-LSTM(:,2)).^2))/270).^0.5;
RMSE2=(sum(((LSTM(:,1)-LSTM(:,3)).^2))/270).^0.5;
RMSE3=(sum(((LSTM(:,1)-LSTM(:,4)).^2))/270).^0.5;
RMSE4=(sum(((LSTM(:,1)-LSTM(:,5)).^2))/270).^0.5;



subplot(2,1,1);
hold on
% xlim([0.79 1])
% ylim([0.79 1])
plot(LSTM0(:,1),'LineWidth',1.5);
plot(LSTM0(:,2),'LineWidth',1.5);
plot(LSTM0(:,3),'LineWidth',1.5);
plot(LSTM0(:,4),'LineWidth',1.5);
plot(LSTM0(:,5),'LineWidth',1.5);
xlabel('Cycle index')
ylabel('SOH')
legend('SOH','lstm jicheng','LSTM','GPR','SVR');





% sz=66;
% xlabel('Real SOH')
% ylabel('Estimate SOH')
% scatter(LSTM(:,1),LSTM(:,2),sz,'filled');
% %scatter(y1(:,2),y1(:,1),sz,'green','filled');
% plot(cz(:,1),cz(:,2),'Linewidth', 4);
% subplot(2,1,2);
hold on

histogram((LSTM0(:,1)-LSTM0(:,2)),[-0.005:0.00020:0.005]);
xlabel('Error[Ah]');
ylabel('Count');
histogram((LSTM1(:,1)-LSTM1(:,2)),[-0.005:0.00020:0.005]);
xlabel('Error[Ah]');
ylabel('Count');
histogram((GPR0(:,1)-GPR0(:,2)),[-0.005:0.00020:0.005]);
xlabel('Error[Ah]');
ylabel('Count');
histogram((SVR0(:,1)-SVR0(:,2)),[-0.005:0.00020:0.005]);
xlabel('Error[Ah]');
ylabel('Count');

% histogram(Vst,[3.25:0.017:3.6]);
xlabel('Error[Ah]');
ylabel('Count');

hold on
xlabel('Cycle index')
ylabel('SOH')
plot(capa_1,'LineWidth',1.5);
plot(capacity(:,2),'LineWidth',1.5);
plot(capacity1,'LineWidth',1.5);
legend('Cell1','Cell4','Cell7');

%% 迁移LSTM
%七个模型对cell11 进行20个数据初步估计 根据误差选择5个net进行迁移
temp = randperm(300);

P_train = f1(temp(1: 80), 1: 4)';
T_train = f1(temp(1: 80), 5)';
M = size(P_train, 2);

P_test = f1(temp(81: end), 1: 4)';
T_test = f1(temp(81: end), 5)';
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
P_train =  double(reshape(P_train, 4, 1, 1, M));
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end
net = trainNetwork(p_train, t_train, layers1, options);

%%  仿真预测
t1_sim1 = predict(net1, p_train);
t1_sim2 = predict(net1, p_test );

t2_sim1 = predict(net2, p_train);
t2_sim2 = predict(net2, p_test );

t3_sim1 = predict(net3, p_train);
t3_sim2 = predict(net3, p_test );

t4_sim1 = predict(net4, p_train);
t4_sim2 = predict(net4, p_test );

t5_sim1 = predict(net5, p_train);
t5_sim2 = predict(net5, p_test );

t6_sim1 = predict(net6, p_train);
t6_sim2 = predict(net6, p_test );

t7_sim1 = predict(net7, p_train);
t7_sim2 = predict(net7, p_test );
%%  数据反归一化
T1_sim1 = mapminmax('reverse', t1_sim1, ps_output);
T1_sim2 = mapminmax('reverse', t1_sim2, ps_output);


T2_sim1 = mapminmax('reverse', t2_sim1, ps_output);
T2_sim2 = mapminmax('reverse', t2_sim2, ps_output);


T3_sim1 = mapminmax('reverse', t3_sim1, ps_output);
T3_sim2 = mapminmax('reverse', t3_sim2, ps_output);


T4_sim1 = mapminmax('reverse', t4_sim1, ps_output);
T4_sim2 = mapminmax('reverse', t4_sim2, ps_output);


T5_sim1 = mapminmax('reverse', t5_sim1, ps_output);
T5_sim2 = mapminmax('reverse', t5_sim2, ps_output);


T6_sim1 = mapminmax('reverse', t6_sim1, ps_output);
T6_sim2 = mapminmax('reverse', t6_sim2, ps_output);


T7_sim1 = mapminmax('reverse', t7_sim1, ps_output);
T7_sim2 = mapminmax('reverse', t7_sim2, ps_output);


RMSE1=(sum(((T1_sim2-T_test').^2))/N).^0.5;
RMSE2=(sum(((T2_sim2-T_test').^2))/N).^0.5;
RMSE3=(sum(((T3_sim2-T_test').^2))/N).^0.5;
RMSE4=(sum(((T4_sim2-T_test').^2))/N).^0.5;
RMSE5=(sum(((T5_sim2-T_test').^2))/N).^0.5;
RMSE6=(sum(((T6_sim2-T_test').^2))/N).^0.5;
RMSE7=(sum(((T7_sim2-T_test').^2))/N).^0.5;
% 根据结果选择net1 net2 net3 net4 net5 net7
%% 迁移 197个数据 
% numFrozenLayers = 4;
% for i = 2:2:4
%     layer = net.Layers(i);
%     layer.WeightLearnRateFactor = 0;
%     layer.BiasLearnRateFactor = 0;
% end
%  
for i = 2:2:4
    layer = net1.Layers(i);
    if i==2
        layer.InputWeightsLearnRateFactor = 0.2;
        layer.BiasLearnRateFactor = 0.2;
    elseif i==4
        layer.WeightLearnRateFactor = 0.2;
        layer.BiasLearnRateFactor = 0.2;
    end
end


for i = 2:2:4
    layer = net1.Layers(i);
    if i==2
        layer.InputWeightsLearnRateFactor = 1;
        layer.BiasLearnRateFactor = 1;
    elseif i==4
        layer.WeightLearnRateFactor = 1;
        layer.BiasLearnRateFactor = 1;
    end
end



%net1 迁移
temp = randperm(200);

P_train = f1(temp(1: 70), 1: 4)';
T_train = f1(temp(1: 70), 5)';
M = size(P_train, 2);

P_test = f1(temp(71: end), 1: 4)';
T_test = f1(temp(71: end), 5)';
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
P_train =  double(reshape(P_train, 4, 1, 1, M));
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end
% for i = 2:2:4
%     layer = net1.Layers(i);
%     layer.WeightLearnRateFactor = 0.5;
%     layer.BiasLearnRateFactor = 0.5;
% end
net1 = trainNetwork(p_train, t_train, layers, options);
net2 = trainNetwork(p_train, t_train, layers, options);
net3 = trainNetwork(p_train, t_train, layers, options);
net4 = trainNetwork(p_train, t_train, layers, options);
net5 = trainNetwork(p_train, t_train, layers, options);
net6 = trainNetwork(p_train, t_train, layers, options);
net7 = trainNetwork(p_train, t_train, layers, options);
%% 
t1_sim1 = predict(net1, p_train);
t1_sim2 = predict(net1, p_test );

t2_sim1 = predict(net2, p_train);
t2_sim2 = predict(net2, p_test );

t3_sim1 = predict(net3, p_train);
t3_sim2 = predict(net3, p_test );

t4_sim1 = predict(net4, p_train);
t4_sim2 = predict(net4, p_test );

t5_sim1 = predict(net5, p_train);
t5_sim2 = predict(net5, p_test );

t6_sim1 = predict(net6, p_train);
t6_sim2 = predict(net6, p_test );

t7_sim1 = predict(net7, p_train);
t7_sim2 = predict(net7, p_test );
%%  数据反归一化
T1_sim1 = mapminmax('reverse', t1_sim1, ps_output);
T1_sim2 = mapminmax('reverse', t1_sim2, ps_output);


T2_sim1 = mapminmax('reverse', t2_sim1, ps_output);
T2_sim2 = mapminmax('reverse', t2_sim2, ps_output);


T3_sim1 = mapminmax('reverse', t3_sim1, ps_output);
T3_sim2 = mapminmax('reverse', t3_sim2, ps_output);


T4_sim1 = mapminmax('reverse', t4_sim1, ps_output);
T4_sim2 = mapminmax('reverse', t4_sim2, ps_output);


T5_sim1 = mapminmax('reverse', t5_sim1, ps_output);
T5_sim2 = mapminmax('reverse', t5_sim2, ps_output);

T6_sim1 = mapminmax('reverse', t6_sim1, ps_output);
T6_sim2 = mapminmax('reverse', t6_sim2, ps_output);



T7_sim1 = mapminmax('reverse', t7_sim1, ps_output);
T7_sim2 = mapminmax('reverse', t7_sim2, ps_output);


RMSE1=(sum(((T1_sim2-T_test').^2))/N).^0.5;
RMSE2=(sum(((T2_sim2-T_test').^2))/N).^0.5;
RMSE3=(sum(((T3_sim2-T_test').^2))/N).^0.5;
RMSE4=(sum(((T4_sim2-T_test').^2))/N).^0.5;
RMSE5=(sum(((T5_sim2-T_test').^2))/N).^0.5;
RMSE6=(sum(((T6_sim2-T_test').^2))/N).^0.5;
RMSE7=(sum(((T7_sim2-T_test').^2))/N).^0.5;


J(1:70,2)=T7_sim1;
J(71:200,2)=T7_sim2;
J(:,3)=temp';
bb=sortrows(J,3);
bb(:,1)=f1(:,5);
RMSE=(sum(((bb(:,1)-bb(:,2)).^2))/N).^0.5;





% RMSEl=(sum(((LSTM(:,1)-LSTM(:,2)).^2))/286).^0.5;

%% 计算 random  rmse mae 
RMSE_rancell1=(sum(((randomforestcell1(:,2)-randomforestcell1(:,3)).^2))/length(randomforestcell1)).^0.5;
RMSE_rancell4=(sum(((randomforestcell4(:,2)-randomforestcell4(:,3)).^2))/length(randomforestcell4)).^0.5;
RMSE_rancell7=(sum(((randomforestcell7(:,2)-randomforestcell7(:,3)).^2))/length(randomforestcell7)).^0.5;
RMSE_rancell11=(sum(((randomforestcell11(:,2)-randomforestcell11(:,3)).^2))/length(randomforestcell11)).^0.5;
RMSE_rancell5=(sum(((randomforestcell5(:,2)-randomforestcell5(:,3)).^2))/length(randomforestcell5)).^0.5;
RMSE_rancell17=(sum(((randomforestcell17(:,2)-randomforestcell17(:,3)).^2))/length(randomforestcell17)).^0.5;
mae_rancell1 = mean(abs(randomforestcell1(:,2)-randomforestcell1(:,3)));
mae_rancell4 = mean(abs(randomforestcell4(:,2)-randomforestcell4(:,3)));
mae_rancell7 = mean(abs(randomforestcell7(:,2)-randomforestcell7(:,3)));
mae_rancell11 = mean(abs(randomforestcell11(:,2)-randomforestcell11(:,3)));
mae_rancell5 = mean(abs(randomforestcell5(:,2)-randomforestcell5(:,3)));
mae_rancell17 = mean(abs(randomforestcell17(:,2)-randomforestcell17(:,3)));
%% 计算 extra rmse mae
RMSE_extcell1=(sum(((extratreespredictionscell1(:,2)-extratreespredictionscell1(:,3)).^2))/length(extratreespredictionscell1)).^0.5;
RMSE_extcell4=(sum(((extratreespredictionscell4(:,2)-extratreespredictionscell4(:,3)).^2))/length(extratreespredictionscell4)).^0.5;
RMSE_extcell7=(sum(((extratreespredictionscell7(:,2)-extratreespredictionscell7(:,3)).^2))/length(extratreespredictionscell7)).^0.5;
RMSE_extcell11=(sum(((extratreespredictionscell11(:,2)-extratreespredictionscell11(:,3)).^2))/length(extratreespredictionscell11)).^0.5;
RMSE_extcell5=(sum(((extratreespredictionscell5(:,2)-extratreespredictionscell5(:,3)).^2))/length(extratreespredictionscell5)).^0.5;
RMSE_extcell17=(sum(((extratreespredictionscell17(:,2)-extratreespredictionscell17(:,3)).^2))/length(extratreespredictionscell17)).^0.5;
mae_extcell1 = mean(abs(extratreespredictionscell1(:,2)-extratreespredictionscell1(:,3)));
mae_extcell4 = mean(abs(extratreespredictionscell4(:,2)-extratreespredictionscell4(:,3)));
mae_extcell7 = mean(abs(extratreespredictionscell7(:,2)-extratreespredictionscell7(:,3)));
mae_extcell11 = mean(abs(extratreespredictionscell11(:,2)-extratreespredictionscell11(:,3)));
mae_extcell5 = mean(abs(extratreespredictionscell5(:,2)-extratreespredictionscell5(:,3)));
mae_extcell17 = mean(abs(extratreespredictionscell17(:,2)-extratreespredictionscell17(:,3)));
%% 检验RMSE
R1=(sum(((LSTM0(:,1)-LSTM0(:,2)).^2))/length(LSTM0)).^0.5;
R2=(sum(((LSTM0(:,1)-LSTM0(:,3)).^2))/length(LSTM0)).^0.5;
R3=(sum(((LSTM0(:,1)-LSTM0(:,4)).^2))/length(LSTM0)).^0.5;
R4=(sum(((LSTM0(:,1)-LSTM0(:,5)).^2))/length(LSTM0)).^0.5;
R5=(sum(((LSTM0(:,1)-LSTM0(:,6)).^2))/length(LSTM0)).^0.5;
R6=(sum(((LSTM0(:,1)-LSTM0(:,7)).^2))/length(LSTM0)).^0.5;
M1= mean(abs(LSTM0(:,1)-LSTM0(:,2)));
M2= mean(abs(LSTM0(:,1)-LSTM0(:,3)));
M3= mean(abs(LSTM0(:,1)-LSTM0(:,4)));
M4= mean(abs(LSTM0(:,1)-LSTM0(:,5)));
M5= mean(abs(LSTM0(:,1)-LSTM0(:,6)));
M6= mean(abs(LSTM0(:,1)-LSTM0(:,7)));
%% 画图CELL1
X1(:,1)=1:1:286;
for i=1:7
    X1(:,i+1)=LSTM0(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X1(:,1), X1(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X1(:,1), X1(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X1(:,1), X1(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X1(:,1), X1(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X1(:,1), X1(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X1(:,1), X1(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X1(:,1), X1(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 画图CELL4
X4(:,1)=1:1:270;
for i=1:7
    X4(:,i+1)=LSTM1(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X4(:,1), X4(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X4(:,1), X4(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X4(:,1), X4(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X4(:,1), X4(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X4(:,1), X4(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X4(:,1), X4(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X4(:,1), X4(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% cell7
%% 画图CELL4
X7(:,1)=1:1:282;
for i=1:7
    X7(:,i+1)=LSTM2(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X7(:,1), X7(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X7(:,1), X7(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X7(:,1), X7(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X7(:,1), X7(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X7(:,1), X7(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X7(:,1), X7(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X7(:,1), X7(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% cell11 
X11(:,1)=1:1:197;
for i=1:7
    X11(:,i+1)=LSTM3(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X11(:,1), X11(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X11(:,1), X11(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X11(:,1), X11(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X11(:,1), X11(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X11(:,1), X11(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X11(:,1), X11(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X11(:,1), X11(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');

%% cell5
X5(:,1)=1:1:300;
for i=1:7
    X5(:,i+1)=LSTM4(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X5(:,1), X5(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X5(:,1), X5(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X5(:,1), X5(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X5(:,1), X5(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X5(:,1), X5(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X5(:,1), X5(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X5(:,1), X5(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% cell17
X17(:,1)=1:1:200;
for i=1:7
    X17(:,i+1)=LSTM5(:,i);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(X17(:,1), X17(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(X17(:,1), X17(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(X17(:,1), X17(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(X17(:,1), X17(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(X17(:,1), X17(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(X17(:,1), X17(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(X17(:,1), X17(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell1
E1(:,1)=1:1:286;
for i=1:7
    E1(:,i+1)=LSTM0(:,i)-LSTM0(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E1(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E1(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E1(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E1(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E1(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E1(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E1(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell4
E4(:,1)=1:1:270;
for i=1:7
    E4(:,i+1)=LSTM1(:,i)-LSTM1(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E4(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E4(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E4(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E4(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E4(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E4(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E4(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell7
E7(:,1)=1:1:282;
for i=1:7
    E7(:,i+1)=LSTM2(:,i)-LSTM2(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E7(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E7(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E7(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E7(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E7(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E7(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E7(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell11
E11(:,1)=1:1:197;
for i=1:7
    E11(:,i+1)=LSTM3(:,i)-LSTM3(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E11(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E11(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E11(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E11(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E11(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E11(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E11(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell5
E5(:,1)=1:1:300;
for i=1:7
    E5(:,i+1)=LSTM4(:,i)-LSTM4(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E5(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E5(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E5(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E5(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E5(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E5(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E5(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 误差图  cell17
E17(:,1)=1:1:200;
for i=1:7
    E17(:,i+1)=LSTM5(:,i)-LSTM5(:,1);
end
CM = colormap('parula');
hold on 
% 创建 plot
plot(E17(:,2),'DisplayName','Observed','LineWidth',1.5);

% 创建 plot
plot(E17(:,3),'DisplayName','II-LSTM','LineWidth',1.5);

% 创建 plot
plot(E17(:,4),'DisplayName','LSTM','LineWidth',1);

% 创建 plot
plot(E17(:,5),'DisplayName','GPR','LineWidth',1);

% 创建 plot
plot(E17(:,6),'DisplayName','SVR','LineWidth',1);

% 创建 plot
plot(E17(:,7),'DisplayName','RF','LineWidth',1);

% 创建 plot
plot(E17(:,8),'DisplayName','ET','LineWidth',1);

% 创建 ylabel
ylabel('SOH','FontName','Times New Roman');

% 创建 xlabel
xlabel('Cycle number','FontName','Times New Roman');
legend('show');
%% 计算R方
% 实际观测值
%actual_values = [/* your actual values here */];

% 模型预测值
%predicted_values = [/* your predicted values here */];

% 计算均值cell1
for i=1:6
    mean_actual_cell1(:,i) = mean(LSTM0(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell1(:,i) = sum((LSTM0(:,1) - mean_actual_cell1(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell1(:,i) = sum((LSTM0(:,1) - LSTM0(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell1(1,i) = 1 - (residual_sum_squares_cell1(1,i) / total_sum_squares_cell1(1,i));
end

%% 计算均值cell4
for i=1:6
    mean_actual_cell4(:,i) = mean(LSTM1(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell4(:,i) = sum((LSTM1(:,1) - mean_actual_cell4(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell4(:,i) = sum((LSTM1(:,1) - LSTM1(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell4(1,i) = 1 - (residual_sum_squares_cell4(1,i) / total_sum_squares_cell4(1,i));
end
%% 计算均值cell7
for i=1:6
    mean_actual_cell7(:,i) = mean(LSTM2(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell7(:,i) = sum((LSTM2(:,1) - mean_actual_cell7(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell7(:,i) = sum((LSTM2(:,1) - LSTM2(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell7(1,i) = 1 - (residual_sum_squares_cell7(1,i) / total_sum_squares_cell7(1,i));
end
%% 计算均值cell11
for i=1:6
    mean_actual_cell11(:,i) = mean(LSTM3(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell11(:,i) = sum((LSTM3(:,1) - mean_actual_cell11(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell11(:,i) = sum((LSTM3(:,1) - LSTM3(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell11(1,i) = 1 - (residual_sum_squares_cell11(1,i) / total_sum_squares_cell11(1,i));
end
%% 计算均值cell5
for i=1:6
    mean_actual_cell5(:,i) = mean(LSTM4(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell5(:,i) = sum((LSTM4(:,1) - mean_actual_cell5(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell5(:,i) = sum((LSTM4(:,1) - LSTM4(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell5(1,i) = 1 - (residual_sum_squares_cell5(1,i) / total_sum_squares_cell5(1,i));
end
%% 计算均值cell17
for i=1:6
    mean_actual_cell17(:,i) = mean(LSTM5(:,i+1));
end

% 总平方和
for i=1:6
    total_sum_squares_cell17(:,i) = sum((LSTM5(:,1) - mean_actual_cell17(:,i)).^2);
end

% 残差平方和
for i=1:6
    residual_sum_squares_cell17(:,i) = sum((LSTM5(:,1) - LSTM5(:,i+1)).^2);
end
% R方计算
for i=1:6
    r_squared_cell17(1,i) = 1 - (residual_sum_squares_cell17(1,i) / total_sum_squares_cell17(1,i));
end