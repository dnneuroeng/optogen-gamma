% revised 2024/1/16 by Xue Chun
% 对countspikes和spiketime进行修正，避免因onfiring(i)=1导致的寻址问题
% 本函数中spiketime仅返回放电时刻，其矩阵大小与Vmembrane不同

% 首先获得每个神经元放电的序列
% 再结合神经元序号获得每个神经元放电时刻的坐标

% test code
% Vmembrane = resultMatrix1(:,1:n);
% rasterplots_new(Vmembrane,tspan,type_of_neuron)

% 2024/4/13
% 将Vmembrane的列重新排列，将抑制性神经元排在前半部分，兴奋性神经元排在后半部分
% 绘制rasterplots时神经元分类绘制

function [spikedata] = rasterplots_rearrange(Vmembrane,tspan,type_of_neuron)

n = length(Vmembrane(1,:));  % 获取神经元个数，即Vmembrane列数
count_spike = zeros(n,1); 


inhibitory_indices = type_of_neuron == 2;
excitatory_indices = type_of_neuron == 1;

count = sum(inhibitory_indices);

% 将Vmembrane的列重新排列，将抑制性神经元排在前半部分，兴奋性神经元排在后半部分
rearranged_Vmembrane = [Vmembrane(:, inhibitory_indices), Vmembrane(:, excitatory_indices)];

type_of_neuron = ones(n,1);
type_of_neuron(1:count) = 2;   % 将中间神经元排布在前半部分

for k = 1:n
    count_spike(k) =  countspikes(rearranged_Vmembrane(:,k),tspan);
end

spikedata = zeros(sum(count_spike),3); 
for j = 2:n %获取每个神经元的放电具体时间，并填入矩阵
    if count_spike(j) ~= 0
    spikedata(sum(count_spike(1: j-1))+1:sum(count_spike(1: j)) , 1) = spiketime(rearranged_Vmembrane(:,j),tspan);
    spikedata(sum(count_spike(1: j-1))+1:sum(count_spike(1: j)) , 2) = j;
    spikedata(sum(count_spike(1: j-1))+1:sum(count_spike(1: j)) , 3) = type_of_neuron(j);
    end
end

spikedata(1:count_spike(1),1) = spiketime(rearranged_Vmembrane(:,1),tspan); % 行
spikedata(1:count_spike(1),2) = 1;                               % 列
spikedata(1:count_spike(1),3) = type_of_neuron(1);               % 神经元类型

type = spikedata(:,3); 
colors = zeros(size(type));
colors(type == 1) = 1;   % pyramidal blue
colors(type == 2) = 2;   % interneuron red

figure;
scatter(spikedata(:,1), spikedata(:,2), 5, colors, 'filled');
% colormap ([0 0 1; 1 0 0]);   % 蓝色 红色
colormap ([1 0 0; 0 0 0]);   % I-cell黑色，E-cell红色
% colormap ([0 0.45 0.9;0.85 0.33 0.1 ]);
% 0 0.45 0.9蓝色
% 0.85 0.33 0.1黄色

title('')
xlabel('放电时刻 t/ms');
ylabel('神经元序号');
xmin = 0;
xmax = max(tspan);
ymin = 0;
ymax = n;
axis([xmin xmax ymin ymax]);
% count_spike 每个神经元放电次数
% spike_time  每个神经元放电时刻
function count_spike = countspikes(Vmembrane,tspan)
    threshold = -10;
    thresholdSpikes = zeros(length(Vmembrane),1);
    onfiring = find(Vmembrane(2:end-1)>threshold);
    for i = 1:length(onfiring)
        if onfiring(i) > 1
            if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
            thresholdSpikes(onfiring(i),1)=1;
            end
        end
    end
    count_spike = sum(thresholdSpikes);
end

function spike_time = spiketime(Vmembrane,tspan)
    threshold = -10;
    thresholdSpikes = zeros(length(Vmembrane),1);
%     spike_time = zeros(length(Vmembrane),1);
    onfiring = find(Vmembrane(2:end-1)>threshold);
    for i = 1:length(onfiring)
        if onfiring(i) > 1
            if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
            thresholdSpikes(onfiring(i),1)=1;
            end
        end
    end
    detectedSpikes = thresholdSpikes;
    index=find(detectedSpikes);
    spike_time = round(index*0.1);   % 获取放电时刻
end
end