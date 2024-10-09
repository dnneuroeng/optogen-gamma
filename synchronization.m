% revised 2024/1/16
% 修改spiketime函数，避免因index_revised导致的数组长度溢出

% revised 2024/1/12
% 修改spiketime函数，spike_time数据长度与Vmembrane(:,i)长度一致

% revised 2023/12/28
% 使用压缩矩阵减少内存占用
% spiketime函数中使用了向量化操作和索引查找，以提高计算效率

function [synchro] = synchronization(Vmembrane,n,tspan)

% k = zeros(n);
    sequence = zeros(length(tspan),n);
    for i = 1:n
        sequence(:,i) = spiketime(Vmembrane(:,i),tspan);
    end

%     sequence = cell(n, 1);
%     for i = 1:n
%         sequence{i} = spiketime(Vmembrane(:, i), tspan);
%     end
    
    k = sparse(n,n);
    
    for i = 1:n
        for j = 1:n
            if ~isempty(sequence(:,i)) && ~isempty(sequence(:,j)) && ~(all(sequence(:,i) == 0) || all(sequence(:,j) == 0))
                k_ij = sum(sequence(:,i) .* sequence(:,j)) / sqrt(sum(sequence(:,i)) * sum(sequence(:,j)));
                k(i, j) = k_ij;
            else
                k(i, j) = 0;
            end
        end
    end
    % synchro = mean(k(:));
    
    count = n*(n-1)/2;
    k_sum = 0;
    for i = 1: n-1
        for j = i+1:n
            k_sum = k_sum+k(i,j);
        end
    end
    synchro =  k_sum/count;

end

function spike_time = spiketime(Vmembrane,tspan)
    step = tspan(2) - tspan(1);
    threshold = -1;
    thresholdSpikes = zeros(length(Vmembrane),1);
    spike_time      = zeros(length(Vmembrane),1);
%     spike_time = zeros(length(Vmembrane),1);
    onfiring = find(Vmembrane(2:end-1)>threshold);
    for i = 1:length(onfiring)
        if onfiring(i) > 1

            if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
            thresholdSpikes(onfiring(i),1)=1;
            end
        end
    
    end
    
    index=find(thresholdSpikes);
    index_revised = floor(index*0.1)*10+1;
    spike_time(index_revised) = 1;
end


