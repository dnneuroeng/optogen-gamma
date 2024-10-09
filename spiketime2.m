function spike_time = spiketime2(Vmembrane,tspan)
        step = tspan(2) - tspan(1);
        threshold = -10;
        thresholdSpikes = zeros(length(Vmembrane),1);
%     spike_time = zeros(length(Vmembrane),1);
        onfiring = find(Vmembrane > threshold);
        
        if ~isempty(onfiring) && Vmembrane(onfiring(1),1) > Vmembrane(onfiring(2),1)
            thresholdSpikes(onfiring(1),1)=1;
        end
        
        for i = 2:length(onfiring)-1
            if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
                thresholdSpikes(onfiring(i),1)=1;
            end
        end
        detectedSpikes = thresholdSpikes;
        index=find(detectedSpikes);
    % spike_time = round(index*step);   % 获取放电时刻
        spike_time = index*step;
end

% function spike_time = spiketime(Vmembrane,tspan)
%     step = 0.1;
%     threshold = -1;
%     thresholdSpikes = zeros(length(Vmembrane),1);
%     spike_time      = zeros(length(Vmembrane),1);
%     onfiring = find(Vmembrane>threshold);
%     for i = 2:length(onfiring)-1
%         if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
%             thresholdSpikes(onfiring(i),1)=1;
%         end
%     end
% 
%     index=find(thresholdSpikes);
%     index_revised = ceil(index*0.1)*10+1;
%     spike_time(index_revised) = 1;
% end

% function spike_time = spiketime(Vmembrane,tspan)
%     step = tspan(2) - tspan(1);
%     threshold = -1;
%     thresholdSpikes = zeros(length(Vmembrane),1);
%     spike_time      = zeros(length(Vmembrane),1);
%     onfiring = find(Vmembrane(2:end-1)>threshold);
%     for i = 1:length(onfiring)
%         if onfiring(i) > 1
% 
%             if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
%             thresholdSpikes(onfiring(i),1)=1;
%             end
%         end
% 
%     end
% 
%     index=find(thresholdSpikes);
%     index_revised = floor(index*0.1)*10+1;
%     spike_time(index_revised) = 1;
% end
