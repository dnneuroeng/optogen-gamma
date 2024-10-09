
function [ins_firing_rate] = countspikes2(Vmembrane, tspan)
    % tspan = 整个仿真时间 default 1000(ms)
    % step  = 时间间隔 default 0.1(ms)
    V_threshold     = -10; 
%     thresholdSpikes = double(Vmembrane > threshold); 
    thresholdSpikes = zeros(length(Vmembrane),1); 
    ins_firing_rate = zeros(length(Vmembrane),1); 
    onfiring        = find(Vmembrane > V_threshold); 
            
    if ~isempty(onfiring) 
        if Vmembrane(onfiring(1),1) > Vmembrane(onfiring(2),1)
            thresholdSpikes(onfiring(1),1)  =  1; 
            ins_firing_rate(onfiring(1),1)  =  1./((tspan(onfiring(1)) - tspan(1))/1000); 
        end
        for i = 2:length(onfiring)-1
            if Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)-1,1) && Vmembrane(onfiring(i),1) >= Vmembrane(onfiring(i)+1,1)
                thresholdSpikes(onfiring(i),1)  =  1; 
            end
        end
        for i = 2:length(Vmembrane)
            ins_firing_rate(i,1)   = sum(thresholdSpikes(1:i),1)./((tspan(i) - tspan(1))/1000);
        end
    end
    