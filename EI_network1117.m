%% created on 2023/3/14 E:I=4：1固定
%  modified on 2023/3/20
%  modified on 2023/5/24
% pyramidal cell model Reduced Traub-Mile model(2003 Olufsen)
% interneuron model Wang-Buzsaki model(1996 Waang)
%% Interneuron network  

% 11.3 xuechun revised
% 使用压缩矩阵与稀疏矩阵联合的方法 

% function returnMatrix = EI_network1103(t, y, tspan, elecStimulus,elecStimulusMatrix, ...
%     optoStimulus, optoStimulusMatrix,irradiance, ConnectMatrix, type_of_neuron, n, sstoch, noiseValue, ...
%     e_to_e_indices,e_to_i_indices,i_to_e_indices,i_to_i_indices) 

function returnMatrix = EI_network1117(t, y, tspan, elecStimulus,elecStimulusMatrix, sstoch, noiseValue, ...
    optoStimulus, optoStimulusMatrix,irradiance, ConnectMatrix, type_of_neuron, n, ...
    e_to_e_indices,e_to_i_indices,i_to_e_indices,i_to_i_indices)
%%  basic parameter

ENa_i               =  55;      % Reversal potential for sodium (mV)
EK_i                = -90;     % Reversal potential for potassium (mV)
EL_i                = -65;     % Reversal potential for calcium (mV)
ENa_e               =  50;      % Reversal potential for sodium (mV)
EK_e                = -100;    % Reversal potential for potassium (mV)
EL_e                = -67;     % Reversal potential for calcium (mV)
Esyn_gaba           = -75;     % synaptic potential
Esyn_ampa           =  0;       % synaptic potential

gNa_i               = 35;    
gK_i                = 9;    
gL_i                = 0.1;   
c_soma_i            = 1;
gNa_e               = 100;    % nS/um^2
gK_e                = 80;    % nS/um^2
gL_e                = 0.1;    % nS/um^2
c_soma_e            = 1;    % Soma capacitance (nF)

% gNa_e               = 200;    % nS/um^2
% gK_e                = 160;
% c_soma_e            = 0.5; 
c_soma_e            = 1; 
% c_soma_i            = 0.4;
c_soma_i            = 1;

% 以下参数可调
% g_ee = 0.04;
% g_ei = 0.034;%0.034*synapticMatrix(2);
% g_ie = 0.11;
% g_ii = 0.062;

% g_ee = 0.016; 
% g_ei = 0.035; 
% g_ie = 0.15;
% g_ii = 0.062;

% g_ee = 0.01; 
% g_ei = 0.01; 
% g_ie = 0.15;
% g_ii = 0.01;

% g_ee = 0.001; 
% g_ei = 0.008; 
% g_ie = 0.16;
% g_ii = 0.003;

% g_ee = 0.002; 
% g_ei = 0.005; 
% g_ie = 0.012;
% g_ii = 0.003;

% g_ee = 0.005; 
% g_ei = 0.008; 
% g_ie = 0.015;
% g_ii = 0.005;

% g_ee = 0.008;
% g_ei = 0.01; 
% g_ie = 0.06; 
% g_ii = 0.008; 
% g_ee = 0.0073;
g_ee = 0.0073;
g_ii = 0.086;
% g_ii = 0.01;
% g_ei = 0.15; 
% g_ie = 0.05; 
% g_ei = 0.025; 
% g_ei = 0.015; 
% g_ie = 0.075; 
% g_ei = 0.008; 
g_ei = 0.008;
% g_ie = 0.05; 
g_ie = 0.05;

% 应当进一步区分，E - I I - E的上升和下降时间常数 
% alpha_gaba          = 4;
% beta_gaba           = 0.1;
% 
% alpha_ampa          = 10;
% beta_ampa           = 0.5;
% 
% alpha_gaba          = 4;
% beta_gaba           = 2;
% alpha_ampa          = 10;
% beta_ampa           = 1;

% alpha_gaba          = 4;
% beta_gaba           = 0.4;
% alpha_ampa          = 10;
% beta_ampa           = 1;

alpha_gaba          = 2;     % GABA synapse rise time constant = 0.5ms
beta_gaba           = 0.2;   % GABA synapse decay time constant = 5ms
alpha_ampa          = 2;     % AMPA synapse rise time constant = 0.5ms
beta_ampa           = 0.5;   % AMPA synapse decay time constant = 2ms   following Geisler work


%% import  parameters
% HH神经元模型参数


% 计算连接总数
connection_count = length(ConnectMatrix(:,1));   
% 将计算得到的ds 稀疏矩阵，填入上述的含有连接总数的压缩矩阵中
% 稀疏矩阵用于计算循环中的数值


v_soma  = y(1:n);
% mNa     = y((n+1):(2*n));
hNa     = y((n+1):(2*n));
nK      = y((2*n+1):(3*n));
O1      = y((3*n+1):(4*n));
O2      = y((4*n+1):(5*n));
C2      = y((5*n+1):(6*n));
% s 作为压缩矩阵
% s 表示神经元之间相互作用的参数，需要进行计算，其第一第二列表示坐标关系，第三列表示数值
% s            = ConnectMatrix(:,1:2);
s(:,1:2)     = ConnectMatrix(:,1:2);                  
s(:,3)       = y((6*n+1):(6*n+connection_count));    % 需要计算的s即为有连接的神经元相互作用

% 关键是要如何将s中的数值和坐标对应起来

% interneuron
mNa       = zeros(1,n);
dhNa      = zeros(1,n);
dnK       = zeros(1,n);
dv_soma   = zeros(1,n);


%% electrical stimulation

% continuous electrical stimulation without connections: LUT---------------
% Continuous_amp (constant A*random quantity)
% The constant A 所能引起的神经元放电频率 (pyramidal cells)
% 0.15    - 约17Hz
% 0.14    - 约14Hz (α)
% 0.13    - 约10Hz
% 0.125   - 约7-8Hz
% 0.122   - 约5Hz  (θ)
% 0.12    - 约2-3Hz(δ)
% The constant A 所能引起的神经元放电频率 (interneurons)
% 0.2     - 约20Hz (β)
% 0.18    - 约14Hz
% 0.175   - 约12Hz (α)
% 0.17    - 约10Hz 
% 0.165   - 约7Hz
% 0.163   - 约5Hz  (θ)
% 0.1605  - 约2Hz  (δ)
%--------------------------------------------------------------------------  

Ielectrical               = zeros(1,n);
noise_elecStimulus        = zeros(1,n);
gstoch                    = 0.05;

% Ielectrical(find(elecStimulusMatrix == 1)) = 0.4*elecStimulus(ceil(t/0.1)+1);
Ielectrical(elecStimulusMatrix == 1) = elecStimulus(elecStimulusMatrix == 1);

% Ielectrical(1:100) = 0.5;

% for k = find(elecStimulusMatrix == 1)
%     % sstoch_temp = sstoch(:,k); 
%     % Ielectrical(k) = -v_soma(k)*sstoch_temp(ceil(t/0.1)+1)*gstoch;
%     % Ielectrical(k) = -v_soma(k)*sstoch_temp(ceil(t/0.1)+1)*gstoch + 0.5;
%     % Continuous_amp    = - rand(1,1)*rand(1,1) + 0.5*rand(1,1); 
%     Continuous_amp    = elecStimulus(k).*elecStimulusMatrix(k); 
%     Ielectrical(k)    = Continuous_amp; 
% end

% for i = 1:n
%     sstoch_temp = sstoch(:,i);
%     if t > 0
%         Ielectrical(i)    = elecStimulusMatrix(i)*elecStimulus(ceil(t/0.1)+1)-v_soma(i)*sstoch_temp(ceil(t/0.1)+1)*gstoch; %0.06;%0.12;%*0.01;
%     else
%         Ielectrical(i)    = 0;
%     end
%     mean                    = 0;  % 均值 0 
%     sigmo                   = noiseValue;  % 方差 
%     noise_elecStimulus(i)   = normrnd(mean,sigmo);    
%     Ielectrical(i)          = Ielectrical(i) + noise_elecStimulus(i);
% end

%% opto stimulation

optoStimulus    = interp1(tspan,optoStimulus,t); % 光部分不设置扰动

optoStart       = 0;

irradiance0     = 1;                        % Stimulation threshold (mW/mm^2) or (nW/um^2)
wavelength      = 470;                      % nm
E_photon        = 1242/wavelength;          % eV
E_ph            = E_photon*1.6*10^-11;      % nJ
flux            = irradiance/E_ph ;         % photon flux
flux0           = irradiance0/E_ph;         % photons/um^2
Aretinal        = 10^-8;                        % um^2 1.9*10^-8;  about 1.2x10^20 m^2
Fo              = double(flux0*Aretinal/1000);  % photons per ChR per millisecond
Flux            = double((flux*Aretinal/1000)); % photons per ChR per millisecond

Q10.e12dark     = 1.1;
Q10.e21dark     = 1.95;
tau_ChR2L       = 1.3;                      % activation time of ChR2(light)    ms 
tau_ChR2D       = 0.3;                      % activation time of ChR2(dark)     ms 
tau_rdark       = 3000;                     % ms 
gama            = 0.05;
Gd1             = 0.13;%                    % rates of transition O1->C1
Gd2             = 0.0025;%                  % rates of transition O2->C2
e_ctdark        = 0.053;%                   % O1 -> O2 (initial value)
e_tcdark        = 0.023;%                   % O2 -> O1 (initial value)
c1              = 0.005;%                   % constant1
c2              = 0.004;%                   % constant2
Gr_d            = 1/tau_rdark;              % rate of thermal conversion C2->C1
% QEtrans         = 0.5;                        % quantum efficiency (light)
% QEcis           = 0.1;                      % quantum efficiency (dark)
QEtrans         = 0.5;                        % quantum efficiency (light)
QEcis           = 0.1;                        % quantum efficiency (dark)
% e_ct            = e_ctdark + c1*log((Flux^1.2)/Fo); % O1 -> O2
e_ct            = e_ctdark + c1*log(Flux/Fo);   % O1 -> O2
e_tc            = e_tcdark + c2*log(Flux/Fo);   % O2 -> O1
r1              = QEtrans*Flux/Fo;             % transition rate  (C1->O1)   
r2              = QEcis*Flux/Fo;               % transition rate  (C2->O2) 

Ga1=zeros(1,n);
Ga2=zeros(1,n);
dO1=zeros(1,n);
dO2=zeros(1,n);
dC2=zeros(1,n);
I_ChR2=zeros(1,n);

for i = 1:n
    if optoStimulusMatrix(i) == 0 
        Ga1(i)=0;
        Ga2(i)=0;

        dO1(i)=0;        % dO1 dO2 dC2 需要求解
        dO2(i)=0;
        dC2(i)=0;

        I_ChR2(i)=0;

    elseif optoStimulusMatrix(i) == 1
        if optoStimulus ~= 0
            Ga1(i)     = r1 * (1-exp(-t/tau_ChR2L));    % Generate Ga(t) when light is ON
            Ga2(i)     = r2 * (1-exp(-t/tau_ChR2L));    % Generate Ga(t) when light is ON
        elseif   optoStimulus == 0 
            Ga1(i)     = r1 * (exp(-(t-optoStart)/tau_ChR2D)-exp(-t/tau_ChR2D)); 
            Ga2(i)     = r2 * (exp(-(t-optoStart)/tau_ChR2D)-exp(-t/tau_ChR2D)); 
        end
        
        % Force the ChR2 states when there is no sitmulus to prevent convergence
        % problems in the ODE solver at t < optoStart.
        
        if t > optoStart        
          Ga1(i)     = Ga1(i);
          Ga2(i)     = Ga2(i);
        else
          Ga1(i)     = 0;
          Ga2(i)     = 0;
        end 

        % Calculate the ChR2 state populations 
        dO1(i)         = Ga1(i) * (1-O1(i)-O2(i)-C2(i)) - (Gd1+e_ct) * O1(i) + e_tc * O2(i) ;
        dO2(i)         =                  e_ct * O1(i)  - (e_tc+Gd2) * O2(i) + Ga2(i) * C2(i) ;
        dC2(i)         =                                         Gd2 * O2(i) - (Gr_d+Ga2(i))* C2(i) ;
        
%         I_ChR2(i)      = -(O1(i) + gama*O2(i)) * G_rectify(v_soma(i));
        I_ChR2(i)      = -10*(O1(i) + gama*O2(i)) * G_rectify(v_soma(i));    % G_ChR2 = 1nS/um^2
        
    end
end

% Realistic parameters 
% threshold:  irradiance0 = 10^17 to 10^19 photons/(s*cm^2)
% ChR2_density            = 15, 150, 1100, 1300 pS/um^2      2013 Grossman % et al.   for unmyelinated segment it is about 15-150
% Aretinal_soma           = 200     um^2 (?)                 2013 Grossman et al. 
% Aretinal_axon           = 1200    um^2                     2013 Grossman et al. 
% Aretinal_axon           = 1200    um^2                     2013 Grossman et al. 
% ChR2_conductance_ttl    = ChR2_density*Aretinal_soma;
% ChR2_conductance_ttl, typically 12,13,15,20,30nS (~12000-30000pS)
% 1nS * 1mV = 1pA   -60mV*15nS = -900pA
% if use ChR2 density, it is about 150pS/um^2, or -60mV*0.15nS/um^2 = -9pA/um^2 = -0.09nA/um^2
% soma = 940um^2                   1995 Traub et al. CA3 interneuron model 
% dendrite cluster over 3000um^2   1995 Traub et al. CA3 interneuron model 
% soma = 2400um^2                  1994 Traub et al. CA3 pyramial neuron model
% dendrite cluster over 6000um^2   1994 Traub et al. CA3 pyramial neuron model


%%  neuron network
% 目前ConnectMatrix不作为输入参数，直接使用e_to_e_indices等参数作为传入结果

% 获取不同类型连接的突触前、突触后神经元序号范围1~n，长度与连接数相同
% eepre_indices  = ConnectMatrix(e_to_e_indices, 2);
% eepost_indices = ConnectMatrix(e_to_e_indices, 1);
% 
% eipre_indices  = ConnectMatrix(e_to_i_indices, 2);
% eipost_indices = ConnectMatrix(e_to_i_indices, 1);
% 
% iepre_indices  = ConnectMatrix(i_to_e_indices, 2);
% iepost_indices = ConnectMatrix(i_to_e_indices, 1);
% 
% iipre_indices  = ConnectMatrix(i_to_i_indices, 2);
% iipost_indices = ConnectMatrix(i_to_i_indices, 1);

% 注：此处发现了错误，突触前突触后神经元分别为ConnectMatrix的第一列、第二列

eepre_indices  = ConnectMatrix(e_to_e_indices, 1);
eepost_indices = ConnectMatrix(e_to_e_indices, 2);

eipre_indices  = ConnectMatrix(e_to_i_indices, 1);
eipost_indices = ConnectMatrix(e_to_i_indices, 2);

iepre_indices  = ConnectMatrix(i_to_e_indices, 1);
iepost_indices = ConnectMatrix(i_to_e_indices, 2);

iipre_indices  = ConnectMatrix(i_to_i_indices, 1);
iipost_indices = ConnectMatrix(i_to_i_indices, 2);

ds = s;
ds(:,3) = 0;
Isyni = zeros(1,n);

% 注意： eepre_indices是i个要计算的数值对应的神经元序号； e_to_e_indices是第i个要计算的数值对应的行号
for i = 1:length(e_to_e_indices)   
    ds(e_to_e_indices(i),3) = alpha_ampa * scale(v_soma(eepre_indices(i))) *(1 - s(e_to_e_indices(i),3)) - beta_ampa * s(e_to_e_indices(i),3); 
    Isyni(eepost_indices(i)) = Isyni(eepost_indices(i)) + g_ee * s(e_to_e_indices(i),3) .* (v_soma(eepost_indices(i)) - Esyn_ampa); 
end

% e -> i
for i = 1: length(e_to_i_indices)
    ds(e_to_i_indices(i),3) = alpha_ampa * scale(v_soma(eipre_indices(i)))* (1 - s(e_to_i_indices(i),3)) - beta_ampa * s(e_to_i_indices(i),3); 
    Isyni(eipost_indices(i)) = Isyni(eipost_indices(i)) + g_ei * s(e_to_i_indices(i),3) .* (v_soma(eipost_indices(i)) - Esyn_ampa); 
end

% i -> e
for i = 1: length(i_to_e_indices)
    ds(i_to_e_indices(i),3) = alpha_gaba * scale(v_soma(iepre_indices(i)))* (1 - s(i_to_e_indices(i),3)) - beta_gaba * s(i_to_e_indices(i),3); 
    Isyni(iepost_indices(i)) = Isyni(iepost_indices(i)) + g_ie * s(i_to_e_indices(i),3) .* (v_soma(iepost_indices(i)) - Esyn_gaba); 
end

% i -> i
for i = 1: length(i_to_i_indices)
    ds(i_to_i_indices(i),3) = alpha_gaba * scale(v_soma(iipre_indices(i)))* (1 - s(i_to_i_indices(i),3)) - beta_gaba * s(i_to_i_indices(i),3); 
    Isyni(iipost_indices(i)) = Isyni(iipost_indices(i)) + g_ii * s(i_to_i_indices(i),3) .* (v_soma(iipost_indices(i)) - Esyn_gaba); 
end

%% neuron network
% revised 1117 xuechun 去除电刺激部分影响
% 注释部分代码为多种刺激叠加
% for i = 1:n
%     if type_of_neuron(i) == 1
%         mNa(i)         = am_e(v_soma(i))/(am_e(v_soma(i))+bm_e(v_soma(i)));             % am bm 均为vsoma的函数
%         dv_soma(i)     = (-gNa_e*(mNa(i)^3)*hNa(i)*(v_soma(i)-ENa_e)-gK_e*(nK(i)^4)*(v_soma(i)-EK_e)-gL_e*(v_soma(i)-EL_e) ...
%                         -I_ChR2(i)-Isyni(i)+Ielectrical(i))/c_soma_e;
%     
%         dhNa(i)        = -(ah_e(v_soma(i))+bh_e(v_soma(i)))*hNa(i)+ah_e(v_soma(i));  
%         dnK(i)         = -(an_e(v_soma(i))+bn_e(v_soma(i)))*nK(i)+an_e(v_soma(i));    
%     elseif type_of_neuron(i) == 2
%         mNa(i)         = am_i(v_soma(i))/(am_i(v_soma(i))+bm_i(v_soma(i)));             % am bm 均为vsoma的函数
%         dv_soma(i)     = (-gNa_i*(mNa(i)^3)*hNa(i)*(v_soma(i)-ENa_i)-gK_i*(nK(i)^4)*(v_soma(i)-EK_i)-gL_i*(v_soma(i)-EL_i) ...
%                         -I_ChR2(i)-Isyni(i)+ Ielectrical(i))/c_soma_i;
%         
%         dhNa(i)        = 5*(-(ah_i(v_soma(i))+bh_i(v_soma(i)))*hNa(i)+ah_i(v_soma(i)));   % temperature factor = 5
%         dnK(i)         = 5*(-(an_i(v_soma(i))+bn_i(v_soma(i)))*nK(i)+an_i(v_soma(i)));    % temperature factor = 5
% 
%     end
% end

% 只有光刺激的精简部分
for i = 1:n
    if type_of_neuron(i) == 1
        mNa(i)         = am_e(v_soma(i))/(am_e(v_soma(i))+bm_e(v_soma(i)));             % am bm 均为vsoma的函数
        dv_soma(i)     = (-gNa_e*(mNa(i)^3)*hNa(i)*(v_soma(i)-ENa_e)-gK_e*(nK(i)^4)*(v_soma(i)-EK_e)-gL_e*(v_soma(i)-EL_e) ...
                        -I_ChR2(i)-Isyni(i)+Ielectrical(i))/c_soma_e;
    
        dhNa(i)        = -(ah_e(v_soma(i))+bh_e(v_soma(i)))*hNa(i)+ah_e(v_soma(i));  
        dnK(i)         = -(an_e(v_soma(i))+bn_e(v_soma(i)))*nK(i)+an_e(v_soma(i));    
    elseif type_of_neuron(i) == 2
        mNa(i)         = am_i(v_soma(i))/(am_i(v_soma(i))+bm_i(v_soma(i)));             % am bm 均为vsoma的函数
        dv_soma(i)     = (-gNa_i*(mNa(i)^3)*hNa(i)*(v_soma(i)-ENa_i)-gK_i*(nK(i)^4)*(v_soma(i)-EK_i)-gL_i*(v_soma(i)-EL_i) ...
                        -I_ChR2(i)-Isyni(i)+Ielectrical(i))/c_soma_i;
        
        dhNa(i)        = 5*(-(ah_i(v_soma(i))+bh_i(v_soma(i)))*hNa(i)+ah_i(v_soma(i)));   % temperature factor = 5
        dnK(i)         = 5*(-(an_i(v_soma(i))+bn_i(v_soma(i)))*nK(i)+an_i(v_soma(i)));    % temperature factor = 5

    end
end
%% return result
% mNa_zero = zeros(1,length(mNa));
returnMatrix = [dv_soma,dhNa,dnK,dO1,dO2,dC2,(ds(:,3))'];

returnMatrix = returnMatrix';

%% subfunction
function out = am_e(v_soma)
out         = 0.32*(v_soma+54)/(1-exp(-(v_soma+54)/4));

function out = bm_e(v_soma)
out         = -0.28*(v_soma+27)/(1-exp((v_soma+27)/5));

function out = ah_e(v_soma)
out         = 0.128*exp(-(v_soma+50)/18);

function out = bh_e(v_soma)
out         = 4/(1+exp(-(v_soma+27)/5));

function out = an_e(v_soma)
out         = 0.032*(v_soma+52)/(1-exp(-(v_soma+52)/5));

function out = bn_e(v_soma)
out         = 0.5*exp(-(v_soma+57)/40);

function out = am_i(v_somai)
out         = -0.1*(v_somai+35)/(exp(-0.1*(v_somai+35))-1);

function out = bm_i(v_somai)
out         = 4*exp(-(v_somai+60)/18);

function out = ah_i(v_somai)
out         = 0.07*exp(-0.05*(v_somai+58));

function out = bh_i(v_somai)
out         = 1/(exp(-0.1*(v_somai+28))+1);

function out = an_i(v_somai)
out         = -0.01*(v_somai+34)/(exp(-0.1*(v_somai+34))-1);

function out = bn_i(v_somai)
out         = 0.125*exp(-(v_somai+44)/80);

function out = scale(v_soma_pre) % 通用
out           = 1./(1+exp(-(v_soma_pre)/2));


function out = G_rectify(v_soma)
% out         = (v_soma-70)*(1-exp(-v_soma/40))/(v_soma/15);

out         = (1-exp(-v_soma/40))/(v_soma/15);

function compressedMatrix = sparseToCompressed(sparseMatrix)
    [row, col, val] = find(sparseMatrix);
    numNonZeros = nnz(sparseMatrix);  % 非零矩阵元素的数目
    numRows = size(sparseMatrix, 1); % 行数目
    
    compressedMatrix = zeros(numNonZeros,3);
    compressedMatrix(:, 1) = row;
    compressedMatrix(:, 2) = col;
    compressedMatrix(:, 3) = val;


function sparseMatrix = compressedToSparse(compressedMatrix)
    numRows = max(compressedMatrix(:,1));
    numCols = max(compressedMatrix(:,2));
    numNonZeros = size(compressedMatrix,1); 
    compressedData = compressedMatrix(:,3);
    
    row = compressedMatrix(1:numNonZeros);
    col = compressedMatrix(numNonZeros+1:2*numNonZeros);
    val = compressedMatrix(2*numNonZeros+1:end);
    
    sparseMatrix = sparse(row, col, val, numRows, numCols);

