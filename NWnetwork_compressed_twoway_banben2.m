function [ConnectMatrix]= NWnetwork_compressed_twoway_banben2(N,m,p,type_of_neuron)
% 用于直接生成压缩矩阵形式的NW网络连接
% 该网络为双向连接
% 生成网络后会自动绘制网络图像
%  2024/4/9 修正了绘图自动打印属性；增加了度分布统计图
matrix = zeros(N,N);
for i=m+1:N- m
    matrix(i,i- m:i+m)=1;
end
for i=1:m
    matrix(i,1:i+m)=1;
end
for i=N- m+1:N
    matrix(i,i- m:N)=1;
end
for i=1:m
    matrix(i,N- m+i:N)=1;
    matrix(N- m+i:N,i)=1;
end
% Random add edge

for i = 1:N
    matrix(i,i) = 0;
end

mmatrix = matrix;
addedges = [];

% 目前生成的网络本身就是双向连接的 4.10
for i = 1:N-1
    for j = i+1:N
        if(matrix(i,j) == 0)&&(rand<p)
            matrix(i,j) = 1;
            matrix(j,i) = 1;
            addedges = [addedges; i j];
        end
    end
end

degrees = sum(matrix, 2);
% 绘制统计直方图
figure
histogram(degrees, 'BinMethod', 'integers');
% xlabel('节点连接数量');
% ylabel('频数');
% title('网络度分布');

t=linspace(0,2*pi,N+1);
x=sin(t);
y=cos(t);
figure;
set(gcf,'color','w')
figure
plot(x,y,'o','markerfacecolor','k','MarkerEdgeColor','k'),hold on
% title('网络连接示意图')
for i=1:N
    for j=1:N
        if (matrix(i,j)==1)
            plot([x(i),x(j)],[y(i),y(j)],'k-','HandleVisibility','off');
        end
    end
end
for i = 1:length(addedges)
    plot([x(addedges(i,1)),x(addedges(i,2))],[y(addedges(i,1)),y(addedges(i,2))],'k-','HandleVisibility','off','Color','red');
end
axis([-1.05,1.05,-1.05,1.05])
axis square
axis off
sum(sum(matrix))
hold off


ConnectMatrix_temp = zeros(N,N);
n = N;
k = 1;
for i = 1:n
    for j = 1:n
        if matrix(i,j) == 0
            ConnectMatrix_temp(i,j) = 0;
        else
            if type_of_neuron(i) == 1 && type_of_neuron(j) == 1
                ConnectMatrix_temp(i,j) = 11;
            elseif type_of_neuron(i) == 1 && type_of_neuron(j) == 2
                ConnectMatrix_temp(i,j) = 12;
            elseif type_of_neuron(i) == 2 && type_of_neuron(j) == 1
                ConnectMatrix_temp(i,j) = 21;  
            elseif type_of_neuron(i) == 2 && type_of_neuron(j) == 2
                ConnectMatrix_temp(i,j) = 22;
            end

                connection_matrix(k,1) = i;                             % 第1列 i为发出刺激信号的神经元
                connection_matrix(k,2) = j;   % 第2列 j为收到刺激信号的神经元
                connection_matrix(k,3) = ConnectMatrix_temp(i,j);      % 根据i和j定位到该神经元连接的类型
                k = k+1;

        end
    end
end

ConnectMatrix = connection_matrix;


