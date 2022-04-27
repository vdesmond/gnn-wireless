clc
clear
clear all

data_size_set = [500,1000];%number of network layouts
link_num_set = [50]; %number of D2D pairs
for i = 1:length(link_num_set)
    num = link_num_set(i);%single-antenna transceivers pairs
    for j = 1:length(data_size_set)
        disp('####### Generate Training Data #######');
        data_size = data_size_set(j);
        [Channel,Label,Distance,Distance_quan, Tx, Ty, Rx, Ry]=generate(num,data_size);
        disp('#######Done #######');
         t = datestr(datetime('now'));
         disp(t);
         d = replace(t, {':'; '-'; ' '}, {''; ''; '_'});
         disp(d);
         save(sprintf('./mat/dataset_%d_%d_%s.mat',data_size, num, d),'Channel','Label','Distance','Distance_quan', 'Tx', 'Ty', 'Rx', 'Ry');
    end
end

%% Layout_coords format
%% An array of size MxN where
%%  M = 2*number of D2D pairs
%%  N = 2*number of network layouts
%%  First M/2 rows contain Transmitter coordinates
%%      For the i th network layout, column i contains Tx, i+1 contains Ty
%%  M/2+1 to M rows contain Receiver coordinates
%%      Similarly, for the i th network layout, column i contains Rx i+1 contains Ry
