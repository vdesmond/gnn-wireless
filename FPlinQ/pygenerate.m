function pygenerate(train_num, val_num, test_num, d2d)

    data_size_set = [train_num,val_num,test_num];%number of network layouts
    link_num_set = [d2d]; %number of D2D pairs
    for i = 1:length(link_num_set)
        num = link_num_set(i);%single-antenna transceivers pairs
        for j = 1:length(data_size_set)
            disp('####### Generate Training Data #######');
            data_size = data_size_set(j);
            [Channel,Label,Distance,Distance_quan, Tx, Ty, Rx, Ry]=generate(num,data_size);
            disp('#######Done #######');
            save(sprintf('./mat/dataset_%d_%d.mat',data_size, num),'Channel','Label','Distance','Distance_quan', 'Tx', 'Ty', 'Rx', 'Ry');
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
end 
