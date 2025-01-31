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
end 
