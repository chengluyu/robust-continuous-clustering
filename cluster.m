function cluster(data)
    path = strcat('Data/', data, '.mat');
    [rcc, nc, opttime, Y, n] = RCC(path, 100, 4);
    [rccdr, nc, opttime, Y, n] = RCCDR(path, 100, 4);
    save(strcat('Data/', data, '_eval.mat'));