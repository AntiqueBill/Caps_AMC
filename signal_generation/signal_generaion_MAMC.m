close all;
clear all;
clc;

warning off

fc=3.5; %Carrier Frequency
fs=14;  %Sample Frequency
fd=0.1; %Code Rate
freqsep=0.15;  %Frequency Interval
N_code=15;  %Number of Symbols
length = 2000;%Final length of signals
N_samples_m = 300000;%Number of overlapped samples
num_classes = 10;

fc_max = 1.15;
fc_min = 0.85;

Ac_max = 1.1;
Ac_min = 0.9;

snr_max = 15;
snr_min = 0;
max_targets = 2;
min_targets = 1;

max_shift = fs*N_code/fd - length;

fprintf('Generating overlapped samples...\nMax_target = %d\n', max_targets);

x_train = zeros(N_samples_m, length);
y_train = zeros(N_samples_m, num_classes);

idx_tar = randi([min_targets, max_targets], 1, N_samples_m);
for i=1:N_samples_m
    if mod(i, 2000) == 0
        fprintf('   itr=%d\n',i);
    end
    class_i = randperm(10);
    class_i = class_i(1:idx_tar(i));
    fcc = unifrnd (fc_min, fc_max,size(class_i,2),1);
    Acc = unifrnd (Ac_min, Ac_max,size(class_i,2),1);
    shift = unifrnd (0, max_shift,size(class_i,2),1);
    y = zeros(idx_tar(i), length);
    for j =1:size(class_i,2)
        switch class_i(j)
            case 1
                yr=ask2(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 2
                yr=fsk2(N_code,fcc(j),fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 3
                yr=fsk4(N_code,fcc(j),fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 4
                yr=fsk8(N_code,fcc(j),fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 5
                yr=psk2(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 6
                yr=psk4(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 7
                yr=psk8(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 8
                yr=qam16(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 9
                yr=qam64(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
            case 10
                yr=msk(N_code,fs,fd,fcc(j),1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                y_train(i, class_i(j))=1;
        end
    end
    y_r = sum(y, 1);
    snr = randi([snr_min, snr_max],1);
    x_train(i,:) = awgn(y_r, snr, 'measured','db');
end

Ac = [Ac_min, Ac_max];
fc = [fc_min, fc_max];
snr = [snr_min, snr_max];
fprintf('Saving...\n');
x_train = x_train';
y_train = y_train';
save('../samples/dataset_MAMC','x_train','y_train','Ac','fc', 'snr','length','-v7.3')