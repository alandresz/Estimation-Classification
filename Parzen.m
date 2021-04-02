%% TRABAJO FINAL TEORIA DE DETECCION Y ESTIMACION %%
clc; close all; clear variables; 

%% Condiciones iniciales %%
w1 = 1;
w2 = 2;
P(w1) = 0.4;
P(w2) = 1 - P(w1);


%% Generacion de muestras %%
n_samples = 10^4;

X1 = 2 + 8*rand([n_samples,1]);
X2 = 2 + randn([n_samples,1]);

tic


figure
H1 = histogram(X1);
hold on
H2 = histogram(X2);
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\histog_iniciales_parzen.png','-dpng');

figure
count_1 = histogram(X1,'BinWidth',0.0001);

hold on
count_2 = histogram(X2,'BinWidth',count_1.BinWidth);

print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\histog_reales_parzen.png','-dpng');

% ALINEACION DE HISTOGRAMAS

parzen_window_vol = 5000;
zeros_extra = parzen_window_vol;

zeros_before_count_1 = ceil(abs(count_1.BinLimits(1) - count_2.BinLimits(1))*(1/count_1.BinWidth)) + zeros_extra;
zeros_after_count_2 = ceil(abs(count_1.BinLimits(2) - count_2.BinLimits(2))*(1/count_2.BinWidth)) + zeros_extra;

sample_count_1 = [zeros(1,zeros_before_count_1) count_1.Values zeros(1,zeros_extra) ];
sample_count_2 = [zeros(1,zeros_extra) count_2.Values zeros(1,zeros_after_count_2)  ];

sample_count_1 = sample_count_1(1:length(sample_count_2));

%% ESTIMACION DE DENSIDAD CON VENTANA DE PARZEN %%

parzen_window = ones(parzen_window_vol,1);


len_P = ceil( ((count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5) - (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) )/count_1.BinWidth);

P_1 = zeros(1,len_P);

for i = 1:(length(sample_count_1) - parzen_window_vol)
    P_1(i) = sample_count_1(i:i + parzen_window_vol - 1) * parzen_window;
end
P_1 = P_1/(n_samples*parzen_window_vol);


P_2 = zeros(1,len_P);

for i = 1:(length(sample_count_2) - parzen_window_vol)
    P_2(i) = sample_count_2(i:i + parzen_window_vol - 1) * parzen_window;
end
P_2 = P_2/(n_samples*parzen_window_vol);


P_x = P(w1)*P_1 + P(w2)*P_2;

%%%%%% Chequeo %%%%%%%%%%%%%%%%
P_x_w1_total = sum(P_1);
P_x_w2_total = sum(P_2);
P_x_total = sum(P_x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

support = (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) : count_1.BinWidth : (count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5);

disp('TIEMPO DE EJECUCION PARA ESTIMACION DE DENSIDAD')
toc

figure
plot(support,P(w1)*P_1,'b')
hold on
plot(support,P(w2)*P_2,'r')
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\estim_separadas_parzen.png','-dpng');

figure
plot(support,P_x)
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\estim_conjunta_parzen.png','-dpng');

%% Error Teorico %%

P_1_idxs = find(P_1);
P_2_idxs = find(P_2);

differences = zeros(1,P_2_idxs(length(P_2_idxs) - P_1_idxs(1)) + 1);

for n = P_1_idxs(1):P_2_idxs(length(P_2_idxs))
    differences(n-P_1_idxs(1)+1) = abs(P(w1)*P_1(n) - P(w2)*P_2(n));
end

[~,closestIndex] = min(differences);

intersec_index = closestIndex + P_1_idxs(1);

% P_e_1

P_e_1_array = zeros(1,intersec_index - P_1_idxs(1) + 1);

for n = P_1_idxs(1):intersec_index
    P_e_1_array(n - P_1_idxs(1) + 1) = P_1(n);
end

P_e_2_array = zeros(1,P_2_idxs(length(P_2_idxs)) - intersec_index + 1);

for n = intersec_index:P_1_idxs(length(P_1_idxs))
    P_e_2_array(n - intersec_index + 1) = P_2(n);
end


err_teor = P(w2)*sum(P_e_2_array) + P(w1)*sum(P_e_1_array);


%% Clasificador-Clasificacion

tic

nSamples = 10^2;

toClassifySamples = [ (2 + 8*rand([round(P(w1)*nSamples),1]))' (2 + randn([round(P(w2)*nSamples),1]))' ];
check_array = [ones(round(P(w1)*nSamples), 1)' 2*ones(round(P(w2)*nSamples), 1)'];
classify_array = zeros(1,length(toClassifySamples));

for i = 1:nSamples
    [minValue,closestIndex] = min(abs(support - toClassifySamples(i)));
    if P(w1)*P_1(closestIndex)>P(w2)*P_2(closestIndex)
        classify_array(i) = 1;
    elseif P(w1)*P_1(closestIndex)<P(w2)*P_2(closestIndex)
        classify_array(i) = 2;
    else
        continue
    end
end

errors = find(check_array - classify_array);
n_errors = length(errors);

e_bayes = n_errors/nSamples;

disp('error bayes: ')
disp(e_bayes)

%% Clasificacion por KNN

classify_array_knn = zeros(1,length(toClassifySamples));
Knn = 51;

total_samples = sample_count_1 + sample_count_2;

widen = 1000; % por si es necesario ensanchar el array de muestras por la longitud de la ventana

increase_window = 5;

for i = 1:nSamples
    
    total_samples = sample_count_1 + sample_count_2;
    [minValue,closestIndex] = min(abs(support - toClassifySamples(i)));
    
    window_vol = 1;
    n_neighbours = 0;
    j = 0;
    k = 0;
    while n_neighbours < Knn
        window = ones(window_vol,1);
        try
            n_neighbours = total_samples(closestIndex-j+k*widen:closestIndex+j+k*widen)*window;
            %t = t + 1;
        catch
            disp('extending sample array...')
            k = k+1;
            total_samples = [zeros(1,widen) total_samples zeros(1,widen)];
            %t = 0;
        end
        window_vol = window_vol + 2*increase_window;
        j = j+increase_window;
    end
    

    window_vol = window_vol - 2*increase_window;
    j = j-increase_window;
    

    
    aux_sample_count_1 = [zeros(1,k*widen) sample_count_1 zeros(1,k*widen)];
    aux_sample_count_2 = [zeros(1,k*widen) sample_count_2 zeros(1,k*widen)];
   

    if aux_sample_count_1(closestIndex-j+k*widen:closestIndex+j+k*widen)*window > aux_sample_count_2(closestIndex-j+k*widen:closestIndex+j+k*widen)*window
        classify_array_knn(i) = 1;
    elseif aux_sample_count_1(closestIndex-j+k*widen:closestIndex+j+k*widen)*window < aux_sample_count_2(closestIndex-j+k*widen:closestIndex+j+k*widen)*window
        classify_array_knn(i) = 2;
    else
        continue
    end
    
    n_neighbours_check(i,1) = aux_sample_count_1(closestIndex-j+k*widen:closestIndex+j+k*widen)*window;
    n_neighbours_check(i,2) = aux_sample_count_2(closestIndex-j+k*widen:closestIndex+j+k*widen)*window;
    n_neighbours_check(i,3) = n_neighbours_check(i,1) + n_neighbours_check(i,2);
    
    checkright(i) = aux_sample_count_1(closestIndex-j+k*widen:closestIndex+j+k*widen)*window + aux_sample_count_2(closestIndex-j+k*widen:closestIndex+j+k*widen)*window;
%     drawnow('update')
%     disp('checked')
%     
%     if (checkright(i) ~= Knn) && ( (checkright(i) ~= (Knn+1)) && (checkright(i) ~= (Knn+2)) )
%        disp('bad op 2')
%        disp('neighbours computed for sample nr')
%        disp(i)
%        disp('test sample')
%        disp(checkright(i))
%     end
    
    if checkright(i) == 0
       disp('bad op 2')
       disp('neighbours computed for sample nr')
       disp(i)
       disp('test sample')
       disp(checkright(i))
    end

end

errors_knn = find(check_array - classify_array_knn);
n_errors_knn = length(errors_knn);

e_knn = n_errors_knn/nSamples;

disp('error knn (Parzen): ')
disp(e_knn)


%% Error Teorico %%

P_1_idxs = find(P_1);
P_2_idxs = find(P_2);

differences = zeros(1,P_2_idxs(length(P_2_idxs) - P_1_idxs(1)) + 1);

for n = P_1_idxs(1):P_2_idxs(length(P_2_idxs))   
    differences(n-P_1_idxs(1)+1) = abs(P(w1)*P_1(n) - P(w2)*P_2(n));
end

[~,closestIndex] = min(differences);

intersec_index = closestIndex + P_1_idxs(1);

% P_e_1
P_e_1_array = zeros(1,intersec_index - P_1_idxs(1) + 1);

for n = P_1_idxs(1):intersec_index
    P_e_1_array(n - P_1_idxs(1) + 1) = P_1(n);
end

% P_e_2
P_e_2_array = zeros(1,P_2_idxs(length(P_2_idxs)) - intersec_index + 1);

for n = intersec_index:P_1_idxs(length(P_2_idxs))
    P_e_2_array(n - intersec_index + 1) = P_2(n);
end

err_teor = P(w2)*sum(P_e_2_array) + P(w1)*sum(P_e_1_array);

disp('error teorico:')
disp(err_teor)

disp('TIEMPO DE EJECUCION PARA ESTIMACIONES DE ERRORES')
toc