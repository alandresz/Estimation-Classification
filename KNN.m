%% TRABAJO FINAL TEORIA DE DETECCION Y ESTIMACION %%
clc; close all; clear variables; 

%% Condiciones iniciales %%
w1 = 1;
w2 = 2;
P(w1) = 0.4;
P(w2) = 1 - P(w1);

knn = 200;


%% Generacion de muestras %%
n_samples = 10^4;

X1 = 2 + 8*rand([n_samples,1]);
X2 = 2 + randn([n_samples,1]);

tic;

% Histogramas iniciales, para comparar luego con las densidades estimadas
figure;
H1 = histogram(X1);
hold on
H2 = histogram(X2);
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\histog_iniciales_knn.png','-dpng');

%PRUEBA: X1 = round(X1,3);
%PRUEBA: X2 = round(X2,3);

figure;
count_1 = histogram(X1,'BinWidth',0.0001);

hold on
count_2 = histogram(X2,'BinWidth',count_1.BinWidth);
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\histog_reales_knn.png','-dpng');


%% ALINEACION DE HISTOGRAMAS

parzen_window_vol = 1000;        % ceros pre y post muestras, para alinearlas
zeros_extra = parzen_window_vol;

zeros_before_count_1 = ceil(abs(count_1.BinLimits(1) - count_2.BinLimits(1))*(1/count_1.BinWidth)) + zeros_extra;
zeros_after_count_2 = ceil(abs(count_1.BinLimits(2) - count_2.BinLimits(2))*(1/count_2.BinWidth)) + zeros_extra;

sample_count_1 = [zeros(1,zeros_before_count_1) count_1.Values zeros(1,zeros_extra) ];
sample_count_2 = [zeros(1,zeros_extra) count_2.Values zeros(1,zeros_after_count_2)  ];

sample_count_1 = sample_count_1(1:length(sample_count_2));

%% ESTIMACION DE DENSIDAD CON KNN %%

N = find(sample_count_1);

%PRUEBA: len_P = ceil( ((count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5) - (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) )/count_1.BinWidth);

%PRUEBA: P_1 = zeros(1,len_P);

KNN = knn;


max_window_vol = KNN*1.5*10^2; % Valor maximo de ventana, para controlar la velocidad de ejecucion

%FALLIDO: for i = (ceil(zeros_before_count_1) + 1): KNN :(length(sample_count_1) - ceil(zeros_extra))
increase_window = 5; 

for i = 1: KNN :length(sample_count_1) % Me desplazo sobre el juego de muestras cada KNN muestras, sino el script tarda mucho
    window_vol = 1;
    n_neighbours = 0;
    j = 0;
    
    while n_neighbours < knn
        window = ones(window_vol,1);
        try
            n_neighbours = sample_count_1(i-j:i+j)*window;
        catch
            window_vol = max_window_vol;
            break
        end
        window_vol = window_vol + 2*increase_window;
        if window_vol >= max_window_vol
            break
        end
        j = j+increase_window;
    end
    
    %FALLIDO: P_1(  (i - (ceil(zeros_before_count_1) + 1 ) + KNN)/KNN  ) = 1/((window_vol/2) * n_samples * count_1.BinWidth);
    %FALLIDO: P_1( i - ceil(zeros_before_count_1) ) = 1/((window_vol/2)*n_samples);

    P_1(  (i - 1 + KNN)/KNN  ) = 1/((window_vol/2) * n_samples * count_1.BinWidth);

    
 % Chequeo de avance sobre juego de muestras y sobre ancho de ventana
    i
    j
end



%FALLIDO: for i = (ceil(zeros_extra) + 1): KNN :(length(sample_count_2) - ceil(zeros_before_count_1))
for i = 1 : KNN : length(sample_count_2) % Me desplazo sobre el juego de muestras cada KNN muestras, sino el script tarda mucho
    window_vol = 1;
    n_neighbours = 0;
    j = 0;
    
    while n_neighbours < KNN
        window = ones(window_vol,1);
        try
            n_neighbours = sample_count_2(i-j:i+j)*window;
        catch
            window_vol = max_window_vol;
            break
        end
        window_vol = window_vol + 2*increase_window;
        if window_vol >= max_window_vol
            break
        end
        j = j+increase_window;
    end
    
    %FALLIDO: P_1(  (i - (ceil(zeros_before_count_1) + 1 ) + KNN)/KNN  ) = 1/(j*n_samples*count_1.BinWidth);
    %FALLIDO: P_2( (i - (ceil(zeros_extra) + 1 ) + (KNN) )/(KNN)  ) = 1/( (window_vol/2) * n_samples * count_2.BinWidth);
 
    P_2( (i - 1 + KNN)/ KNN ) = 1/( (window_vol/2) * n_samples * count_2.BinWidth);

% Control de avance sobre juego de muestras y sobre ancho de ventana
    i
    j
end


% Soporte de las densidades estimadas

%FALLIDO: support = (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) : ( (count_1.BinLimits(2) - count_2.BinLimits(1) + zeros_extra*count_2.BinWidth )/length(P_2)) : (count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5 - ( (count_1.BinLimits(2) - count_2.BinLimits(1))/length(P_2)));
support = (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) : ( (count_1.BinLimits(2) - count_2.BinLimits(1) + zeros_extra*count_2.BinWidth )/length(P_2)) : (count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5 - ( (count_1.BinLimits(2) - count_2.BinLimits(1))/length(P_2)));

%% Densidad

P_x = P(w1)*P_1 + P(w2)*P_2;

disp('TIEMPO DE EJECUCION PARA ESTIMACION DE DENSIDAD')
toc

P_1 = P_1(1:length(P_2));

figure
plot(support, P_1, 'b')
hold on
plot(support, P_2, 'r')
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\estimaciones_separadas_knn.png','-dpng');


figure
plot(support, P_x, 'g')
print('C:\Users\A\Documents\facultad\tde\tpfinal\Informe\Imagenes\estimacion_conjunta_knn.png','-dpng');

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
Knn = 100;

total_samples = sample_count_1 + sample_count_2;
support = (count_2.BinLimits(1) - zeros_extra*count_2.BinWidth*0.5) : count_1.BinWidth : (count_1.BinLimits(2) + zeros_extra*count_2.BinWidth*0.5);

widen = 1000; % por si es necesario ensanchar el array de muestras por la longitud de la ventana

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
            % t = t + 1;
        catch
            %disp('bad operation')
            k = k+1;
            total_samples = [zeros(1,widen) total_samples zeros(1,widen)];
            % t = 0;
        end
        window_vol = window_vol + 2*increase_window;
        j = j + increase_window;
    end
    

    window_vol = window_vol - 2*increase_window;
    j = j - increase_window;
    
    
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
%     if (checkright(i) ~= Knn) && ((checkright(i) ~= Knn + 1) && (checkright(i) ~= Knn + 2))
%        disp('bad op 2')
%        disp('neighbours computed for')
%        disp(i)
%        disp('test sample')
%        disp(checkright(i))
%     end
    
    if checkright(i) == 0
       disp('bad op 2')
       disp('neighbours computed for')
       disp(i)
       disp('test sample')
       disp(checkright(i))
    end

end

errors_knn = find(check_array - classify_array_knn);
n_errors_knn = length(errors_knn);

e_knn = n_errors_knn/nSamples;

disp('error knn (KNN): ')
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

for n = intersec_index:P_1_idxs(length(P_1_idxs))
    P_e_2_array(n - intersec_index + 1) = P_2(n);
end

err_teor = (P(w2)*sum(P_e_2_array) + P(w1)*sum(P_e_1_array))/sum(P_x);

disp('error teorico:')
disp(err_teor)

disp('TIEMPO DE EJECUCION PARA ESTIMACIONES DE ERRORES')
toc