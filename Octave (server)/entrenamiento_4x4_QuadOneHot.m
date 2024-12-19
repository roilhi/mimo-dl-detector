# ///////////////////////////////////////////////////////////////////////
# This MATLAB script genereates a DL model using a MIMO 4x4 configuration
# and the one-hot per antenna encoding strategy. It is used for the paper:
#
#  Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.;  Del-Puerto-Flores, J.A;
#  Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L. "Efficient 
#  Deep Learning-Based Detection Scheme for MIMO Communication System" 
#  Submitted to the Journal Sensors of MDPI
# 
#
# License: This code is licensed under the GPLv2 license. If you in any way
# use this code for research that results in publications, please cite our
# paper as described above.
#
#   Authors: Roilhi Frajo Ibarra Hernández (roilhi.ibarra@uaslp.mx)
#            Francisco Rubén Castillo-Soria (ruben.soria@uaslp.mx)
# ///////////////////////////////////////////////////////////////////////
warning('off','all');
clear
close all
clc

pkg load communications;
% símbolos qam para el dataset de entrenamiento
N = 1e4; % número de símbolos dataset de entrenamiento
M = 4; % orden de la modulación
qam_idx = 0:M-1; %índices de la modulación QAM
Nt = 4; % Número de transmisores
Nr = 4; % número de receptoras
input_size = 2*Nr;
output_size = M*Nt;
rand_sym_idx = randi(M^Nt,1,N);
%y = zeros(N,M^Nt); 
y = zeros(N,output_size);
X = zeros(N,2*Nt);
qam_sym = qammod(qam_idx,M);
alpha = 0.01; % tasa de aprendizaje

% Realiza el producto cartesiano (todas las combinaciones) del alfabeto
% es decir, todas las combinaciones en Nt antenas
[Ww, Xx, Yy, Zz] = ndgrid(qam_sym,qam_sym,qam_sym,qam_sym);
[aa, bb, cc, dd] = ndgrid(qam_idx+1, qam_idx+1,qam_idx+1, qam_idx+1);
prod_cart = [Ww(:) Xx(:) Yy(:) Zz(:)];
prod_cart_idx = [aa(:) bb(:) cc(:) dd(:)];
quad_one_hot = zeros(length(prod_cart),output_size);
for r=1:length(prod_cart)
    row_sel = quad_one_hot(r,:);
    idx_sel = prod_cart_idx(r,:);
    a1_one_hot = row_sel(1:M); % one hot for antenna 1
    a2_one_hot = row_sel(M+1:2*M); % one hot for antenna 2
    a3_one_hot = row_sel(2*M+1:3*M);
    a4_one_hot = row_sel(3*M+1:end);
    a1_one_hot(idx_sel(1)) = 1;
    a2_one_hot(idx_sel(2)) = 1;
    a3_one_hot(idx_sel(3)) = 1;
    a4_one_hot(idx_sel(4)) = 1;
    new_double_row = [a1_one_hot a2_one_hot a3_one_hot a4_one_hot];
    quad_one_hot(r,:) = new_double_row;
end

SNR_dB = 3; 
SNR_l = 10.^(SNR_dB./10);
No = 1;
idx_member = zeros(1,N);

for i=1:N
    %one_hot = y(i,:);
    %one_hot(rand_sym_idx(i)) = 1; %asigna un 1 en el símbolo seleccionado
    %y(i,:) = one_hot;
    sel_symbol = prod_cart(rand_sym_idx(i),:); % selecciona un símbolo
    y(i,:) = quad_one_hot(rand_sym_idx(i),:); 
    H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
    n = (No/sqrt(2))*(randn(Nr,1) + 1i*randn(Nr,1));
    n = (1/sqrt(SNR_l))*n;
    r_x = H*sel_symbol.';
    H_inv = pinv(H);
    r_x = H_inv*r_x+n;
    X(i,:) = [real(r_x.') imag(r_x.')];
end
orden = [1,5,2,6,3,7,4,8];% [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden); % [real(r1) imag(r1) real(r2) imag(r2)]

% 
%  for r=1:length(X)
%      idx_member(r) = find(ismember(idx_sign,X(r,:)<0,'rows')); %encuentra el cuadrante en la matriz one hot
%      one_hot = y(r,:);
%      one_hot(idx_member(r)) = 1; % asigna 1 en la combinación de signos que encontró
%      y(r,:) = one_hot;
%  end

% Normalización
X = X-mean(X(:));
X = X./std(X(:));

train_qty = round(0.8*length(X)); % obtiene la cantidad de filas correspondiente al 80%
test_qty = N-train_qty;

% 80% de las filas datos de entrenamiento
Xtrain = X(1:train_qty,:);
ytrain = y(1:train_qty,:);
%ytr_vls = ytrain';
%[~,idx_train] = max(ytr_vls); 
idx_train = rand_sym_idx(1:train_qty);
% 20% de las filas restantes, datos de validación
Xtest = X(train_qty+1:end,:);
ytest = y(train_qty+1:end,:);
%ytest_vls = ytest';
%[~,idx_test] = max(ytest_vls);
idx_test = rand_sym_idx(train_qty+1:end);
% valores iniciales de la red neuronal
%n_layers = 2;
n_neuronas_oculta = 1000; %10 neuronas capa oculta
%[a,b] = size(Xtrain);

n_epocas = 10e3;

% Arquitectura:
% input (4 neuronas) --- oculta (10 neuronas) --- salida (16 símbolos) 
% n[0] = 4 % número de features
% n[1] = 10 % número de neuronas capa oculta
% n[2] = 16 % número de salidas (solo tenemos 3 capas)
% pesos y bias de la capa 1 
% size(W^[l]) = (n[l] , n[l-1])

% Inicialización Xavier
xavier_limit = sqrt(6)/sqrt(input_size+output_size);
W1 = rand(n_neuronas_oculta,input_size);  %(10,4)
W1 = -xavier_limit+2*xavier_limit*W1;
% size(b^[l] = (n[l],1)
b1 = zeros(n_neuronas_oculta,1); %(10,1)
% pesos y bias de la capa 2
W2 = rand(n_neuronas_oculta,n_neuronas_oculta); % (16,10)
W2 = -xavier_limit+2*xavier_limit*W2;
b2 = zeros(n_neuronas_oculta,1); %(16,1)

W3 = rand(output_size,n_neuronas_oculta); % (16,10)
W3 = -xavier_limit+2*xavier_limit*W3;
b3 = zeros(output_size,1); %(16,1)


train_loss = zeros(1,n_epocas);
test_loss = zeros(1,n_epocas);
train_acc = zeros(1,n_epocas);
test_acc = zeros(1,n_epocas);

n_part_a3 = output_size/4;

for i=1:n_epocas
    % **************************
    % FORWARD PROPAGATION
    % **************************
    % Forward propagation
    % size(Z[l]) = (n[l],m)
    % Si es MATLAB 2020 en adelante es posible el broadcasting,
    % descomentar esta opción
    Z1 = (W1*Xtrain')+b1;
    % si el MATLAB < 2020 no es posible el broadcasting, descomentar la
    % línea
    %Z1 = W1*Xtrain';
    %[~,cols_Z1] = size(Z1);
    %b1 = repmat(b1,1,cols_Z1);
    %Z1 = Z1+b1;
    % función de activación ReLU después de la salida capa 1
    A1 = max(0,Z1); % (10,m)
    % Entrada a la capa 2 (multiplica por los pesos)
    % (16,10)(10,m) + (16,1)
    % Con broadcasting, activar esta opcion si es MATLAB > 2022
    Z2 = W2*A1+b2; % (16,m)
    % Sin broadcasting activar esta opción si MATLAB < 2022
    %Z2 = W2*A1;
    %[~,cols_Z2] = size(Z2);
    %b2 = repmat(b2,1,cols_Z2);
    %Z2 = Z2+b2;
    A2 = max(0,Z2); % ReLU capa 2
    Z3 = W3*A2+b3;
    % función softmax después de la salida capa 2
    %A2 = exp(Z2)./sum(exp(Z2)); % (16,m)
    %A2 = [A2_first_rows; A2_last_rows];
    A3 = 1./(1+exp(-Z3));
    % Extracting rows corresponding to each one hot label antenna
    A3_rows_a1 = A3(1:n_part_a3,:);
    A3_rows_a2 = A3((n_part_a3)+1:2*n_part_a3,:);
    A3_rows_a3 = A3((2*n_part_a3)+1:3*n_part_a3,:);
    A3_rows_a4 = A3((3*(n_part_a3))+1:end,:);
    % argmax(A2), selecciona el índice máximo del vector one hot
    [~, y_hat1] = max(A3_rows_a1);
    [~, y_hat2] = max(A3_rows_a2);
    [~, y_hat3] = max(A3_rows_a3);
    [~, y_hat4] = max(A3_rows_a4);
    [~, y_hat] = ismember([y_hat1' y_hat2' y_hat3' y_hat4'],prod_cart_idx,'rows');
    %[~, y_hat] = ismember((A2>0.5)',double_one_hot,'rows');
    % función de pérdidas o loss
    train_loss(i) = sqrt((1/train_qty)*sum((y_hat'-idx_train).^2));
    % accuracy (predicciones correctas / total)
    f = y_hat'==idx_train;
    train_acc(i) = length(find(f==1))/train_qty;
    % **************************
    % BACK PROPAGATION
    % **************************
    % dZ[l] = dA[l]*g[l]'(Z[l])
    dZ3 = A3 - ytrain'; % (16,m)
    %ytrain_bp = ytrain';
    %dZ2_1 = A2_first_rows - ytrain_bp(1:output_size/2,:);
    %dZ2_2 = A2_first_rows - ytrain_bp(output_size/2+1:end,:);
    %dZ2 = [dZ2_1; dZ2_2];
    % dW[l] = 1/m dZ[l] * A[l-1].T
    dW3 = (1/train_qty)*(dZ3*A2'); % (16,m)
    % derivada promedio bias db2
    db3 = (1/train_qty)*(sum(dZ3,2));

    % (10,16)(16,m) * (10*m)
    dZ2_prev = (W3'*dZ3); % (10,m)
    % la derivada de ReLU es f(x) = x>0 esa es g'(x) 
    % esta función regresa 1 para x>0 y 0 en otro caso
    dZ2 = dZ2_prev.*(Z2>0); %(10,m)
    dW2 = (1/train_qty)*(dZ2*A1');
    db2 = (1/train_qty)*(sum(dZ2,2));
    dZ1_prev = (W2'*dZ2); % (10,m)
    dZ1 = dZ1_prev.*(Z1>0); %(10,m)
    dW1 = (1/train_qty)*(dZ1*Xtrain); %(10,4)
    db1 = (1/train_qty)*sum(dZ1,2); % (10,1)

    % Actualiza los pesos
    W1 = W1 - alpha*dW1;
    b1 = b1 - alpha*db1;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    W3 = W3 - alpha*dW3;
    b3 = b3 - alpha*db3;
    % ****************************
    % Validación
    % ****************************
    % Con broadcasting para Z1 validación
    Z1_V = (W1*Xtest')+b1; % FC capa 1
    % Sin broadcasting para la validación
    A1_V = max(0,Z1_V); % ReLU(Z1)
    % con broadcasting
    Z2_V = W2*A1_V + b2; % FC capa 2
    A2_V = max(0,Z2_V);
    % Con broadcasting
    %A2_V = exp(Z2_V)./sum(exp(Z2_V),2); % softmax capa 2
    % Sin broadcasting
    %expA2V = exp(Z2_V);
    %sum_expA2V = repmat(sum(expA2V),size(expA2V,1),1);
    %A2_V = expA2V./sum_expA2V;
    Z3_V = W3*A2_V + b3; % FC capa 3
    A3_V = 1./(1+exp(-Z3_V));
    % Extracting rows corresponding to each one hot label antenna
    A3V_rows_a1 = A3_V(1:n_part_a3,:);
    A3V_rows_a2 = A3_V((n_part_a3)+1:2*n_part_a3,:);
    A3V_rows_a3 = A3_V((2*n_part_a3)+1:3*n_part_a3,:);
    A3V_rows_a4 = A3_V((3*(n_part_a3))+1:end,:);
    % argmax(A2), selecciona el índice máximo del vector one hot
    [~, y_hat1V] = max(A3V_rows_a1);
    [~, y_hat2V] = max(A3V_rows_a2);
    [~, y_hat3V] = max(A3V_rows_a3);
    [~, y_hat4V] = max(A3V_rows_a4);
    [~, y_hat_v] = ismember([y_hat1V' y_hat2V' y_hat3V' y_hat4V'], prod_cart_idx,'rows');
    test_loss(i) = sqrt((1/test_qty)*sum((y_hat_v'-idx_test).^2)); 
    f_test = y_hat_v' == idx_test;
    test_acc(i) = length(find(f_test==1))/test_qty;
    if rem(i,100)==0
        fprintf('******************************** \n');
        txt= 'Época %d | Train loss %2.2f | Test loss %2.2f | Train acc %2.2f | Test acc %2.2f\n';
        fprintf(txt,i,train_loss(i),test_loss(i), train_acc(i), test_acc(i));
    end
end

% Fill confusion matrix
clases = output_size;

cm = zeros(clases);

ytrue = idx_test;
ypred = y_hat_v;

for i=1:clases
    for j=1:clases
        for l=1:length(ytrue)
            if (ytrue(l) ==i)
                if(ypred(l) == j)
                    cm(i,j) = cm(i,j)+1;
                end
            end
        end
    end
end

F1 = 0;
sum_prec = 0;
sum_rec = 0;
for ii=1:clases;
    if (cm(ii,ii)==0)
        Prec = 0;
        Recall = 0;
        F1 = F1+0;
    else
        Prec = cm(ii,ii)/sum((cm(ii,:)));
        Recall = cm(ii,ii)/sum((cm(:,ii)));
        F1 = F1 + (2*Prec*Recall)/(Prec+Recall);
        sum_prec = sum_prec+Prec;
        sum_rec = sum_rec+Recall;
    end
end
F1_macro = F1/clases;
Prec_macro = sum_prec/clases;
Recall_macro = sum_rec/clases;

fprintf('------ Classification Report -------------------------------- \n');

txt2 = ' F1 macro average: %2.2f | Precision average:  %2.2f | Recall average: %2.2f \n';

fprintf(txt2,F1_macro,Prec_macro,Recall_macro);

save -mat7-binary 'modelMIMO4x4_QuadOneHot_serv.mat' 'W1' 'W2' 'b1' 'b2' 'W3' 'b3';
save -mat7-binary 'lossAcc_4x4_QuadOneHot.mat' 'train_loss' 'train_acc' 'test_loss' 'test_acc';


