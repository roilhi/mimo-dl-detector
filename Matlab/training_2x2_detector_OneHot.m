# ///////////////////////////////////////////////////////////////////////
# This MATLAB script genereates a DL model using a MIMO 2x2 configuration
# and the one-hot encoding strategy. It is used for the paper:
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
clear
close all
clc
% Generate symbols for the training dataset 
N = 1e4; % number of training symbols
M = 4; % modulation order
qam_idx = 0:M-1; %QAM modulation indices
Nt = 2; % % Number of Tx antennas
Nr = 2; % Number of Rx antennas
rand_sym_idx = randi(M^Nt,1,N); % random indices for the QAM combinations
% Initializing data and labels arrays
y = zeros(N,M^Nt); 
X = zeros(N,2*Nt);
qam_sym = qammod(qam_idx,M);
alpha = 0.01; % learning rate

% Cartesian product for all the combinations of 
% transmitted symbols M^Nt
[Xx, Yy] = meshgrid(qam_sym,qam_sym);
prod_cart = [Xx(:) Yy(:)];
real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;

idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2)];

SNR_dB = 3; % SNR for add noise to training data
SNR_l = 10.^(SNR_dB./10);
No = 1;
% generating data X and one-hot encoded "y"
idx_member = zeros(1,N);
for i=1:N
    sel_symbol = prod_cart(rand_sym_idx(i),:); % selects a symbol
    H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
    n = (No/sqrt(2))*(randn(Nr,1) + 1i*randn(Nr,1));
    n = (1/sqrt(SNR_l))*n;
    r_x = H*sel_symbol.';
    H_inv = pinv(H);
    r_x = H_inv*r_x+n;
    X(i,:) = [real(r_x.') imag(r_x.')];
end

orden = [1,3,2,4]; % [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden); % [real(r1) imag(r1) real(r2) imag(r2)]


 for r=1:length(X)
     idx_member(r) = find(ismember(idx_sign,X(r,:)<0,'rows')); % finds the one hot combination or quadrant
     one_hot = y(r,:);
     one_hot(idx_member(r)) = 1; % assings the 1 for the combination
     y(r,:) = one_hot;
 end

% Normalize 
X = X-mean(X(:));
X = X./std(X(:));

% generates training data and labels
train_qty = round(0.8*length(X)); 
test_qty = N-train_qty;
% train-test split datasets
Xtrain = X(1:train_qty,:);
ytrain = y(1:train_qty,:);
ytr_vls = ytrain';
[~,idx_train] = max(ytr_vls); 


Xtest = X(train_qty+1:end,:);
ytest = y(train_qty+1:end,:);
ytest_vls = ytest';
[~,idx_test] = max(ytest_vls);
%idx_test = rand_sym_idx(train_qty+1:end);

%n_layers = 2;
% -------------------
% Neural network hyperparameters
n_neuronas_oculta = 100; % number of hidden units
%[a,b] = size(Xtrain);
input_size = 2*Nt;
output_size = M^Nt;
n_epocas = 2000; % epochs

% Xavier initialization
xavier_limit = sqrt(6)/sqrt(input_size+output_size);
W1 = rand(n_neuronas_oculta,input_size);  %(10,4)
W1 = -xavier_limit+2*xavier_limit*W1;
% size(b^[l] = (n[l],1)
b1 = zeros(n_neuronas_oculta,1); %(10,1)
% pesos y bias de la capa 2
W2 = rand(output_size,n_neuronas_oculta); % (16,10)
W2 = -xavier_limit+2*xavier_limit*W2;
b2 = zeros(output_size,1); %(16,1)

train_loss = zeros(1,n_epocas);
test_loss = zeros(1,n_epocas);
train_acc = zeros(1,n_epocas);
test_acc = zeros(1,n_epocas);

for i=1:n_epocas
    % **************************
    % FORWARD PROPAGATION
    % **************************
    % Forward propagation
    % size(Z[l]) = (n[l],m)
    % Si es MATLAB 2020 en adelante es posible el broadcasting,
    % descomentar esta opci�n
    %Z1 = (W1*Xtrain')+b1;
    % si el MATLAB < 2020 no es posible el broadcasting, descomentar la
    % l�nea
    Z1 = W1*Xtrain';
    [~,cols_Z1] = size(Z1);
    b1 = repmat(b1,1,cols_Z1);
    Z1 = Z1+b1;
    % funci�n de activaci�n ReLU despu�s de la salida capa 1
    A1 = max(0,Z1); % (10,m)
    % Entrada a la capa 2 (multiplica por los pesos)
    % (16,10)(10,m) + (16,1)
    % Con broadcasting, activar esta opcion si es MATLAB > 2022
    %Z2 = W2*A1+b2; % (16,m)
    % Sin broadcasting activar esta opci�n si MATLAB < 2022
    Z2 = W2*A1;
    [~,cols_Z2] = size(Z2);
    b2 = repmat(b2,1,cols_Z2);
    Z2 = Z2+b2;
    % funci�n softmax despu�s de la salida capa 2
    %A2 = exp(Z2)./sum(exp(Z2)); % (16,m)
    % si MATLAB < 2022, sin broadcasting
    expA2 = exp(Z2);
    [rowsA2, colsA2] = size(expA2);
    sum_expZ2 = repmat(sum(exp(Z2)),rowsA2,1);
    A2 = exp(Z2)./sum_expZ2;
    % argmax(A2), selecciona el �ndice m�ximo del vector one hot
    [~, y_hat] = max(A2);
    % funci�n de p�rdidas o loss
    train_loss(i) = (1/train_qty)*sum((y_hat-idx_train).^2);
    % accuracy (predicciones correctas / total)
    f = y_hat==idx_train;
    train_acc(i) = length(find(f==1))/train_qty;
    % **************************
    % BACK PROPAGATION
    % **************************
    % dZ[l] = dA[l]*g[l]'(Z[l])
    dZ2 = A2 - ytrain'; % (16,m)
    % dW[l] = 1/m dZ[l] * A[l-1].T
    dW2 = (1/train_qty)*(dZ2*A1'); % (16,m)
    % derivada promedio bias db2
    db2 = (1/train_qty)*(sum(dZ2,2));
    % (10,16)(16,m) * (10*m)
    dZ1_prev = (W2'*dZ2); % (10,m)
    % la derivada de ReLU es f(x) = x>0 esa es g'(x) 
    % esta funci�n regresa 1 para x>0 y 0 en otro caso
    dZ1 = dZ1_prev.*(Z1>0); %(10,m)
    dW1 = (1/train_qty)*(dZ1*Xtrain); %(10,4)
    db1 = (1/train_qty)*sum(dZ1,2); % (10,1)
    % Si no se aplica broadcasting (MATLAB > 2020), regresar b1,b2
    % a su tama�o real
    b1 = b1(:,1);
    b2 = b2(:,1);
    % Actualiza los pesos
    W1 = W1 - alpha*dW1;
    b1 = b1 - alpha*db1;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    % ****************************
    % Validaci�n
    % ****************************
    % Con broadcasting para Z1 validaci�n
    %Z1_V = (W1*Xtest')+b1; % FC capa 1
    % Sin broadcasting para la validaci�n
    Z1_V = W1*Xtest';
    [~,cols_Z1V] = size(Z1_V);
    b1 = repmat(b1,1,cols_Z1V);
    Z1_V = Z1_V + b1;
    A1_V = max(0,Z1_V); % ReLU(Z1)
    % con broadcasting
    %Z2_V = W2*A1_V + b2; % FC capa 2
    % sin broadcasting
    Z2_V = W2*A1_V;
    [~,cols_Z2V] = size(Z2_V);
    b2 = repmat(b2,1,cols_Z2V);
    Z2_V = Z2_V + b2;
    % Con broadcasting
    %A2_V = exp(Z2_V)./sum(exp(Z2_V),2); % softmax capa 2
    % Sin broadcasting
    expA2V = exp(Z2_V);
    sum_expA2V = repmat(sum(expA2V),size(expA2V,1),1);
    A2_V = expA2V./sum_expA2V;
    [~,y_hat_v] = max(A2_V); % argmax(A2) o �nidice m�xima probabilidad
    test_loss(i) = (1/test_qty)*sum((y_hat_v-idx_test).^2); 
    f_test = y_hat_v == idx_test;
    test_acc(i) = length(find(f_test==1))/test_qty;
    if rem(i,100)==0
        fprintf('******************************** \n');
        txt= '�poca %d | Train loss %2.2f | Test loss %2.2f | Train acc %2.2f | Test acc %2.2f\n';
        fprintf(txt,i,train_loss(i),test_loss(i), train_acc(i), test_acc(i));
    end
    % Regresar tama�os reales b1,b2 por el broadcasting
    b1 = b1(:,1);
    b2 = b2(:,1);
end

figure
title('Loss curves')
plot(train_loss,'LineWidth',2), grid on, hold on,
plot(test_loss,'--r','LineWidth',2),
xlabel('�pocas')
ylabel('Loss')
legend('train loss','test loss')

figure
title('Curvas de accuracy')
plot(train_acc, 'LineWidth',2), grid on, hold on,
plot(test_acc,'--r','LineWidth',2),
xlabel('�pocas')
ylabel('Accuracy')
legend('train acc','test acc')

save('modelMIMO_2x2_4QAM_10dB.mat','W1','W2','b1','b2');


ytrue = idx_test;
ypred = y_hat_v;
clases = output_size;
cm = zeros(clases);
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
sum_recall = 0;
for ii=1:clases;
    if (cm(ii,ii)==0)
        Prec = 0;
        Recall = 0;
        F1 = F1+0;
    else
        Prec = cm(ii,ii)/sum((cm(ii,:)));
        Recall = cm(ii,ii)/sum((cm(:,ii)));
        F1 = F1 + (2*Prec*Recall)/(Prec+Recall);
        sum_prec = sum_prec + Prec;
        sum_recall = sum_recall + Recall;
    end
end

F1_macro = F1/clases;
Prec_macro = sum_prec/clases;
Recall_macro = sum_recall/clases;

fprintf('------ Classification Report -------------------------------- \n');

txt2 = ' F1 macro average: %2.2f | Precision average:  %2.2f | Recall average: %2.2f \n';

fprintf(txt2,F1_macro,Prec_macro,Recall_macro);