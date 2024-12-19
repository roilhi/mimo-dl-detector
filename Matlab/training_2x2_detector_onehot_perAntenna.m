# ///////////////////////////////////////////////////////////////////////
# This MATLAB script genereates a DL model using a MIMO 2x2 configuration
# and the one-hot encoding per antenna strategy. It is used for the paper:
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
N = 1e5; % number of training symbols
M = 4; % QAM modulation order
qam_idx = 0:M-1; % QAM modulation indices
Nt = 2; % Number of Tx antennas
Nr = 2; % Number of Rx antennas
input_size = 2*Nr; % input size for the neural network
output_size = M*Nt; % output size for the label vector
rand_sym_idx = randi(M^Nt,1,N); % random indices for the QAM combinations
%y = zeros(N,M^Nt); 
% Initializing data and labels arrays
y = zeros(N,output_size);
X = zeros(N,2*Nt);
qam_sym = qammod(qam_idx,M);
alpha = 0.01; % learning rate

% Cartesian product for all the combinations of 
% transmitted symbols M^Nt
[Xx, Yy] = meshgrid(qam_sym,qam_sym);
[aa, bb] = ndgrid(qam_idx+1, qam_idx+1);
prod_cart = [Xx(:) Yy(:)];
prod_cart_idx = [aa(:) bb(:)];
double_one_hot = zeros(length(prod_cart),output_size);
% one-hot per antenna encoding
for r=1:length(prod_cart)
    row_sel = double_one_hot(r,:);
    idx_sel = prod_cart_idx(r,:);
    first_one_hot = row_sel(1:M);
    second_one_hot = row_sel(M+1:end);
    first_one_hot(idx_sel(1)) = 1;
    second_one_hot(idx_sel(2)) = 1;
    new_double_row = [first_one_hot second_one_hot];
    double_one_hot(r,:) = new_double_row;
end

SNR_dB = 3; % SNR for the training data
SNR_l = 10.^(SNR_dB./10);
No = 1;
idx_member = zeros(1,N);
% generates training data and labels
for i=1:N
    sel_symbol = prod_cart(rand_sym_idx(i),:); % select a symbol
    y(i,:) = double_one_hot(rand_sym_idx(i),:); 
    H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
    n = (No/sqrt(2))*(randn(Nr,1) + 1i*randn(Nr,1));
    n = (1/sqrt(SNR_l))*n;
    r_x = H*sel_symbol.';
    H_inv = pinv(H);
    r_x = H_inv*r_x+n;
    X(i,:) = [real(r_x.') imag(r_x.')];
end
% puts an order of first real part and then imaginary part
orden = [1,3,2,4]; % [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden); % [real(r1) imag(r1) real(r2) imag(r2)]

% 
%  for r=1:length(X)
%      idx_member(r) = find(ismember(idx_sign,X(r,:)<0,'rows')); %encuentra el cuadrante en la matriz one hot
%      one_hot = y(r,:);
%      one_hot(idx_member(r)) = 1; % asigna 1 en la combinaci�n de signos que encontr�
%      y(r,:) = one_hot;
%  end

% Normalize data (normal distribution)
X = X-mean(X(:));
X = X./std(X(:));

% Split training and test/validation subsets (by 80%)
train_qty = round(0.8*length(X)); 
test_qty = N-train_qty;

Xtrain = X(1:train_qty,:);
ytrain = y(1:train_qty,:);
%ytr_vls = ytrain';
%[~,idx_train] = max(ytr_vls); 
idx_train = rand_sym_idx(1:train_qty);
% 20% de las filas restantes, datos de validaci�n
Xtest = X(train_qty+1:end,:);
ytest = y(train_qty+1:end,:);
%ytest_vls = ytest';
%[~,idx_test] = max(ytest_vls);
idx_test = rand_sym_idx(train_qty+1:end);

% -------------------
% Neural network hyperparameters

%n_layers = 2;
n_neuronas_oculta = 100; % number of hidden units
%[a,b] = size(Xtrain);

n_epocas = 2000; % number of epochs


% Xavier initialization
xavier_limit = sqrt(6)/sqrt(input_size+output_size);
W1 = rand(n_neuronas_oculta,input_size);  %(10,4)
W1 = -xavier_limit+2*xavier_limit*W1;
% size(b^[l] = (n[l],1)
b1 = zeros(n_neuronas_oculta,1); %(10,1)
% Layer 2 biases and weights
W2 = rand(output_size,n_neuronas_oculta); % (16,10)
W2 = -xavier_limit+2*xavier_limit*W2;
b2 = zeros(output_size,1); %(16,1)

% initialize vectors for loss and acc curves
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
    % If MATLAB > 2020, uncomment for broadcasting
    %Z1 = (W1*Xtrain')+b1;
    % If MATLAB < 2020, uncomment since it does not include
    % broadcasting
    Z1 = W1*Xtrain';
    [~,cols_Z1] = size(Z1);
    b1 = repmat(b1,1,cols_Z1);
    Z1 = Z1+b1;
    % ReLU activation
    A1 = max(0,Z1); % (10,m)
    % Layer 2
    % Uncomment if MATLAB > 2022 (broadcasting)
    %Z2 = W2*A1+b2; % (16,m)
    % Uncomment lines if MATLAB < 2022 (no broadcasting)
    Z2 = W2*A1;
    [~,cols_Z2] = size(Z2);
    b2 = repmat(b2,1,cols_Z2);
    Z2 = Z2+b2;
    % Split sigmoid function for the output layer
    A2 = 1./(1+exp(-Z2));
    A2_first_rows = A2(1:output_size/2,:);
    A2_last_rows = A2((output_size/2)+1:end,:);
    % argmax(A2) for each split of rows
    [~, y_hat1] = max(A2_first_rows);
    [~, y_hat2] = max(A2_last_rows);
    [~, y_hat] = ismember([y_hat1' y_hat2'],prod_cart_idx,'rows');
    %[~, y_hat] = ismember((A2>0.5)',double_one_hot,'rows');
    % Loss function and accuracy
    train_loss(i) = (1/train_qty)*sum((y_hat'-idx_train).^2);
    % accuracy (predicciones correctas / total)
    f = y_hat'==idx_train;
    train_acc(i) = length(find(f==1))/train_qty;
    % **************************
    % BACK PROPAGATION
    % **************************
    % dZ[l] = dA[l]*g[l]'(Z[l])
    dZ2 = A2 - ytrain'; % (16,m)
    %ytrain_bp = ytrain';
    %dZ2_1 = A2_first_rows - ytrain_bp(1:output_size/2,:);
    %dZ2_2 = A2_first_rows - ytrain_bp(output_size/2+1:end,:);
    %dZ2 = [dZ2_1; dZ2_2];
    % dW[l] = 1/m dZ[l] * A[l-1].T
    dW2 = (1/train_qty)*(dZ2*A1'); % (16,m)
    % derivate or gradients of bias
    db2 = (1/train_qty)*(sum(dZ2,2));
    % (10,16)(16,m) * (10*m)
    dZ1_prev = (W2'*dZ2); % (10,m)
    % derivarive of ReLU is f(x) = x>0  
    dZ1 = dZ1_prev.*(Z1>0); %(10,m)
    dW1 = (1/train_qty)*(dZ1*Xtrain); %(10,4)
    db1 = (1/train_qty)*sum(dZ1,2); % (10,1)
    % uncomment of MATLAB > 2020, so b1,b2
    % are turned back to its original dimensions
    b1 = b1(:,1);
    b2 = b2(:,1);
    % Weights and biases update
    W1 = W1 - alpha*dW1;
    b1 = b1 - alpha*db1;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    % ****************************
    % Validation or inference
    % ****************************
    % Broadcasting if MATLAB > 2020
    %Z1_V = (W1*Xtest')+b1; % layer 1 FC
    % Without broadcasting (MATLAB < 2020)
    Z1_V = W1*Xtest';
    [~,cols_Z1V] = size(Z1_V);
    b1 = repmat(b1,1,cols_Z1V);
    Z1_V = Z1_V + b1;
    A1_V = max(0,Z1_V); % ReLU(Z1)
    % with broadcasting
    %Z2_V = W2*A1_V + b2; % FC capa 2
    % without broadcasting
    Z2_V = W2*A1_V;
    [~,cols_Z2V] = size(Z2_V);
    b2 = repmat(b2,1,cols_Z2V);
    Z2_V = Z2_V + b2;
    % Validation of last layer
    A2_V = 1./(1+exp(-Z2_V));
    A2V_first_rows = A2_V(1:output_size/2,:);
    A2V_last_rows = A2_V(output_size/2+1:end,:);

    [~, y_hatv1] = max(A2V_first_rows);
    [~, y_hatv2] = max(A2V_last_rows);
    %[~,y_hat_v] = max(A2_V); % argmax(A2) o �nidice m�xima probabilidad
    [~, y_hat_v] = ismember([y_hatv1' y_hatv2'],prod_cart_idx,'rows');
    test_loss(i) = (1/test_qty)*sum((y_hat_v'-idx_test).^2); 
    f_test = y_hat_v' == idx_test;
    test_acc(i) = length(find(f_test==1))/test_qty;
    % print out what's happening 
    if rem(i,100)==0
        fprintf('******************************** \n');
        txt= '�poca %d | Train loss %2.2f | Test loss %2.2f | Train acc %2.2f | Test acc %2.2f\n';
        fprintf(txt,i,train_loss(i),test_loss(i), train_acc(i), test_acc(i));
    end 
    % put back biases due to broadcasting
    b1 = b1(:,1);
    b2 = b2(:,1);
end

figure
title('Loss curves')
plot(train_loss,'LineWidth',2), grid on, hold on,
plot(test_loss,'--r','LineWidth',2),
xlabel('Epochs')
ylabel('Loss')
legend('train loss','test loss')

figure
title('Accuracy curves')
plot(train_acc, 'LineWidth',2), grid on, hold on,
plot(test_acc,'--r','LineWidth',2),
xlabel('epochs')
ylabel('Accuracy')
legend('train acc','test acc')
% Saving model 
% save('modelMIMO_2x2_4QAM_DoubleOneHot_3dB.mat','W1','W2','b1','b2');

% Calculates confusion matrix and F1 score
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




