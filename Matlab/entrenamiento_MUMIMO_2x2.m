clear
close all
clc
% s�mbolos qam para el dataset de entrenamiento
N = 1e6; % n�mero de s�mbolos dataset de entrenamiento
M = 4; % orden de la modulaci�n
qam_idx = 0:M-1; %�ndices de la modulaci�n QAM
Nt = 2; % N�mero de transmisores
Nr = 2; % n�mero de receptoras
% valores iniciales de la red neuronal
%n_layers = 2;
n_neuronas_oculta = 100; % neuronas capa oculta
input_size = 2*Nt;
output_size = log2(M)*Nt;
n_epocas = 2000;
% S�mbolo aleatorio entre todas las combinaciones posibles
rand_sym_idx = randi(M^Nt,1,N);
%y = zeros(N,M^Nt); 
% Cambiamos el encoding para tener n-bits recibidos por antena
% y:= [ bits_a1 | bits_a2 ... | bits_aNt ]
y = zeros(N,output_size);
X = zeros(N,input_size);
qam_sym = qammod(qam_idx,M);
alpha = 0.01; % tasa de aprendizaje

% Realiza el producto cartesiano (todas las combinaciones) del alfabeto
% es decir, todas las combinaciones en Nt antenas
[Xx, Yy] = meshgrid(qam_sym,qam_sym);
prod_cart = [Xx(:) Yy(:)];
real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;

idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2)];

SNR_dB = 3; 
SNR_l = 10.^(SNR_dB./10);
No = 1;
%idx_member = zeros(1,N);
for i=1:N
    %one_hot = y(i,:);
    %one_hot(rand_sym_idx(i)) = 1; %asigna un 1 en el s�mbolo seleccionado
    %y(i,:) = one_hot;
    sel_symbol = prod_cart(rand_sym_idx(i),:); % selecciona un s�mbolo 
    y(i,:) = idx_sign(rand_sym_idx(i),:);
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


%for r=1:length(X)
%    idx_member(r) = find(ismember(idx_sign,X(r,:)<0,'rows')); %encuentra el cuadrante en la matriz one hot
    %prev_coding = y(r,:);
    %prev_coding(r,:) = X(idx_member(r)); % asigna 1 en la combinaci�n de signos que encontr�
    %y(r,:) = prev_coding;
%    y(r,:) = idx_sign(idx_member(r),:);
%end

% Normalizaci�n
X = X-mean(X(:));
X = X./std(X(:));

train_qty = round(0.9*length(X)); % obtiene la cantidad de filas correspondiente al 80%
test_qty = N-train_qty;

% 80% de las filas datos de entrenamiento
Xtrain = X(1:train_qty,:);
ytrain = y(1:train_qty,:);
%ytr_vls = ytrain';
idx_train = rand_sym_idx(1:train_qty);
%[~,idx_train] = max(ytr_vls); 
% 20% de las filas restantes, datos de validaci�n
Xtest = X(train_qty+1:end,:);
ytest = y(train_qty+1:end,:);
%ytest_vls = ytest';
%[~,idx_test] = max(ytest_vls);
idx_test = rand_sym_idx(train_qty+1:end);

% Dataloaders y divisi�n en batches
batch_size = N; % tama�o batch
n_batches_train = ceil(train_qty/batch_size);
n_batches_test = ceil(test_qty/batch_size);

Xtrain_dataloader = cell(1,n_batches_train);
ytrain_dataloader = cell(1,n_batches_train);
Xtest_dataloader = cell(1,n_batches_test);
ytest_dataloader = cell(1,n_batches_test);
idx_train_dataloader = cell(1,n_batches_train);
idx_test_dataloader = cell(1,n_batches_test);
lim_inf = 1;
for r = 1:n_batches_train
    if r*batch_size <= length(Xtrain)
        Xtrain_dataloader{r} = Xtrain(lim_inf:r*batch_size,:);
        ytrain_dataloader{r} = ytrain(lim_inf:r*batch_size,:);
        idx_train_dataloader{r} = idx_train(lim_inf:r*batch_size);
    else
        Xtrain_dataloader{r} = Xtrain(lim_inf:end,:);
        ytrain_dataloader{r} = ytrain(lim_inf:end,:);
        idx_train_dataloader{r} = idx_train(lim_inf:end);
    end
    lim_inf = lim_inf + batch_size;
end

lim_inf = 1;
for r = 1:n_batches_test
    if r*batch_size <= length(Xtest)
        Xtest_dataloader{r} = Xtest(lim_inf:r*batch_size,:);
        ytest_dataloader{r} = ytest(lim_inf:r*batch_size,:);
        idx_test_dataloader{r} = idx_test(lim_inf:r*batch_size);
    else
        Xtest_dataloader{r} = Xtest(lim_inf:end,:);
        ytest_dataloader{r} = ytest(lim_inf:end,:);
        idx_test_dataloader{r} = idx_test(lim_inf:end);
    end
    lim_inf = lim_inf + batch_size;
end
%idx_train_dataloader = reshape(idx_train,n_batches_train,batch_size);
%idx_test_dataloader = reshape(idx_test,n_batches_test,batch_size);
%[a,b] = size(Xtrain);
% Arquitectura:
% input (4 neuronas) --- oculta (10 neuronas) --- salida (16 s�mbolos) 
% n[0] = 4 % n�mero de features
% n[1] = 10 % n�mero de neuronas capa oculta
% n[2] = 16 % n�mero de salidas (solo tenemos 3 capas)
% pesos y bias de la capa 1 
% size(W^[l]) = (n[l] , n[l-1])
% Inicializaci�n Xavier
xavier_limit = sqrt(6)/sqrt(input_size+output_size);
W1 = rand(n_neuronas_oculta,input_size);  %(10,4)
W1 = -xavier_limit+2*xavier_limit*W1;
% size(b^[l] = (n[l],1)
b1 = randn(n_neuronas_oculta,1); %(10,1)
% pesos y bias de la capa 2
W2 = rand(output_size,n_neuronas_oculta); % (16,10)
W2 = -xavier_limit+2*xavier_limit*W2;
b2 = randn(output_size,1); 

train_loss = zeros(1,n_epocas);
test_loss = zeros(1,n_epocas);
train_acc = zeros(1,n_epocas);
test_acc = zeros(1,n_epocas);

for i=1:n_epocas
    train_loss_batch = 0;
    train_acc_batch = 0;
    for k=1:n_batches_train
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
        Z1 = W1*Xtrain_dataloader{k}';
        [~,cols_Z1] = size(Z1);
        b1 = repmat(b1,1,cols_Z1);
        Z1 = Z1+b1;
        % funci�n de activaci�n ReLU despu�s de la salida capa 1
        A1 = max(0,Z1); % (10,m)
        %A1 = Z1; % linear activation
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
        %expA2 = exp(Z2);
        %[rowsA2, colsA2] = size(expA2);
        % Funci�n softmax
        % sum_expZ2 = repmat(sum(exp(Z2)),rowsA2,1);
        % A2 = exp(Z2)./sum_expZ2;
        % argmax(A2), selecciona el �ndice m�ximo del vector one hot
        %[~, y_hat] = max(A2);
        % Activaci�n sigmoide
        A2 = 1./(1+exp(-Z2));
        % Activacion lineal
        %A2 = Z2;   
        % Identifica el �ndice de la combinaci�n correspondiente de s�mbolos 
        %y_hat = (A2>0.5)';
        [~, y_hat] = ismember((A2>0.5)',idx_sign,'rows');
        y_hat = y_hat';
        %bits_mismatch = biterr(ytrain, y_hat,[],'row-wise');
        % funci�n de p�rdidas o loss
        train_loss_batch = train_loss_batch + sqrt((1/train_qty)*sum((y_hat-idx_train_dataloader{k}).^2));
        %train_loss(i) = (1/(train_qty*output_size))*sum(bits_mismatch);
        % accuracy (predicciones correctas / total)
        %train_acc(i) = (1/(train_qty*output_size))*sum(output_size-bits_mismatch);
        f = y_hat==idx_train_dataloader{k};
        train_acc_batch = train_acc_batch + (length(find(f==1))/train_qty);
        % **************************
        % BACK PROPAGATION
        % **************************
        % dZ[l] = dA[l]*g[l]'(Z[l])
        dZ2 = A2 - ytrain_dataloader{k}'; % (16,m)
        % dW[l] = 1/m dZ[l] * A[l-1].T
        dW2 = (1/train_qty)*(dZ2*A1'); % (16,m)
        % derivada promedio bias db2
        db2 = (1/train_qty)*(sum(dZ2,2));
        % (10,16)(16,m) * (10*m)
        dZ1_prev = (W2'*dZ2); % (10,m)
        % la derivada de ReLU es f(x) = x>0 esa es g'(x) 
        % esta funci�n regresa 1 para x>0 y 0 en otro caso
        dZ1 = dZ1_prev.*(Z1>0); %(10,m)
        %dZ1 = dZ1_prev; % Derivada lineal es igual a 1
        dW1 = (1/train_qty)*(dZ1*Xtrain_dataloader{k}); %(10,4)
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
    end
    train_loss(i) = train_loss_batch/n_batches_train;
    train_acc(i) = train_acc_batch/n_batches_train;
    test_acc_batch = 0;
    test_loss_batch = 0;
    for k=1:n_batches_test
        % ****************************
        % Validaci�n
        % ****************************
        % Con broadcasting para Z1 validaci�n
        % Z1_V = (W1*Xtest')+b1; % FC capa 1
        % Sin broadcasting para la validaci�n
        Z1_V = W1*Xtest_dataloader{k}';
        [~,cols_Z1V] = size(Z1_V);
        b1 = repmat(b1,1,cols_Z1V);
        Z1_V = Z1_V + b1;
        A1_V = max(0,Z1_V); % ReLU(Z1)
        % A1_V = Z1_V; % Activacion lineal
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
        %expA2V = exp(Z2_V);
        %sum_expA2V = repmat(sum(expA2V),size(expA2V,1),1);
        %A2_V = expA2V./sum_expA2V;
        %[~,y_hat_v] = max(A2_V); % argmax(A2) o �nidice m�xima probabilidad
        %A2_V = Z2_V; % activaci�n lineal
        A2_V = 1./exp(-Z2_V); %activaci�n sigmoide
        %y_hat_v = (A2_V>0.5)';
        %bits_mismatch_v = biterr(ytest,y_hat_v,[],'row-wise');
        %test_loss(i) = (1/(test_qty*output_size))*(sum(bits_mismatch_v));
        %test_acc(i) = (1/(test_qty*output_size))*sum(output_size-bits_mismatch_v);
        [~,y_hat_v] = ismember((A2_V>0.5)',idx_sign,'rows');
        y_hat_v = y_hat_v';
        test_loss_batch = test_loss_batch + ((1/test_qty)*sum((y_hat_v-idx_test_dataloader{k}).^2)); 
        f_test = y_hat_v == idx_test_dataloader{k};
        test_acc_batch = test_acc_batch + (length(find(f_test==1))/test_qty);
        % Regresar tama�os reales b1,b2 por el broadcasting
        b1 = b1(:,1);
        b2 = b2(:,1);
    end
    test_loss(i) = test_loss_batch/n_batches_test;
    test_acc(i) = test_acc_batch/n_batches_train;
    if rem(i,100)==0
        fprintf('******************************** \n');
        txt= '�poca %d | Train loss %2.2f | Test loss %2.2f | Train acc %2.2f | Test acc %2.2f\n';
        fprintf(txt,i,train_loss(i),test_loss(i), train_acc(i), test_acc(i));
    end
end

figure
title('Curvas de p�rdidas')
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

save('modelMIMO2x2_4QAMNoOneHot_3dB.mat','W1','W2','b1','b2');