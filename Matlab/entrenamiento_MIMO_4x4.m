clear
close all
clc
% símbolos qam para el dataset de entrenamiento
N = 1e4; % número de símbolos dataset de entrenamiento
M = 4; % orden de la modulación
qam_idx = 0:M-1; %índices de la modulación QAM
Nt = 4; % Número de transmisores
Nr = 4; % número de receptoras
n_neuronas_oculta = 100; %10 neuronas capa oculta
input_size = 2*Nt;
output_size = log2(M)*Nt;
n_epocas = 1000;

batch_size = N;
rand_sym_idx = randi(M^Nt,1,N);

y = zeros(N,M^Nt); 
X = zeros(N,2*Nt);
qam_sym = qammod(qam_idx,M);
alpha = 0.01; % tasa de aprendizaje

% Realiza el producto cartesiano (todas las combinaciones) del alfabeto
% es decir, todas las combinaciones en Nt antenas

[A, B, C, D] = ndgrid(qam_sym,qam_sym,qam_sym,qam_sym);
prod_cart = [A(:) B(:) C(:) D(:)];
real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;

idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2), ...
            real_sign(:,3) imag_sign(:,3) real_sign(:,4) imag_sign(:,4)];

SNR_dB = 3; 
SNR_l = 10.^(SNR_dB./10);
No = 1;
%idx_member = zeros(1,N);
for i=1:N
    %one_hot = y(i,:);
    %one_hot(rand_sym_idx(i)) = 1; %asigna un 1 en el símbolo seleccionado
    %y(i,:) = one_hot;
    sel_symbol = prod_cart(rand_sym_idx(i),:); % selecciona un símbolo 
    y(i,:) = idx_sign(rand_sym_idx(i),:); % almacena en la etiqueta y el dato
    H = (1/sqrt(2))*(randn(Nr,Nt) + 1i*randn(Nr,Nt));
    n = (No/sqrt(2))*(randn(Nr,1) + 1i*randn(Nr,1));
    n = (1/sqrt(SNR_l))*n;
    r_x = H*sel_symbol.';
    H_inv = pinv(H);
    r_x = H_inv*r_x+n;
    X(i,:) = [real(r_x.') imag(r_x.')];
end
orden = [1,3,2,4,5,7,6,8]; % [real(x1) imag(x1) real(x2) imag(x2)]
X = X(:,orden); % [real(r1) imag(r1) real(r2) imag(r2)]


% for r=1:length(X)
%     idx_member = find(ismember(idx_sign,X(r,:)<0,'rows')); %encuentra el cuadrante en la matriz one hot
%     one_hot = y(r,:);
%     one_hot(idx_member) = 1; % asigna 1 en la combinación de signos que encontró
%     y(r,:) = one_hot;
% end

% Normalización
X = X-mean(X(:));
X = X./std(X(:));

train_qty = round(0.8*length(X)); % obtiene la cantidad de filas correspondiente al 80%
test_qty = N-train_qty;

% 80% de las filas datos de entrenamiento
Xtrain = X(1:train_qty,:);
ytrain = y(1:train_qty,:);
ytr_vls = ytrain';
[~,idx_train] = max(ytr_vls); 
% 20% de las filas restantes, datos de validación
Xtest = X(train_qty+1:end,:);
ytest = y(train_qty+1:end,:);
ytest_vls = ytest';
[~,idx_test] = max(ytest_vls);

% valores iniciales de la red neuronal
%n_layers = 2;

n_batches_train = ceil(length(Xtrain)/batch_size);
n_batches_test = ceil(length(Xtest)/batch_size);
Xtrain_dataloader = cell(1,n_batches_train);
ytrain_dataloader = cell(1,n_batches_train);
idx_train_dataloader = reshape(idx_train,n_batches_train,batch_size);
lim_inf = 1;
for r=1:n_batches_train;
    Xtrain_dataloader{r} = Xtrain(lim_inf:r*batch_size,:);
    ytrain_dataloader{r} = ytrain(lim_inf:r*batch_size,:);
    lim_inf = lim_inf+batch_size;
end
Xtest_dataloader = cell(1,n_batches_test);
ytest_dataloader = cell(1,n_batches_test);
idx_test_dataloader = reshape(idx_test,n_batches_test,batch_size);
lim_inf = 1;
for r=1:n_batches_test;
    Xtest_dataloader{r} = Xtest(lim_inf:r*batch_size,:);
    ytest_dataloader{r} = ytest(lim_inf:r*batch_size,:);
    lim_inf = lim_inf+batch_size;
end

% Arquitectura:
% input (4 neuronas) --- oculta (10 neuronas) --- salida (16 símbolos) 
% n[0] = 4 % número de features
% n[1] = 10 % número de neuronas capa oculta
% n[2] = 16 % número de salidas (solo tenemos 3 capas)
% pesos y bias de la capa 1 
% size(W^[l]) = (n[l] , n[l-1])
W1 = rand(n_neuronas_oculta,input_size);  %(10,4)
% size(b^[l] = (n[l],1)
b1 = zeros(n_neuronas_oculta,1); %(100,1)
% pesos y bias de la capa 2
W2 = rand(output_size,n_neuronas_oculta); % (16,10)
b2 = zeros(output_size,1); %(16,1)

train_loss = zeros(1,n_epocas);
test_loss = zeros(1,n_epocas);
train_acc = zeros(1,n_epocas);
test_acc = zeros(1,n_epocas);

for i=1:n_epocas
% Inicializando las métricas por batch
    train_loss_batch = 0;
    train_acc_batch = 0;
    for k=1:n_batches_train
        % **************************
        % FORWARD PROPAGATION
        % **************************
        % Forward propagation
        % size(Z[l]) = (n[l],m)
        % Si es MATLAB 2020 en adelante es posible el broadcasting,
        % descomentar esta opción
        % Z1 = (W1*Xtrain')+b1;
        % si el MATLAB < 2020 no es posible el broadcasting, descomentar la
        % línea
        Z1 = W1*Xtrain_dataloader{k}';
        [~,cols_Z1] = size(Z1);
        b1 = repmat(b1,1,cols_Z1);
        Z1 = Z1+b1;
        % función de activación ReLU después de la salida capa 1
        A1 = max(0,Z1); % (10,m)
        % Entrada a la capa 2 (multiplica por los pesos)
        % (16,10)(10,m) + (16,1)
        % Con broadcasting, activar esta opcion si es MATLAB > 2022
        %Z2 = W2*A1+b2; % (16,m)
        % Sin broadcasting activar esta opción si MATLAB < 2022
        Z2 = W2*A1;
        [~,cols_Z2] = size(Z2);
        b2 = repmat(b2,1,cols_Z2);
        Z2 = Z2+b2;
        % función softmax después de la salida capa 2
        %A2 = exp(Z2)./sum(exp(Z2)); % (16,m)
        % si MATLAB < 2022, sin broadcasting
        expA2 = exp(Z2);
        [rowsA2, colsA2] = size(expA2);
        sum_expZ2 = repmat(sum(exp(Z2)),rowsA2,1);
        A2 = exp(Z2)./sum_expZ2;
        % argmax(A2), selecciona el índice máximo del vector one hot
        [~, y_hat] = max(A2);
        % función de pérdidas o loss
        train_loss_batch = train_loss_batch + sqrt((1/train_qty)*sum((y_hat-idx_train_dataloader(k,:)).^2));
        % accuracy (predicciones correctas / total)
        f = y_hat==idx_train_dataloader(k,:);
        train_acc_batch = train_acc_batch + length(find(f==1))/train_qty;
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
        % esta función regresa 1 para x>0 y 0 en otro caso
        dZ1 = dZ1_prev.*(Z1>0); %(10,m)
        dW1 = (1/train_qty)*(dZ1*Xtrain_dataloader{k}); %(10,4)
        db1 = (1/train_qty)*sum(dZ1,2); % (10,1)
        % Si no se aplica broadcasting (MATLAB > 2020), regresar b1,b2
        % a su tamaño real
        b1 = b1(:,1);
        b2 = b2(:,1);
        % Actualiza los pesos
        W1 = W1 - alpha*dW1;
        b1 = b1 - alpha*db1;
        W2 = W2 - alpha*dW2;
        b2 = b2 - alpha*db2;
        % txt1= 'Entrenando %d/%d muestras \n';
        % fprintf(txt1,k*size(Xtrain_dataloader,3),length(Xtrain));
    end
    train_loss(i) = train_loss_batch/n_batches_train;
    train_acc(i) = train_acc_batch/n_batches_train;
    test_acc_batch = 0;
    test_loss_batch = 0;
    for k=1:n_batches_test
        % ****************************
        % Validación
        % ****************************
        % Con broadcasting para Z1 validación
        %Z1_V = (W1*Xtest')+b1; % FC capa 1
        % Sin broadcasting para la validación
        Z1_V = W1*Xtest_dataloader{k}';
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
        [~,y_hat_v] = max(A2_V); % argmax(A2) o ínidice máxima probabilidad
        test_loss_batch = test_loss_batch + sqrt((1/test_qty)*sum((y_hat_v-idx_test_dataloader(k,:)).^2)); 
        f_train = y_hat_v == idx_test_dataloader(k,:);
        test_acc_batch = test_acc_batch + length(find(f_train==1))/test_qty;
        b1 = b1(:,1);
        b2 = b2(:,1);
    end
    test_loss(i) = test_loss_batch/n_batches_test;
    test_acc(i) = test_acc_batch/n_batches_test;

    if rem(i,100)==0
        fprintf('**********ÉPOCA %d********************** \n',i);
        txt= ' Train loss %2.2f | Test loss %2.2f | Train acc %2.2f | Test acc %2.2f\n';
        fprintf(txt, train_loss(i),test_loss(i), train_acc(i), test_acc(i));
    end
    % Regresar tamaños reales b1,b2 por el broadcasting
    b1 = b1(:,1);
    b2 = b2(:,1);
end

figure
title('Curvas de pérdidas')
plot(train_loss,'LineWidth',2), grid on, hold on,
plot(test_loss,'--r','LineWidth',2),
xlabel('Épocas')
ylabel('Loss')
legend('train loss','test loss')

figure
title('Curvas de accuracy')
plot(train_acc, 'LineWidth',2), grid on, hold on,
plot(test_acc,'--r','LineWidth',2),
xlabel('Épocas')
ylabel('Accuracy')
legend('train acc','test acc')

save('modelMIMO_4x4_4QAM_3dB.mat','W1','W2','b1','b2');