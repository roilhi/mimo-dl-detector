clear
close all
clc

M = 4;
bpsym = log2(M);
qam_idx = 0:M-1; %índices de la modulación QAM
Nt = 2; % Número de transmisores
Nr = 2; % número de receptoras
%rand_sym_idx = randi(M^Nt,1,N);
qam_sym = qammod(qam_idx,M);

models{1} = 'modelMIMO_2x2_4QAM_3dB.mat';
models{2} = 'modelMIMO2x2_4QAMNoOneHot_3dB.mat'; % carga modelo W1,W2,b1,b2
models{3} = 'modelMIMO_2x2_4QAM_DoubleOneHot_3dB.mat';

W1 = cell(1,3);
W2 = cell(1,3);
b1 = cell(1,3);
b2 = cell(1,3);

for k=1:length(models)
    mat_model = load(models{k});
    W1{k} = mat_model.W1;
    W2{k} = mat_model.W2;
    b1{k} = mat_model.b1;
    b2{k} = mat_model.b2;
end

be = 0;
be2 = be;
SNR_dB = -15:20;
SNR_l = 10.^(SNR_dB./10);
BER = zeros(size(SNR_dB));
BER2 = BER;
FN = 1/sqrt((2/3)*(M-1)); %Factor de normalización
y = FN*qam_sym;
n_iter = 1e4;

suma = 0;
for q=1:M
    pow1 = sqrt(real(y(q))^2+imag(y(q))^2);
    suma = suma+pow1;
end
pow = suma/M;
y = y/pow;
[Xx, Yy] = meshgrid(y,y);
prod_cart = [Xx(:) Yy(:)];
C = (1/sqrt(2))*prod_cart;
real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;
idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2)];

idx_sel = 1;
x = (1/sqrt(2))*[y(idx_sel),y(idx_sel)];
for j=1:length(SNR_l)
    SNR_j = SNR_l(j);   
    for k=1:n_iter
        H = sqrt(1/2)*(randn(Nr,Nt)+1i*(randn(Nr,Nt)));
        %H = eye(Nr,Nt);
        n = sqrt(1/2)*(randn(Nr,1)+1i*(randn(Nr,1)));
        n = (1/sqrt(SNR_j))*n;
        %r = H*x.';
        Hinv = pinv(H);
        H_eqz = H*Hinv;
        %r = H*x.' + n;
        r = H_eqz*x.'+n;
        %---------- Detector MLD ---------------------
        %s = abs(r-sqrt(SNR_j)*(C*H))^2;
        s1 = abs(r(1)-sqrt(SNR_j)*(C*H_eqz(:,1))).^2;
        s2 = abs(r(2)-sqrt(SNR_j)*(C*H_eqz(:,2))).^2;
        s = s1+s2;
        [~,idx] = min(s);
        idx = min(idx); % por si encuentra 2
        if idx ~=1
            a = biterr(idx_sel-1,idx-1);
            be = be+a;
        end
        real_r = real(r);
        imag_r = imag(r);
        % [real(r1) imag(r1) real(r2) imag(r2)]
        Xinput =  [real_r(1) imag_r(1) real_r(2) imag_r(2)];
        Z1 = W1*Xinput'+b1;
        A1 = max(0,Z1); % ReLU
        Z2 = W2*A1+b2;
        % Activación Sigmoide (no one-hot)
        A2 = 1./exp(-Z2);
        [~,idx_DL] = ismember((A2 > 0.5)',idx_sign,'rows');
        %A2 = exp(Z2)./sum(exp(Z2)); % para la red con one hot y softmax
        %[~,idx_DL] = max(A2); % indice máximo del vector one hot       
        if idx_DL ~=1
            a2 = biterr(idx_sel-1,idx_DL-1);
            be2 = be2+a2;
        end
    end
    BER(j) = be/(k*log2(16));
    BER2(j) = be2/(k*log2(16));
    be = 0;
    be2 = 0;
    % ----------- Detector DL --------------------
    
end

figure
semilogy(SNR_dB,BER,'s-')
hold on
semilogy(SNR_dB,BER2,'d-')
xlabel('SNR, (dB)');
ylabel('ABEP');
legend('MLD', 'DL')
grid on


