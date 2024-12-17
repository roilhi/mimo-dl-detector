clear
close all
clc

M = 4;
bpsym = log2(M);
N = 1e4; % número de símbolos
qam_idx = 0:M-1; %índices de la modulación QAM
Nt = 4; % Número de transmisores
Nr = 4; % número de receptoras
%rand_sym_idx = randi(M^Nt,1,N);
qam_sym = qammod(qam_idx,M);

load('modelo_4x4_OneHot_50kep90.mat') % carga modelo W1,W2,b1,b2

be = 0;
be2 = be;
SNR_dB = 0:18;
SNR_l = 10.^(SNR_dB./10);
BER = zeros(size(SNR_dB));
BER2 = BER;
FN = 1/sqrt((2/3)*(M-1)); %Factor de normalización
y = FN*qam_sym;
n_iter = 1e5;

suma = 0;
for q=1:M
    pow1 = sqrt(real(y(q))^2+imag(y(q))^2);
    suma = suma+pow1;
end
pow = suma/M;
y = y/pow;
%[Xx, Yy] = meshgrid(y,y);
[A, B, C, D] = ndgrid(y,y,y,y);
%C = (1/sqrt(2))*[Xx(:) Yy(:)];
prod_cart = [A(:) B(:) C(:) D(:)];
[aa, bb, cc, dd] = ndgrid(qam_idx+1, qam_idx+1,qam_idx+1, qam_idx+1);
prod_cart_idx = [aa(:) bb(:) cc(:) dd(:)];
C_s = (1/sqrt(2))*prod_cart;
idx_sel = 1;
x = (1/sqrt(2))*[y(idx_sel), y(idx_sel), y(idx_sel), y(idx_sel)];

real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;
idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2) ,...
            real_sign(:,3) imag_sign(:,3) real_sign(:,4) imag_sign(:,4)];

%n_part_a3 = (M*Nt)/4;

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
        s1 = abs(r(1)-sqrt(SNR_j)*(C_s*H_eqz(:,1))).^2;
        s2 = abs(r(2)-sqrt(SNR_j)*(C_s*H_eqz(:,2))).^2;
        s3 = abs(r(3)-sqrt(SNR_j)*(C_s*H_eqz(:,3))).^2;
        s4 = abs(r(4)-sqrt(SNR_j)*(C_s*H_eqz(:,4))).^2;
        s = s1+s2+s3+s4;
        [~,idx] = min(s);
        idx = min(idx); % por si encuentra 2
        if idx ~=1
            a = biterr(idx_sel-1,idx-1);
            be = be+a;
        end
        real_r = real(r);
        imag_r = imag(r);
        % [real(r1) imag(r1) real(r2) imag(r2)]
        Xinput =  [real_r(1) imag_r(1) real_r(2) imag_r(2), ... 
                   real_r(3) imag_r(3) real_r(4) imag_r(4)];
        Z1 = W1*Xinput'+b1;
        A1 = max(0,Z1); % ReLU
        Z2 = W2*A1+b2;
        A2 = max(0,Z2); % ReLU 2
        Z3 = W3*A2+b3; 
        A3 = exp(Z3)./sum(exp(Z3));
        [~,idx_DL] = max(flip(A3));
        %A3 = 1./(1+exp(-Z3));
        
        %A3_rows_a1 = A3((n_part_a3)+1:2*n_part_a3,:);
        %A3_rows_a2 = A3((n_part_a3)+1:2*n_part_a3,:);
        %A3_rows_a3 = A3((2*n_part_a3)+1:3*n_part_a3,:);
        %A3_rows_a4 = A3((3*(n_part_a3))+1:end,:);
        
        %[~, y_hat1] = max(A3_rows_a1);
        %[~, y_hat2] = max(A3_rows_a2);
        %[~, y_hat3] = max(A3_rows_a3);
        %[~, y_hat4] = max(A3_rows_a4);
           
        %[~,idx_DL] = ismember((A3 > 0.5)',idx_sign,'rows');
        %[~,idx_DL] = ismember([y_hat1' y_hat2' y_hat3' y_hat4'],prod_cart_idx,'rows');
        if idx_DL ~=1
            a2 = biterr(idx_sel-1,idx_DL-1);
            be2 = be2+a2;
        end
    end
    BER(j) = be/(k*log2(256));
    BER2(j) = be2/(k*log2(256));
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
