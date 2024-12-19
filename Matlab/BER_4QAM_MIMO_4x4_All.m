% ///////////////////////////////////////////////////////////////////////
%  This MATLAB script generates a BER curve for a 4x4 MIMO DL-based detector
%  Includes all labeling strategies and models. 
%  The code presented generates Figure 4 of the paper:
%  Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.;  Del-Puerto-Flores, J.A;
%  Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L. "Efficient 
%  Deep Learning-Based Detection Scheme for MIMO Communication System" 
%  Submitted to the Journal Sensors of MDPI
% 
%
% License: This code is licensed under the GPLv2 license. If you in any way
% use this code for research that results in publications, please cite our
% paper as described above.
%
%   Authors: Roilhi Frajo Ibarra Hernández (roilhi.ibarra@uaslp.mx)
%            Francisco Rubén Castillo-Soria (ruben.soria@uaslp.mx)
% ///////////////////////////////////////////////////////////////////////
clear
close all
clc

M = 4; % Modulation order
bpsym = log2(M);
qam_idx = 0:M-1; % QAM indices
Nt = 4; % Number of Tx antennas
Nr = 4; % Number of Rx antennas
%rand_sym_idx = randi(M^Nt,1,N);
qam_sym = qammod(qam_idx,M); %array of QAM symbols

% Uncomment the model you want to test
load('modelo_4x4_OneHot_50kep90.mat') % one-hot labeling model
%load('modelo_4x4_LabelEncoder_serv.mat') % direct symbol encoding 
%load('modelo_4x4_QuadOneHot_serv.mat') % one-hot per antenna model 

be = 0; % initialize bit error counter
be2 = be;
SNR_dB = 0:18; % SNR to test 
SNR_l = 10.^(SNR_dB./10);
BER = zeros(size(SNR_dB)); % initialize vector of BER
BER2 = BER;
FN = 1/sqrt((2/3)*(M-1)); %Normalization factor
y = FN*qam_sym;
n_iter = 1e5; % Number of iterations, MonteCarlo BER calculation

% normalizing power for each antenna
suma = 0;
for q=1:M
    pow1 = sqrt(real(y(q))^2+imag(y(q))^2);
    suma = suma+pow1;
end
pow = suma/M;
y = y/pow;
%[Xx, Yy] = meshgrid(y,y);

% Cartesian product for all combinations of symbols and
% transmitting antennas
[A, B, C, D] = ndgrid(y,y,y,y);
%C = (1/sqrt(2))*[Xx(:) Yy(:)];
prod_cart = [A(:) B(:) C(:) D(:)];
[aa, bb, cc, dd] = ndgrid(qam_idx+1, qam_idx+1,qam_idx+1, qam_idx+1);
prod_cart_idx = [aa(:) bb(:) cc(:) dd(:)];
C_s = (1/sqrt(2))*prod_cart;
idx_sel = 1;
x = (1/sqrt(2))*[y(idx_sel), y(idx_sel), y(idx_sel), y(idx_sel)];

% Preparing X data dividing real and imaginary parts
real_sign = real(prod_cart)<0;
imag_sign = imag(prod_cart)<0;
idx_sign = [real_sign(:,1) imag_sign(:,1) real_sign(:,2) imag_sign(:,2) ,...
            real_sign(:,3) imag_sign(:,3) real_sign(:,4) imag_sign(:,4)];

%n_part_a3 = (M*Nt)/4;

% Montecarlo BER calculation
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
        %---------- Maximum likelihood detector  ---------------------
        %s = abs(r-sqrt(SNR_j)*(C*H))^2;
        s1 = abs(r(1)-sqrt(SNR_j)*(C_s*H_eqz(:,1))).^2;
        s2 = abs(r(2)-sqrt(SNR_j)*(C_s*H_eqz(:,2))).^2;
        s3 = abs(r(3)-sqrt(SNR_j)*(C_s*H_eqz(:,3))).^2;
        s4 = abs(r(4)-sqrt(SNR_j)*(C_s*H_eqz(:,4))).^2;
        s = s1+s2+s3+s4;
        [~,idx] = min(s);
        idx = min(idx); % if it finds 2 symbols
        if idx ~=1
            a = biterr(idx_sel-1,idx-1);
            be = be+a;
        end
        real_r = real(r);
        imag_r = imag(r);
        % [real(r1) imag(r1) real(r2) imag(r2)]
        % input for the neural network
        Xinput =  [real_r(1) imag_r(1) real_r(2) imag_r(2), ... 
                   real_r(3) imag_r(3) real_r(4) imag_r(4)];
        % forward propagation for inference
        Z1 = W1*Xinput'+b1;
        A1 = max(0,Z1); % ReLU
        Z2 = W2*A1+b2;
        A2 = max(0,Z2); % ReLU 2
        Z3 = W3*A2+b3; 
        % softmax activation
        A3 = exp(Z3)./sum(exp(Z3)); % uncomment for one-hot antenna choice
        [~,idx_DL] = max(flip(A3));
        % uncomment if direct symbol encoding model is chosen
        %A3 = 1./(1+exp(-Z3)); % sigmoid activation
        % uncomment if one-hot encoding per antenna model is chosen
        %A3_rows_a1 = A3((n_part_a3)+1:2*n_part_a3,:); % split sigmoid activation
        %A3_rows_a2 = A3((n_part_a3)+1:2*n_part_a3,:);
        %A3_rows_a3 = A3((2*n_part_a3)+1:3*n_part_a3,:);
        %A3_rows_a4 = A3((3*(n_part_a3))+1:end,:);
        
        %[~, y_hat1] = max(A3_rows_a1);
        %[~, y_hat2] = max(A3_rows_a2);
        %[~, y_hat3] = max(A3_rows_a3);
        %[~, y_hat4] = max(A3_rows_a4);
           
        %[~,idx_DL] = ismember((A3 > 0.5)',idx_sign,'rows');
        %[~,idx_DL] = ismember([y_hat1' y_hat2' y_hat3' y_hat4'],prod_cart_idx,'rows');
        % calculation of error bits
        if idx_DL ~=1
            a2 = biterr(idx_sel-1,idx_DL-1);
            be2 = be2+a2;
        end
    end
    % averaging error bits
    BER(j) = be/(k*log2(256));
    BER2(j) = be2/(k*log2(256));
    be = 0;
    be2 = 0;
    % ----------- Detector DL --------------------
    
end

% generates Figure
figure
semilogy(SNR_dB,BER,'s-')
hold on
semilogy(SNR_dB,BER2,'d-')
xlabel('SNR, (dB)');
ylabel('ABEP');
legend('MLD', 'DL')
grid on
