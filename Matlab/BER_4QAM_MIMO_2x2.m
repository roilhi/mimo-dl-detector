clear
close all
clc

M = 4;
bpsym = log2(M);
N = 1e4; % número de símbolos
qam_idx = 0:M-1; %índices de la modulación QAM
Nt = 2; % Número de transmisores
Nr = 2; % número de receptoras
%rand_sym_idx = randi(M^Nt,1,N);
qam_sym = qammod(qam_idx,M);

be = 0;

SNR_dB = -5:15;
SNR_l = 10.^(SNR_dB./10);
BER = zeros(size(SNR_dB));
BER2 = BER;
FN = 1/sqrt((2/3)*(M-1)); %Factor de normalización
y = FN*qam_sym;
n_iter = 1e3; % iteraciones MonteCarlo

suma = 0;
for q=1:M
    pow1 = sqrt(real(y(q))^2+imag(y(q))^2);
    suma = suma+pow1;
end
pow = suma/M;
y = y/pow;
% Hace las combinaciones de símbolos en las 2 antenas
[Xx, Yy] = meshgrid(y,y);
C = (1/sqrt(2))*[Xx(:) Yy(:)];
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
    end
    BER(j) = be/(k*log2(4^Nt));
    be = 0;
    be2 = 0;
    % ----------- Detector DL --------------------
    
end

figure
semilogy(SNR_dB,BER,'s-')
xlabel('SNR, (dB)');
ylabel('ABEP');
grid on
