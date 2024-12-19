% ///////////////////////////////////////////////////////////////////////
%  This MATLAB script generates Figure 5 of the paper:
%
%  Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.;  Del-Puerto-Flores, J.A;
%  Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L. "Efficient 
%  Deep Learning-Based Detection Scheme for MIMO Communication System" 
%  Submitted to the Journal Sensors of MDPI
%  related to the detectors complexity
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

Nt1 = 2;
Nr1 = 2;

Nt2= 4;
Nr2 = 4;

M = 4; 

N_units1 = 100;
N_units2 = 1000;

Nflops_ML = zeros(1,2);
Nflops_OH = zeros(1,2);
Nflops_OHA = zeros(1,2);
Nflops_SE = zeros(1,2);

Nflops_ML(1) = Nt1*Nr1*M^Nt1;
Nflops_ML(2) = Nt2*Nr2*M^Nt2;


Nflops_OH(1) = 2*(Nr1+N_units1)+M^Nt1;
Nflops_OH(2) = 2*(Nr2+N_units2)+M^Nt2;

Nflops_OHA(1) = 2*(Nr1+N_units1)+M*Nt1;
Nflops_OHA(2) = 2*(Nr2+N_units2)+M*Nt2;

Nflops_SE(1) = 2*(Nr1+N_units1)+log2(M)*Nt1;
Nflops_SE(2) = 2*(Nr2+N_units2)+log2(M)*Nt2;

bar_values = [Nflops_ML; Nflops_OH; Nflops_OHA; Nflops_SE];

figure
x=1:4;
string_vals = {'ML','OH','OHA','SE'};
colormap cool
bar(bar_values), grid
legend('2x2 MIMO','4x4 MIMO')
xticks(x)
xticklabels(string_vals)
xlabel('labeling')
ylabel('flops')

