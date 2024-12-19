% ///////////////////////////////////////////////////////////////////////
%  This MATLAB script test the confusion matrix and F1-score calculation
%  The code is a supplementary material for the paper: 
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
% example vector of classes
clases = 3;
long = 10;
% generates random vectors of true and predicted values
ytrue = randi(clases,1,long); % ytrue or actual labels

ypred = randi(clases,1,long); % ypred or predicted labels

%initialize confusion matrix
cm = zeros(clases);

% creating confusion matrix if finds equal values
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

% Calculates precision, recall and F1
F1 = 0;
for ii=1:clases;
    if (cm(ii,ii)==0)
        Prec = 0;
        Recall = 0;
        F1 = F1+0;
    else
        Prec = cm(ii,ii)/sum((cm(ii,:)));
        Recall = cm(ii,ii)/sum((cm(:,ii)));
        F1 = F1 + (2*Prec*Recall)/(Prec+Recall);
    end
end

F1_macro = F1/clases;





