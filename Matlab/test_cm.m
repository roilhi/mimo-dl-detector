clear
close all
clc

clases = 3;
long = 10;
ytrue = randi(clases,1,long); % ytrue

ypred = randi(clases,1,long); % ypred

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





