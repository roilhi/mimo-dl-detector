# Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems
This code scripts are related to the following scientific article:

Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.;  Del-Puerto-Flores, J.A; Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L. ``Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems``. Submitted to _Sensors_

## Abstract

Multiple input-multiple output (MIMO) is a key enabling technology for the next generation of wireless communication systems. A flexible MIMO scheme design could impact the performance of systems with hardware limitations. This paper presents a deep learning (DL)-based signal detection strategy for MIMO communication systems. More specifically, a preprocessing stage is added to label the input signals conveniently, improving the overall system performance. Based on this strategy, two novel schemes are proposed and evaluated considering the bit error rate (BER) and detection complexity. The performance of the proposed schemes is compared with the conventional one-hot scheme and the optimal maximum likelihood (ML) criterion. Results show that the proposed schemes achieve near-optimal BER performance offering different complexity tradeoffs while reaching a classification performance F1-score of 0.97. The proposed strategy may be useful in adaptive systems with limited computational resources.

# Contents of the repository
* Files starting with _entrenamiento_ or _training_ names are for generating a DL model (saved as .mat) and the training stage, the loss and Acc curves respectively
* Files starting with BER correspond to the calculation of __Bit Error Rate__ curves where you require to load the .mat DL detection models
* We tested both 2x2 and 4x4 MIMO configurations and 4-QAM modulation
* We developed 3 labeling schemes mentioned in the paper:
    * One-hot encoding for each one of the $$M^{N_t}$$ transmitted symbol combinations
    * One-hot encoding per transmitting antenna, having a size of $$M*N_t$$ choices
    * Direct encoding of QAM symbols, having $$\log_2(M)*N_t$$ choices
# Citation and referencing
Ibarra-Hernández, R. F., Castillo-Soria, F. R., Gutiérrez, C. A., Del-Puerto-Flores, J. A., Acosta-Elias, J., Rodriguez-Abdala, V. I., & Palacios-Luengas, L. (2025). __Efficient Deep Learning-Based Detection Scheme for MIMO Communication Systems__. _Sensors_, 25(3), 669. [https://doi.org/10.3390/s25030669]
