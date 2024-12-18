# ///////////////////////////////////////////////////////////////////////
# This Julia script genereates a DL model using a MIMO 4x4 configuration
# and the direct symbol encoding strategy. It is used for the paper:
#
#  Ibarra-Hernández, R.F.; Castillo-Soria, F.R.; Gutiérrez, C.A.;  Del-Puerto-Flores, J.A;
#  Acosta-Elías J., Rodríguez-Abdalá V. and Palacios-Luengas L. "Efficient 
#  Deep Learning-Based Detection Scheme for MIMO Communication System" 
#  Submitted to the Journal Sensors of MDPI
# 
#
# License: This code is licensed under the GPLv2 license. If you in any way
# use this code for research that results in publications, please cite our
# paper as described above.
#
#   Authors: Roilhi Frajo Ibarra Hernández (roilhi.ibarra@uaslp.mx)
#            Francisco Rubén Castillo-Soria (ruben.soria@uaslp.mx)
# ///////////////////////////////////////////////////////////////////////

# Importing libraries
using Plots
using .Iterators
using LinearAlgebra
using Statistics
using Random 
using Base.Broadcast
using MAT

# Function to generate a QAM constellation or 
# modulator which is actually an array of M-QAM
# complex symbols
function QamModulator(M)
    N = log2(M);
    if N != round(N)
        error("M must be a multiple of 2^n")
    end

    m = 0:M-1
    c = sqrt(M)
    b = -2*(m.%c) .+ c .-1
    a = 2*floor.(m./c) .- c .+1
    s = complex.(a,b)
    return s
end

# Function that returns the index that matchs between
# two matrices according to a threshold
function MatchingRow(mat1,mat2, tresh)
    vec_idx = Int32[]
    for row in eachrow(mat1)
        vect_pr = (row.>tresh)'
        match_row_idx = Int32[]
        global mat_match = in.(vect_pr,mat2)
        for (nn, rowi) in enumerate(eachrow(mat_match))
            if all(rowi.==1)
                push!(match_row_idx,nn)
            end
        end
        push!(vec_idx, match_row_idx[1])
    end
    return vec_idx
end


# Initial parameters
N = Int(1e4)
#N = 10
M = 4 # modulation order
qam_sym = QamModulator(M) # Qam symbols array
Nr = 4 # number of Tx antennas
Nt = 4 # number of Rx antennas
input_size = 2*Nr # input size of the neural network
output_size = Int(log2(M)*Nt) # output size or label vector size
# number of units for the hidden layer
n_neuronas_oculta = 1000
# number of epochs
n_epocas = Int(5e3)
α = 0.01 # learning rate
rand_sym_idx = rand(1:M^Nt,1,N) # random indices for symbols
# initializing data and labels arrays
y = zeros(N,output_size)
X = zeros(N,input_size)

# Cartesian product which represents all possible combinations
# of QAM symbols transmitted by the Nt transmission antennas
prod_cart = collect(product(qam_sym, qam_sym, qam_sym, qam_sym))


SNR_dB = 3 # signal-to-noise ratio for generate training data
SNR_l = 10^(SNR_dB/10) # linear SNR

No = 1

idx_sign = zeros(M^Nt,2*Nt) # index array for the symbol combinations

# filling the vector with the corresponding combination index
for q=1:M^Nt
    idx_vec = Float64[]
    sel_cart = collect(prod_cart[q])
    for r in sel_cart
        push!(idx_vec, real(r)<0)
        push!(idx_vec, imag(r)<0)
    end
    idx_sign[q,:] = idx_vec'
end
idx_sign = iszero.(idx_sign)

# Generating training data and the corresponding labels
# each label is directly related with each QAM symbol from 0 to log2(M)*Nt
for i=1:N
    sel_symbol = collect(prod_cart[rand_sym_idx[i]])
    H = (1/sqrt(2))*randn(Complex{Float32},Nr,Nt)
    η = (No/sqrt(2))*randn(Complex{Float32},Nr,1)
    η *= SNR_l
    global r_x = H*sel_symbol
    H_inv = pinv(H)
    global r_x = H_inv*r_x + η
    p = Float64[]
    for z in r_x
        push!(p, real(z))
        push!(p, imag(z))
    end
    X[i,:] = p
    y[i,:] = idx_sign[rand_sym_idx[i],:]
end

# Normalizing data (normal distribution)
X .-= mean(X)
X ./= std(X)

# Train-test split of data
train_qty = Int(round(0.8*size(X,1)))
test_qty = N-train_qty

Xtrain = X[1:train_qty,:]
Xtest = X[train_qty+1:end, :]

ytrain = y[1:train_qty,:]
ytest = y[train_qty+1:end, :]
#idx_train = [val[2][1] for val in argmax(ytrain, dims=2)]
#idx_test = [val[2][1] for val in argmax(ytest, dims=2)]

idx_train = rand_sym_idx[1:train_qty]
idx_test = rand_sym_idx[train_qty+1:end]

# Xavier initialization (avoid vanishing gradients)
σ = sqrt(6)/sqrt(input_size+output_size)

# Initialize weights and biases of neurons by Xavier
W1 = -σ.+2*σ.*rand(n_neuronas_oculta, input_size)
W2 = -σ.+2*σ.*rand(n_neuronas_oculta, n_neuronas_oculta)
W3 = -σ.+2*σ.*rand(output_size, n_neuronas_oculta)

b1 = zeros(n_neuronas_oculta, 1)
b2 = zeros(n_neuronas_oculta, 1)
b3 = zeros(output_size,1)


# Initializing values for loss and acc curves

train_loss = zeros(1,n_epocas)
train_acc = zeros(1,n_epocas)
test_loss = zeros(1,n_epocas)
test_acc = zeros(1,n_epocas)


# Training loop
for i=1:n_epocas
    # Forward propagation
    global Z1 = W1*Xtrain'.+b1
    global A1 = max.(0,Z1) # ReLU
    global Z2 = W2*A1.+b2
    global A2 = max.(0,Z2) # ReLU
    global Z3 = W3*A2.+b3
    # Sigmoid Layer
    global A3 = 1 ./(1 .+exp.(-Z3))
    # give an output from the matching row of sigmoid layer
    global yhat = MatchingRow(A3',idx_sign,0.5) 
    #global yhat = [val[1][1] for val in argmax(A3, dims=1)]
    global TP = [yhat[n] == idx_train[n] ? 1 : 0 for n in eachindex(yhat)]
    train_loss[i] = sqrt((1/train_qty)*sum((yhat-idx_train).^2))
    train_acc[i] = length(findall(x->x==1,TP))/train_qty
    # Backpropagation
    # Calculating gradients by SGD
    global dZ3 = A3 - ytrain'
    global dW3 = (1/train_qty)*(dZ3*A2')
    global db3 = (1/train_qty)*(sum(dZ3,dims=2))
    global dZ2_p = W3'*dZ3
    global dZ2 = dZ2_p.*(Z2.>0)
    global dW2 = (1/train_qty)*(dZ2*A1')
    global db2 = (1/train_qty)*(sum(dZ2,dims=2))
    global dZ1_p = W2'*dZ2
    global dZ1 = dZ1_p.*(Z1.>0)
    global dW1 = (1/train_qty)*(dZ1*Xtrain)
    global db1 = (1/train_qty)*(sum(dZ1, dims=2))
    # Weights and biases update
    global W1 = W1 - α.*dW1
    global W2 = W2 - α.*dW2
    global W3 = W3 - α.*dW3
    global b1 = b1 - α.*db1
    global b2 = b2 - α.*db2
    global b3 = b3 - α.*db3
    # Inference mode for the validation set
    global Z1_V = W1*Xtest'.+b1
    global A1_V = max.(0,Z1_V)
    global Z2_V = W2*A1_V.+b2
    global A2_V = max.(0,Z2_V)
    global Z3_V = W3*A2_V.+b3
    # Sigmoid layer
    global A3_V = 1 ./(1 .+exp.(-Z3_V))
    global yhat_V = MatchingRow(A3_V',idx_sign,0.5)
    #global yhat_V = [val[1][1] for val in argmax(A3_V, dims=1)]
    global TP_V = [yhat_V[n] == idx_test[n] ? 1 : 0 for n in eachindex(yhat_V)]
    test_loss[i] = sqrt((1/test_qty)*sum((yhat_V-idx_test).^2))
    test_acc[i] = length(findall(x->x==1,TP_V))/test_qty
    # printing out what is happening (in series of 100 epochs)
    if rem(i,100)==0
        println("*********************************\n")
        println("Época ",i, "| Train Loss ",train_loss[i],"| Train ACC ",train_acc[i],"| Test loss ",test_loss[i], "| Test Acc ", test_acc[i],"\n")
    end
end

# Generate figures of loss/acc curves
plotloss = plot(1:n_epocas,[train_loss',test_loss'], label=["train loss" "test loss"], linewidth=2)
plotacc = plot(1:n_epocas',[train_acc',test_acc'], label=["train acc" "test acc"], linewidth=2)

savefig(plotloss,"loss_curves.png")
savefig(plotacc,"acc_curves.png")


# Save the model as .mat file variable 
matwrite("modelo_4x4_LabelEnc.mat", Dict(
    "W1" => W1,
    "W2" => W2,
    "W3" => W3,
    "b1" => b1,
    "b2" => b2,
    "b3" => b3
); compress = true)

matwrite("loss_acc_4x4_LabelEnc.mat", Dict(
    "train_loss" => train_loss,
    "train_acc" => train_acc,
    "test_loss" => test_loss,
    "test_acc" => test_acc,

); compress = true)