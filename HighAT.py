"""
Code in this file is based on codes of the paper ''Ribeiro, A. H., Zachariah, D., Bach, F., and Sch ̈on, T. B. (2023). 
Regularization properties of adversarially-trained linear regression. In Thirty-seventh Conference on Neural Information Processing Systems''
and the paper ''Jose Blanchet, Karthyek Murthy, and Fan Zhang. 
Optimal transport-based distributionally robust optimization: Structural properties and iterative schemes. Mathematics of Operations Research, 47(2):1500–1529''
"""


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%
#linear regression generator
class LinearRegressionDataGenerator(object):
    def __init__(self, var, std = 0.1):
        var = np.array(var)
        assert(var.ndim is 1)

        self.var = var
        self.dim = var.size
        self.std = std

    def generate(self, data_num):
        X = np.random.rand(data_num, self.dim)
        y = X.dot(self.var) + self.std * np.random.randn(data_num)
        return X, y
#%% Function of Group Adversarial Training
class GroupAdversarialTraining:
    def __init__(self, X, y):
        m, n = X.shape
        self.param = cp.Variable(n)
        self.X = X
        self.y = y
        self.m = m
        self.n = n
        self.warm_start = False

    def formulate_problem(self, groups, weights):
        abs_error = cp.abs(self.X @ self.param - self.y)
        group_norms = [weights[i] * cp.norm(self.param[groups[i]], 2) for i in range(len(groups))]
        group_weighted_norm = cp.sum(group_norms)
        adv_loss = 1 / self.m * cp.sum((abs_error +  group_weighted_norm ) ** 2)
        self.prob = cp.Problem(cp.Minimize(adv_loss))

    def __call__(self, groups, weights, **kwargs):
        try:
            self.formulate_problem(groups, weights)
            self.prob.solve(warm_start=self.warm_start, **kwargs)
            v = self.param.value
        except Exception as e:
            print(f"An error occurred: {e}")
            v = np.zeros(self.param.shape)
        return v    

##################
# Model 1
##################
#%%true value
true_var = np.zeros(500)
true_var[0:4] = [0.1,0.2,0.15,0.25]
true_var[-4:] = [0.9,0.95,1,1.05]
nvalue = [50,100,150,200,250,300,350,400]
#%%training
FullPredictionError = []
FullPredictionError1 = []
COEFFICIENT = []
COEFFICIENT1 = []
for R in range(0,5):
    Xfull, yfull = LinearRegressionDataGenerator(true_var).generate(500)
    E = []
    PredictionError = []
    group_size = 4
    group_number = len(true_var)/group_size
    for sample_size in nvalue:
        X = Xfull[0:sample_size,:]
        y = yfull[0:sample_size]
        a = np.sqrt(1/sample_size)
        weights = [a]* int(group_number)
        lst = list(range(len(true_var)))
        groups = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
        linfadvtrain = GroupAdversarialTraining(X, y)
        estimator = lambda X, y, groups,weights:  linfadvtrain(groups=groups,weights=weights)
        coef = estimator(X, y, groups,weights)
        print(sample_size)
        predictionerror= np.linalg.norm( X.dot(coef)-X.dot(true_var) )**2/sample_size
        E.append(coef)
        PredictionError.append(predictionerror)   
    FullPredictionError.append(PredictionError) 
    COEFFICIENT.append(E)
    E1 = []
    PredictionError1 = []
    group_size = 1
    group_number = len(true_var)/group_size
    for sample_size in nvalue:
        X = Xfull[0:sample_size,:]
        y = yfull[0:sample_size]
        a =  np.sqrt(1/sample_size)
        weights = [a]* int(group_number)
        lst = list(range(len(true_var)))
        groups = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
        linfadvtrain = GroupAdversarialTraining(X, y)
        estimator = lambda X, y, groups,weights:  linfadvtrain(groups=groups,weights=weights)
        coef = estimator(X, y, groups,weights)
        predictionerror1= np.linalg.norm( X.dot(coef)-X.dot(true_var) )**2/sample_size
        PredictionError1.append(predictionerror1)
        print(sample_size)
        E1.append(coef)
    FullPredictionError1.append(PredictionError1) 
    COEFFICIENT1.append(E1)

    



#%%plot
plt.figure(dpi=1000)
N = np.log10(np.array([50,100,150,200,250,300,350,400]))
FullPredictionErrorlog = np.log10(np.array(FullPredictionError))
GATmean = FullPredictionErrorlog.mean(axis=0)
GATstd = FullPredictionErrorlog.std(axis=0)


FullPredictionError1log = np.log10(np.array(FullPredictionError1))
ATmean = FullPredictionError1log.mean(axis=0)
ATstd = FullPredictionError1log.std(axis=0)



plt.plot(N, GATmean,label="Group Adversarial Training",color='teal', marker = 's',markersize = 5,lw=1)
plt.fill_between(N,GATmean - GATstd, GATmean + GATstd,color='teal',alpha=0.2)


plt.plot(N, ATmean, label = "Adversarial Training",color='purple',marker = 'D',markersize = 5,lw=1)
plt.fill_between(N,ATmean - ATstd,ATmean + ATstd,color='purple',alpha=0.2)


plt.xlabel(r'$log(Sample \ Size)$',fontsize=13)
plt.ylabel(r'$log(Error)$',fontsize=13) 
plt.legend(fontsize=10)
plt.title("Prediction Error")
plt.savefig("predictionerror.pdf", format="pdf")

#%%Plot the Parameter Estimation for Group Adversarial Training

COEFFICIENTarray = np.array(COEFFICIENT).T
COEFFICIENTmean = COEFFICIENTarray.mean(axis=2)
COEFFICIENTstd = COEFFICIENTarray.std(axis=2)



matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] =10
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 15
plt.figure(dpi=1000)
matplotlib.rcParams['lines.linewidth'] = 1


fig, ax1 = plt.subplots()

# Plot on the primary y-axis
for j in np.arange(0, 500):
    ax1.plot(nvalue, COEFFICIENTmean[j])
    ax1.fill_between(nvalue, COEFFICIENTmean[j] - COEFFICIENTstd[j], COEFFICIENTmean[j] + COEFFICIENTstd[j], alpha=0.2)

ax1.set_title('Group Adversarial Training')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Estimation')
ax1.set_xlim(50, 400)
ax1.set_ylim(0, 1.1)


# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylim(0, 1.1)
final = COEFFICIENTmean.T[-1][COEFFICIENTmean.T[-1]>0.05]
test = np.round(final,2)
ax2.set_yticks(test) 
 # Set the specific tick positions
string_array = [str(value) for value in test]
ax2.set_yticklabels(string_array) 


# Example plot on the secondary y-axis (you can replace this with your actual data)
# ax2.plot(nvalue, another_metric, color='red')  # Replace `another_metric` with your actual data for the second y-axis
# ax2.set_ylabel('Another Metric', color='red')

plt.savefig("GAT.pdf", format="pdf")
plt.show()


#%%AT
COEFFICIENTarray1 = np.array(COEFFICIENT1).T
COEFFICIENTmean1 = COEFFICIENTarray1.mean(axis=2)
COEFFICIENTstd1 = COEFFICIENTarray1.std(axis=2)



matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] =10
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 15
plt.figure(dpi=1000)
matplotlib.rcParams['lines.linewidth'] = 1


fig, ax1 = plt.subplots()

# Plot on the primary y-axis
for j in np.arange(0, 500):
    ax1.plot(nvalue, COEFFICIENTmean1[j])
    ax1.fill_between(nvalue, COEFFICIENTmean1[j] - COEFFICIENTstd1[j], COEFFICIENTmean1[j] + COEFFICIENTstd1[j], alpha=0.2)

ax1.set_title('Adversarial Training')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Estimation')
ax1.set_xlim(50, 400)
ax1.set_ylim(0, 1.1)


# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylim(0, 1.1)
final = COEFFICIENTmean1.T[-1][COEFFICIENTmean1.T[-1]>0.05]
test = np.round(final,2)
ax2.set_yticks(test) 
 # Set the specific tick positions
string_array = [str(value) for value in test]
ax2.set_yticklabels(string_array) 


# Example plot on the secondary y-axis (you can replace this with your actual data)
# ax2.plot(nvalue, another_metric, color='red')  # Replace `another_metric` with your actual data for the second y-axis
# ax2.set_ylabel('Another Metric', color='red')

plt.savefig("AT.pdf", format="pdf")
plt.show()
#####################
# Model2: Categorical 
#####################
#%%
from sklearn.preprocessing import OneHotEncoder
# True coefficients for numeric features
true_coef_numeric = np.array([0.4, 0.5, 0.6])  # Coefficients for 'feature_1' and 'feature_2'

# True coefficients for categorical variables (one-hot encoded categories)
true_coef_categorical = np.array([0.2, 0.3, 0.7])

true_coef_zero = np.array([0]*594)

true_var = np.hstack([true_coef_numeric, true_coef_zero,true_coef_categorical ])
#%%
nvalue = [50,100,150,200,250,350,450,550]
FullPredictionError = []
FullPredictionError1 = []
COEFFICIENT = []
COEFFICIENT1 = []
for R in range(0,5):
    X_numeric = np.random.rand(600, 3) 
    
    X_zero = np.random.rand(600, 594)


    categories = np.random.choice(['A', 'B', 'C', 'D'], size = 600)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = encoder.fit_transform(categories.reshape(-1, 1))

    Xfull = np.hstack([X_numeric, X_zero, X_encoded])


    yfull= (X_numeric @ true_coef_numeric +  X_zero @ true_coef_zero + X_encoded @ true_coef_categorical  + np.random.randn(600) * 0.1)
    
    E = []
    PredictionError = []
    group_size = 3
    group_number = len(true_var)/group_size
    for sample_size in nvalue:

        X = Xfull[0:sample_size,:]
        y = yfull[0:sample_size]
        a = np.sqrt(1/sample_size)
        weights = [a]* int(group_number)
        lst = list(range(len(true_var)))
        groups = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
        linfadvtrain = GroupAdversarialTraining(X, y)
        estimator = lambda X, y, groups,weights:  linfadvtrain(groups=groups,weights=weights)
        coef = estimator(X, y, groups,weights)
        print(sample_size)
        predictionerror= np.linalg.norm( X.dot(coef)-X.dot(true_var) )**2/sample_size
        E.append(coef)
        PredictionError.append(predictionerror)   
    FullPredictionError.append(PredictionError) 
    COEFFICIENT.append(E)
    E1 = []
    PredictionError1 = []
    group_size = 1
    group_number = len(true_var)/group_size
    for sample_size in nvalue:
        X = Xfull[0:sample_size,:]
        y = yfull[0:sample_size]
        a =  np.sqrt(1/sample_size)
        weights = [a]* int(group_number)
        lst = list(range(len(true_var)))
        groups = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
        linfadvtrain = GroupAdversarialTraining(X, y)
        estimator = lambda X, y, groups,weights:  linfadvtrain(groups=groups,weights=weights)
        coef = estimator(X, y, groups,weights)
        predictionerror1= np.linalg.norm( X.dot(coef)-X.dot(true_var) )**2/sample_size
        PredictionError1.append(predictionerror1)
        print(sample_size)
        E1.append(coef)
    FullPredictionError1.append(PredictionError1) 
    COEFFICIENT1.append(E1)

    
#%%plot
plt.figure(dpi=1000)
N = np.log10(np.array([50,100,150,200,250,350,450,550]))
FullPredictionErrorlog = np.log10(np.array(FullPredictionError))
GATmean = FullPredictionErrorlog.mean(axis=0)
GATstd = FullPredictionErrorlog.std(axis=0)


FullPredictionError1log = np.log10(np.array(FullPredictionError1))
ATmean = FullPredictionError1log.mean(axis=0)
ATstd = FullPredictionError1log.std(axis=0)



plt.plot(N, GATmean,label="Group Adversarial Training",color='teal', marker = 's',markersize = 5,lw=1)
plt.fill_between(N,GATmean - GATstd, GATmean + GATstd,color='teal',alpha=0.2)


plt.plot(N, ATmean, label = "Adversarial Training",color='purple',marker = 'D',markersize = 5,lw=1)
plt.fill_between(N,ATmean - ATstd,ATmean + ATstd,color='purple',alpha=0.2)


plt.xlabel(r'$log(Sample \ Size)$',fontsize=13)
plt.ylabel(r'$log(Error)$',fontsize=13) 
plt.legend(fontsize=10)
plt.title("Prediction Error")
plt.savefig("predictionerror1.pdf", format="pdf")

#%%Plot the Parameter Estimation for Group Adversarial Training

COEFFICIENTarray = np.array(COEFFICIENT).T
COEFFICIENTmean = COEFFICIENTarray.mean(axis=2)
COEFFICIENTstd = COEFFICIENTarray.std(axis=2)



matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] =10
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 15
plt.figure(dpi=1000)
matplotlib.rcParams['lines.linewidth'] = 1


fig, ax1 = plt.subplots()

# Plot on the primary y-axis
for j in np.arange(0, 600):
    ax1.plot(nvalue, COEFFICIENTmean[j])
    ax1.fill_between(nvalue, COEFFICIENTmean[j] - COEFFICIENTstd[j], COEFFICIENTmean[j] + COEFFICIENTstd[j], alpha=0.2)

ax1.set_title('Group Adversarial Training')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Estimation')
ax1.set_xlim(50, 500)
ax1.set_ylim(0, 0.7)


# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylim(0, 0.7)
final = COEFFICIENTmean.T[-1][COEFFICIENTmean.T[-1]>0.05]
test = np.round(final,2)
ax2.set_yticks(test) 
 # Set the specific tick positions
string_array = [str(value) for value in test]
ax2.set_yticklabels(string_array) 



plt.savefig("GAT1.pdf", format="pdf")
plt.show()


#%%AT
COEFFICIENTarray1 = np.array(COEFFICIENT1).T
COEFFICIENTmean1 = COEFFICIENTarray1.mean(axis=2)
COEFFICIENTstd1 = COEFFICIENTarray1.std(axis=2)



matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] =10
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 15
plt.figure(dpi=1000)
matplotlib.rcParams['lines.linewidth'] = 1


fig, ax1 = plt.subplots()

# Plot on the primary y-axis
for j in np.arange(0, 600):
    ax1.plot(nvalue, COEFFICIENTmean1[j])
    ax1.fill_between(nvalue, COEFFICIENTmean1[j] - COEFFICIENTstd1[j], COEFFICIENTmean1[j] + COEFFICIENTstd1[j], alpha=0.2)

ax1.set_title('Adversarial Training')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Estimation')
ax1.set_xlim(50, 500)
ax1.set_ylim(0, 0.7)


# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylim(0, 0.7)
final = COEFFICIENTmean1.T[-1][COEFFICIENTmean1.T[-1]>0.05]
test = np.round(final,2)
ax2.set_yticks(test) 
 # Set the specific tick positions
string_array = [str(value) for value in test]
ax2.set_yticklabels(string_array) 



plt.savefig("AT1.pdf", format="pdf")
plt.show()