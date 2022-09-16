import matplotlib.pyplot as plt
from scipy import stats   

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x,y) # key values of linear regression

def func(x):
    return slope*x + intercept  # y = a*x+b

plt.scatter(x,y)
plt.plot(x,list(map(func,x)))
plt.show()

# the coefficient of correlation : r,  r in (-1,1),  r = 1 or r = -1 => 100% ,  r = 0 => means not good relationship

print("r : ",round(r,3)) 

# according to r there is a good relationship so predict the speed of any value 
print(func(15))