import numpy as np
import cv2 as cv
import pulp
import matplotlib.pyplot as plt

def variable_sort_key(variable):
    return int(variable.name[2:])


img = cv.imread('kodim12.png', cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([img],[0],None,[256],[0,256])
L = len(hist)
Lnew = L
d = 3
u= 2
p = np.empty(L,dtype='float')
for i in range(L):
    p[i] = hist[i]/(img.size)


prob = pulp.LpProblem("histEQ", pulp.LpMaximize)

 # Variables
s =  pulp.LpVariable.dicts('s', range(L))
# Objective
prob += pulp.lpSum([p[i]* s[i] for i in range(L)]), 'obj'
# (the name at the end is facultative)

# Constraints
prob += pulp.lpSum([s[j] for j in range(L)]) <= Lnew, f'constraint_{1}'
for i in range(L):
    prob += s[i] >= 1/d, f'constraint_{i+2}'
    prob += s[i] <= u, f'constraint_{i+2+L}'    
# Solve the problem using the default solver
prob.solve()

 # Print the status of the solved LP
print("Status:", pulp.LpStatus[prob.status])

 # Print the value of the objective
print("objective =", pulp.value(prob.objective))

sumArray = 0
i=0
t = np.empty(L)

# Print the value of the variables at the optimal
variables = prob.variables()
# Sort the variables based on the numerical index using the custom key function
sorted_variables = sorted(variables, key=variable_sort_key)

for v in sorted_variables:
    sumArray += v.varValue
    t[i] = int(sumArray + 0.5)
    i+=1
    
width,length = len(img[:,0]),len(img[0])
newImg = np.empty((width,length))
for i in range(width):
    for j in range(length):
        newImg[i][j] = t[img[i][j]]

plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
plt.hist(img.ravel(),L,[0,L])
plt.show()
plt.imshow(newImg, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
plt.hist(newImg.ravel(),Lnew,[0,Lnew])
plt.show()