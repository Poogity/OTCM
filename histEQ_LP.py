import numpy as np
import cv2 as cv
import pulp
import matplotlib.pyplot as plt

def variable_sort_key(variable):
    return int(variable.name[2:])


img = cv.imread('kodim12.png', cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([img],[0],None,[256],[0,256])
L = 256
Lnew = 256
d = 2
p = np.empty(L,dtype='float')
for i in range(L):
    p[i] = hist[i]/(img.size)


prob = pulp.LpProblem("histEQ", pulp.LpMaximize)

 # Variables
s =  pulp.LpVariable.dicts('s', range(L), cat=pulp.LpInteger)
# Objective
prob += pulp.lpSum([p[i]* s[i] for i in range(L)]), 'obj'
# (the name at the end is facultative)

# Constraints
prob += pulp.lpSum([s[j] for j in range(L)]) <= Lnew-1, f'constraint_{1}'
for i in range(L):
    prob += s[i] >= 0, f'constraint_{i+2}'
for j in range(L-d):
   prob += pulp.lpSum([s[i] for i in range(j,j+d)]) >= 1, f'constraint_{j+2+L}'
   
   
   
# Solve the problem using the default solver
prob.solve()

 # Print the status of the solved LP
print("Status:", pulp.LpStatus[prob.status])

 # Print the value of the objective
print("objective =", pulp.value(prob.objective))

sumArray = 0
i=0
t = np.empty(L)
# Print the value of the variables at the optimum
variables = prob.variables()
# Sort the variables based on the numerical index using the custom key function
sorted_variables = sorted(variables, key=variable_sort_key)

for v in sorted_variables:
    print(f'{v.name} = {v.varValue:5.2f}')
    sumArray += v.varValue
    t[i] = sumArray
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
