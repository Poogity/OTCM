import numpy as np
import cv2 as cv
import pulp
import matplotlib.pyplot as plt

def variable_sort_key(variable):
    return int(variable.name[2:])
def OTCM(img): 
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    L = len(hist)
    Lnew = 256
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
        delta=1
        if p[i]<=0.05: delta=0
        prob += s[i] >= delta/d, f'constraint_{i+2}'
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
    
    # Print the value of the variables at the optimum
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
    return newImg

img = cv.imread('castle.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
r, g, b = cv.split(img)
if (r==g).all() and (g==b).all():
    newImg = OTCM(r)  
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.hist(img.ravel(),255,[0,255])
    plt.show()
    plt.imshow(newImg, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.hist(newImg.ravel(),255,[0,255])
    plt.show()
else:    #coloured img
    # segregate color streams
    cdf_b = OTCM(b)
    cdf_g = OTCM(g)
    cdf_r = OTCM(r)   
    
    # mask all pixels with value=0 and replace it with mean of the pixel values 
    imgB = np.ma.masked_equal(cdf_b,0)
    imgB = (imgB - imgB.min())*255/(imgB.max()-imgB.min())
    imgBfinal = np.ma.filled(imgB,0).astype('uint8')
  
    imgG = np.ma.masked_equal(cdf_g,0)
    imgG = (imgG - imgG.min())*255/(imgG.max()-imgG.min())
    imgGfinal = np.ma.filled(imgG,0).astype('uint8')
    
    imgR = np.ma.masked_equal(cdf_r,0)
    imgR = (imgR - imgR.min())*255/(imgR.max()-imgR.min())
    imgRfinal = np.ma.filled(imgR,0).astype('uint8')
    
    newImg = cv.merge((imgRfinal, imgGfinal, imgBfinal))
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.hist(img.ravel(),255,[0,255])
    plt.show()
    plt.imshow(newImg)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.hist(newImg.ravel(),255,[0,255])
    plt.show()
    