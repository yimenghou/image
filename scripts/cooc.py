
import numpy as np
import cv2

def cooc(inputmtx, angle, greylevel=16, norm_flag=1):
    # calculate co-occurence matrix, take only greyscale image

    a,b = inputmtx.shape
    inputmatrix = np.zeros(inputmtx.shape)
    cocurrentmtx = np.zeros((256/greylevel, 256/greylevel))
    
    for x in range(inputmtx.shape[0]):
        for y in range(inputmtx.shape[1]):            
            inputmatrix[x,y] = round(inputmtx[x,y]/greylevel )   

    if angle == 0:
        for i in range(a):
            for j in range(b-1):
                leftitem = inputmatrix[i,j]
                rightitem = inputmatrix[i,j+1]
                cocurrentmtx[leftitem-1,rightitem-1] += 1
                cocurrentmtx[rightitem-1,leftitem-1] += 1
    elif angle == 45:
        for i in range(a-1):
            for j in range(1,b):
                upperrightitem = inputmatrix[i,j]
                lowerleftitem = inputmatrix[i+1,j-1]
                cocurrentmtx[upperrightitem-1,lowerleftitem-1] += 1
                cocurrentmtx[lowerleftitem-1,upperrightitem-1] += 1
    elif angle == 90:
        for i in range(a-1):
            for j in range(b):
                upperitem = inputmatrix[i,j]
                loweritem = inputmatrix[i+1,j]
                cocurrentmtx[upperitem-1,loweritem-1] += 1
                cocurrentmtx[loweritem-1,upperitem-1] += 1 
    elif angle == 135:
        for i in range(a-1):
            for j in range(b-1):
                upperleftitem = inputmatrix[i,j]
                lowerrightitem = inputmatrix[i+1,j+1]
                cocurrentmtx[upperleftitem-1,lowerrightitem-1] += 1
                cocurrentmtx[lowerrightitem-1,upperleftitem-1] += 1  

    if norm_flag == 1:
        return cocurrentmtx/float(sum(sum(cocurrentmtx)))
    else:
        return cocurrentmtx

def featureDescriptor(coocmtx):
    # feature descriptors: asm, con, ent, corr

    a,b = coocmtx.shape
    mu_a = coocmtx.mean(1)
    mu_b = coocmtx.mean(0)
    sigma_a = coocmtx.std(1)
    sigma_b = coocmtx.std(0)    
    asm_val, con_val, ent_val, corr_val, home_val = 0,0,0,0,0
    
    # asm_val = sum(sum(coocmtx*coocmtx))
    for i in range(a):
        for j in range(b):
            cont_val += coocmtx[i,j]*((i-j)**2)            
            if coocmtx[i,j] !=0:
                ent_val += coocmtx[i,j]*np.log2(coocmtx[i,j])
            # corr_val += i*j*coocmtx[i,j]
            home_val += coocmtx[i,j]/float(1+abs(i-j))
            corr_val += ((i-mu_a*i)*(j-mu_b*j)*coocmtx[i,j])/np.dot(sigma_a,sigma_b)
            asm_val += coocmtx[i,j]**2

    # corr_val = (corr_val-np.dot(mu_a,mu_b))/np.dot(sigma_a,sigma_b)
    ent_val = -ent_val
    
    return asm_val, cont_val, ent_val, corr_val, home_val

path = r'C:\Users\westwell\Desktop\xProject\landscape.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

print "image shape:", img.shape
coocMtx = cooc(img, 45)
print coocMtx
ans = featureDescriptor(coocMtx)
for i in range(5):
    print ans[i]