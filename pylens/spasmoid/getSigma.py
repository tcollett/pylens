import numpy

def clip(arr,nsig=3.):
    a = arr.copy()
    m,s,l = a.mean(),a.std(),a.size

    while 1:
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s
        m,s,l = a.mean(),a.std(),a.size

def getSigma(img,mask,exptime):
    mask = mask==0
    m,s = clip(img[fullmask])
    cond = (img>1.2*s)
    cnts = img*0.+s**2
    cnts[cond] += img[cond]/exptime
    return cnts**0.5
        
