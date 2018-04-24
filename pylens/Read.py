import cPickle,numpy,pyfits
import pymc
from math import pi,log10
import random
import pylab as plt
import glob,os,sys
a,b=0,-1

resfiles=glob.glob('/data/tcollett/Lenses/*/*/*.pkl')

if len(sys.argv)>1:
    resfiles=sys.argv[1:]

bsofar=-1e100
for res in resfiles:
    print res
    f=open(res,"rb")
    try: pos,prob,res0,flatchain,startpars=cPickle.load(f)
    except cPickle.UnpicklingError: continue
    print prob.max()
    #prob=prob.T
    #pos=pos.T

    if prob.max()<bsofar:
        continue
    else:bsofar=prob.max()

    for i in range(len(prob[0,:])):  
        p=prob[:,i].ravel()
        plt.plot(range(len(p)),p,c='k',lw=0.1)
        if p[-1]==p[int(len(p)/3)]:
            plt.plot(range(len(p)),p,c='r',lw=2)
    plt.ylim([-1e4,prob.max()+100])
    plt.title(prob.max())
    plt.show(block=True)


    keys=res0.keys()
    keys=sorted(keys)
    prob=prob.T
    for i in range(len(prob[0,:])):
        p=prob[:,i].ravel()
        if p.max()==prob.max():
        #    print i,p.max(),p.argmax()
            I=i*1
            argm=p.argmax()


    for el in keys: 
        print "      %s.value=%.5f"%(el,res0[el][argm][I])


exit()










#code to update startpars names.
"""
#resfiles=glob.glob('/data/tcollett/Lenses/HST1/S1/*.pkl')
for res in resfiles:
    print res
    f=open(res,"rb")
    pos,prob,result,flatchain,startpars=cPickle.load(f)
    f.close() 
    a=startpars.split('HST1/S2')
    print a
    if len(a)>1:
        startpars=a[0]+'HST1'+a[1]
        f=open(res,"wb")
        cPickle.dump([pos,prob,result,flatchain,startpars],f,2)
        f.close()
"""

for res in resfiles:
    print res
    f=open(res,"rb")
    try: pos,prob,res0,flatchain,startpars=cPickle.load(f)
    except cPickle.UnpicklingError: continue
    print prob.max()
    prob=prob.T

    keys=res0.keys()
    keys=sorted(keys)
    for i in range(len(prob[0,:])):
        p=prob[:,i].ravel()
        if p.max()==prob.max():
        #    print i,p.max(),p.argmax()
            I=i*1
            argm=p.argmax()

    for el in keys: 
        print "%s.value="%el,res0[el][argm][I]
    f.close()

    try: 
        g=open(startpars,"r")
        h=g.readline()
        g.close() 
        try: float(h)
        except ValueError: h=-1e99
    except IOError:  h=-1e99

    if float(h)<prob.max():
        try: 
            g=open(startpars,"r")
            g=open(startpars,"r")
            content=g.readlines()[1:]
            g.close()
            g=open(startpars,"w")
        except IOError:  
            g=open(startpars,"w")
            content=[]

        print "nothing"
        g.write("%f\n"%prob.max())

        for parline in content:
            name=parline.split('.')[0]
            if name not in keys:
                g.write(parline)


        for i in range(len(keys)):
            #print keys[i],res0[keys[i]][argm][I]
            outtext=("%s.value=%f"%(keys[i],res0[keys[i]][argm][I])).strip()

            print outtext
            g.write("%s\n"%(outtext))
            
        g.close()


    
    if (float(h)<prob.max()+2 or len(resfiles)==1):        
        #first plot the probabilities.
    #if 1==2:  
        
        for i in range(len(prob[0,:])):  
            p=prob[:,i].ravel()
            plt.plot(range(len(p)),p,c='k',lw=0.1)
            if p[-1]==p[int(len(p)/3)]:
                plt.plot(range(len(p)),p,c='r',lw=2)
        plt.title(prob.max())
        plt.show(block=True)
        
        

        a=-500####
        b=-1
        print "_____________________________"
        print res
        for el in keys:
            if el in ['LE0','LB0','LQ0','XB']:
                """
                plt.plot(res0[el])
                plt.title(res)
                plt.xlabel(el)
                plt.show(block=True)
                """
                print el,numpy.median(res0[el][prob>prob.max()-100][a:b]),numpy.std(res0[el][prob>prob.max()-100][a:b])
                pass

        print prob.max()
"""
for i in range(len(prob[0,:])):  
    p=prob[:,i].ravel()
    plt.plot(range(len(p)),p,c='k',lw=0.1)
plt.show(block=True)
for el in keys:
            if el in ['LE0','LB0','LQ0','XB']:
                plt.plot(res0[el])
                plt.title(res)
                plt.xlabel(el)
                plt.show(block=True)
"""
