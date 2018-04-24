import cPickle,numpy,pyfits
import pymc,os,glob
from math import pi,log10
from pylensNew import *
from imageSim import profiles,convolve,models,SBModels
import indexTricks as iT
import random
from scipy.linalg import toeplitz
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix,dia_matrix,csc_matrix,identity,coo_matrix
import pylab as plt
from pylensNew import standardObjects as so
from pylensNew import pixellatedTools as pT
import numpy
import sys


#load the data
Name = "Mock1"
maskname="mask"
maskname="tight"
rdir = '/data/tcollett/pylensdistro/pylensdistro/%s'%Name

image = pyfits.open('%s/image.fits'%rdir)[0].data.copy()
sig = pyfits.open('%s/sigma.fits'%rdir)[0].data.copy()
PSF = pyfits.open('%s/psf.fits'%rdir)[0].data.copy()
psf=PSF.copy()
mask = pyfits.open('%s/%s.fits'%(rdir,maskname))[0].data.copy()
mask = mask==1
outmask = mask!=1
imshape=image.shape

#the image and sigmamap has to be flattened:
image=image.ravel()
sig=sig.ravel()

#The pixellated source fitting needs to know the uncertainty in the masked region. It's quickest to make it once at the start, as a sparse diagonal matrix.
from scipy.sparse import diags
vflt = sig[mask.flatten()]**2
cmat = diags(1./vflt,0)

#The psf has to be turned into a sparse matrix operator (instead of a convolution kernel
maskedpsf,fullpsf= pT.trimandmaskPSFMatrix(psf,imshape,mask,size=10)


#define the image co-ordiante system
indexes=iT.overSample(imshape,1)
yc=indexes[0].flatten()
xc=indexes[1].flatten()
x0,y0=xc.mean(),yc.mean()

#define the lens mass model
isothermal=False
LX = pymc.Uniform('LX',x0-80.,x0+80.)
LY = pymc.Uniform('LY',y0-80.,y0+80.)
LB = pymc.Uniform('LB',1,100)
LQ = pymc.Uniform('LQ',0.2,1.)
LP = pymc.Uniform('LP',-180.,180.)
lenspars=[LX,LY,LB,LQ,LP]
lenscov=[0.05,0.05,0.05,0.05,2.]
if isothermal:
    var = {'x':LX-1,'y':LY,'b':LB,'q':LQ,'pa':LP}
    const = {'eta':1}
else :
    LE = pymc.Uniform('LE',0.5,1.5)
    const = {}
    var = {'x':LX-1,'y':LY,'b':LB,'q':LQ,'pa':LP,'eta':LE}
    lenspars+=[LE]
    lenscov+=[0.1]
lens = massmodel.PowerLaw('lens',var,const)
lenses=[lens]
XB = pymc.Uniform('XB',0.,0.2,value=0.064199)
XP = pymc.Uniform('XP',-180.,180.,value=67.332210)
lenspars +=[XB,XP]
lenscov += [0.02,0.3]
lenscov=list(numpy.array(lenscov)*1.)
const = {}
var = {'x':LX-1,'y':LY,'b':XB,'pa':XP}
es = massmodel.ExtShear('shear',var,const) 
lenses.append(es)

#define the galaxy light model
gals=[]
galpars=[]
galcov=[]
fixamp=False
GX0 = pymc.Uniform('GX0',x0-25.,x0+25.,value=x0)
GY0 = pymc.Uniform('GY0',y0-25.,y0+25.,value=y0)
GP0 = pymc.Uniform('GP0',-180.,180.,value=-23)
GQ0 = pymc.Uniform('GQ0',0.2,1,value=0.79)
GR0 = pymc.Uniform('GR0',1.,100.,value=10)
GN0 = pymc.Uniform('GN0',0.25,8.,value=2.34)
galpars+= [GX0,GY0,GP0,GQ0,GR0,GN0]
galcov += list(numpy.array([0.02,0.02,.5,0.01,0.4,0.03]))#the typical uncertainty you expect for each parameter, varies (significantly) with data-quality.
if fixamp!=True:
    GA0 = pymc.Uniform('GA0',0,100.,value=0.1)
    galpars +=[GA0]
    galcov += [0.01]
    var = {'x':GX0-1,'y':GY0,'q':GQ0,'pa':GP0,'re':GR0,'n':GN0,'amp':GA0}
    gals.append(SBModels.Sersic('galaxy0',var))
else: 
    var = {'x':GX0-1,'y':GY0,'q':GQ0,'pa':GP0,'re':GR0,'n':GN0,'amp':1}
    gals.append(SBModels.Sersic('galaxy0',var))
GX1 = pymc.Uniform('GX1',x0-25.,x0+25.,value=x0)
GY1 = pymc.Uniform('GY1',y0-25.,y0+25.,value=y0)
GP1 = pymc.Uniform('GP1',-180.,180.,value=-23)
GQ1 = pymc.Uniform('GQ1',0.2,1,value=0.79)
GR1 = pymc.Uniform('GR1',1.,100.,value=10)
GN1 = pymc.Uniform('GN1',0.25,8.,value=2.34)
galpars+= [GX1,GY1,GP1,GQ1,GR1,GN1]
galcov += list(numpy.array([0.1,0.1,.5,0.01,0.7,0.03]))
if fixamp!=True:
    GA1 = pymc.Uniform('GA1',0,100.,value=0.1)
    galpars +=[GA1]
    galcov += [0.001]
    var = {'x':GX1-1,'y':GY1,'q':GQ1,'pa':GP1,'re':GR1,'n':GN1,'amp':GA1}
    gals.append(SBModels.Sersic('galaxy1',var))
else: 
    var = {'x':GX1-1,'y':GY1,'q':GQ1,'pa':GP1,'re':GR1,'n':GN1,'amp':1}
    gals.append(SBModels.Sersic('galaxy1',var))

#define the source model (It's an NbyN pixelgrid, regularized in someway)
Npix=80
sshape = (Npix,Npix)
rmat= pT.getRegularizationMatrix(numpy.arange(sshape[0]),numpy.arange(sshape[1]),kernel="c")#c=curvature regularization. Other options are in the pT code.
regconst = pT.RegConst('reg',300) #this is an object that stores the current value of the regularization strength


#Define the start values of the model parameters.
GA0.value=0.19422
GA1.value=0.02781
GN0.value=2.51524
GN1.value=1.53063
GP0.value=-50.11549
GP1.value=125.46486
GQ0.value=0.77557
GQ1.value=0.79209
GR0.value=14.34225
GR1.value=59.78337
GX0.value=99.75266
GX1.value=101.56097
GY0.value=99.01229
GY1.value=95.70958
LB.value=67.69398
LE.value=0.99204
LP.value=-54.35966
LQ.value=0.85060
LX.value=100.00661
LY.value=96.63027
XB.value=0.04434
XP.value=-5.32800

presubtractgal=True #you can subtract the galaxy at the very start, and fix it like that, or let it vary with the other params
if presubtractgal:
    for gal in gals:
        gal.setPars()
        image -= fullpsf*gal.pixeval(xc,yc,csub=11)
    gals=[]

    pars=lenspars
    cov=lenscov
else:
    pars=lenspars+galpars
    cov=lenscov+galcov


#This is the way to call the modelling code:
interactive=True
#interactive=False
res,reg=ssplModel.ssplModelPixellated(lenses,gals,sshape,image,sig,imshape,cmat,mask,xc,yc,maskedpsf,rmat,fullpsf,regconst.value,outputimages=False,fixreg=False,regmin=1e-100,regacc=0.001,interactive=interactive,name=Name)

regmin=reg/3.#we don't know the regularization a-priori, (or even the order of magnitde), but it can't get too small or the source can fragment.

regconst.value = reg #regconst stores the current value of the regularization strength


#now we phrase it as a likelihood (res is the value of the merit function - like a chisq but penalizes spiky sources as well)
@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0
    res,reg=ssplModel.ssplModelPixellated(lenses,gals,sshape,image,sig,imshape,cmat,mask,xc,yc,maskedpsf,rmat,fullpsf,regconst.value,outputimages=False,fixreg=False,regmin=regmin,regacc=0.001,interactive=False)
    regconst.value = reg # update the current regularization value
    lp+=res
    value=lp
    return lp
@pymc.observed
def likelihood(value=0.,lp=logP):
    value=lp
    return lp


NWALKERS=32
NTHREADS=8
NTEMPS=3
NSAMPLES=20

fixfirstwalker=False #Option to make sure one of the walkers starts at the location you've fed it. (Other walkers are in a n-D ball around the start position, radius cov in each parameter)

#Now do the sampling (You can use whatever your favourite sampler is)

#from LensEmcee import TemperedEmcee as TE
#T=TE(pars+[likelihood],(numpy.array(cov)*1),[regconst],nwalkers=NWALKERS,nthreads=NTHREADS,ntemps=NTEMPS)

from LensEmcee import ParEmcee as PE
T=PE(pars+[likelihood],(numpy.array(cov)*1),[regconst],nwalkers=NWALKERS,nthreads=NTHREADS)
T.sample(NSAMPLES,burn=0,fixfirstwalker=fixfirstwalker)

pos,prob,result,bestpos = T.output()


keys=sorted(result.keys())

for key in keys:
    outtext=("%s=%f +/- %f"%(key,numpy.mean(result[key]),numpy.std(result[key])))
    print outtext
for key in keys:
    outtext=("%s.value=%f"%(key,numpy.mean(result[key]))).strip()
    print outtext

#save the output
outstr="output"
outlist=glob.glob("%s*.pkl"%outstr)
globres=len(outlist)
f=open("%s%i.pkl"%(outstr,globres),"wb")
cPickle.dump([pos,prob,result],f,2)
f.close()



"""

for i in range(len(prob[0,:])):
    p=prob[:,i].ravel()
    if p.max()==prob.max():
        I=i*1
        argm=p.argmax()

for i in range(len(keys)):
    outtext=("%s.value=%f"%(keys[i],result[keys[i]][argm][I])).strip()
    print outtext

try: 
    g=open(startpars,"r")
    h=g.readline()
    g.close() 
    try: float(h)
    except ValueError: h=-1e99
except IOError:  h=-1e99
if float(h)<prob.max():
        g=open(startpars,"w")
        g.write("%f\n"%prob.max())
        g.write("%f\n"%pscale)
        g.write("%f\n"%sshape[0])
        g.write("%f\n"%sshape[1])
        for i in range(len(keys)):
            outtext=("%s.value=%f"%(keys[i],result[keys[i]][argm][I])).strip()
            print outtext
            g.write("%s\n"%(outtext))
            
        g.close()

print("Mean acceptance fraction: {0:.3f}"
                .format(numpy.mean(T.sampler.acceptance_fraction)))
"""
