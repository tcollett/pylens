import pymc,numpy
from pylens import *

def buildLens(x0,y0,isothermal=False,spherical=False,externalshear=True,masssheet=False,startpars=None,LBprior=None):
    import massmodel
    LX = pymc.Uniform('LX',x0-10.,x0+10.,value=x0)
    LY = pymc.Uniform('LY',y0-10.,y0+10.,value=y0)
    if LBprior!=None:
        LB = pymc.Uniform('LB',LBprior[0],LBprior[1])
    else:
        LB = pymc.Uniform('LB',10.,180.)
    LQ = pymc.Uniform('LQ',0.2,1.0)
    LP = pymc.Uniform('LP',-180.,180.)
    lenspars=[LX,LY,LB,LQ,LP]
    lenscov=[0.05,0.05,0.05,0.05,2.]
    if isothermal:
        var = {'x':LX,'y':LY,'b':LB,'q':LQ,'pa':LP}
        const = {'eta':1}
    else :
        LE = pymc.Uniform('LE',0.4,1.6,value=1.)
        const = {}
        var = {'x':LX,'y':LY,'b':LB,'q':LQ,'pa':LP,'eta':LE}
        lenspars+=[LE]
        lenscov+=[0.1]
    lens = massmodel.PowerLaw('lens',var,const)

    lenses=[lens]

    if externalshear:
        XB = pymc.Uniform('XB',0.,0.2,value=0.02)
        XP = pymc.Uniform('XP',-180,180)
        lenspars +=[XB,XP]
        lenscov += [0.02,0.3]
        const = {}
        var = {'x':LX,'y':LY,'b':XB,'pa':XP}
        es = massmodel.ExtShear('shear',var,const) 
        lenses.append(es)

    if externalshear and masssheet:
        assert externalshear, "If you want a particular value of kappa you'll have to code it up"
        const = {}
        var = {'x':LX,'y':LY,'b':XB}
        kappaext = massmodel.MassSheet('kappaext',var,const) 
        lenses.append(kappaext)

    return lenses,lenspars,lenscov

def buildGal(x0,y0,components=1,gaussian=False,spherical=False,startpars=None,fixamp=False):
    from imageSim import SBModels
    GX={}
    GY={}
    GR={}
    GQ={}
    GP={}
    GN={}
    GA={}
    gals=[]
    for i in range(components):
        GX[i] = pymc.Uniform('GX%i'%i,x0-25.,x0+25.,value=x0)
        GY[i] = pymc.Uniform('GY%i'%i,y0-25.,y0+25.,value=y0)
        GP[i] = pymc.Uniform('GP%i'%i,-180.,180.,value=-23)
        if i==0:
            galpars=  [GX[i],GY[i],GP[i]]
            galcov =  list(numpy.array([0.1,0.1,1.])/1e0)
        else:
            galpars+= [GX[i],GY[i],GP[i]]
            galcov += list(numpy.array([0.05,0.05,1.])/1e0)
        GQ[i] = pymc.Uniform('GQ%i'%i,0.2,1,value=0.79)
        GR[i] = pymc.Uniform('GR%i'%i,1.,100.,value=10)
        GN[i] = pymc.Uniform('GN%i'%i,0.25,8.,value=2.34)
        galpars +=[GQ[i],GR[i],GN[i]]
        galcov += list(numpy.array([0.03,0.03,0.07])/1e0)
        if fixamp!=True:
            GA[i] = pymc.Uniform('GA%i'%i,0,100.,value=0.1)#no good in longrun
            galpars +=[GA[i]]
            galcov += [0.01]
            var = {'x':GX[i],'y':GY[i],'q':GQ[i],'pa':GP[i],'re':GR[i],'n':GN[i],'amp':GA[i]}
            gals.append(SBModels.Sersic('galaxy%i'%i,var))
        else: 
            var = {'x':GX[i],'y':GY[i],'q':GQ[i],'pa':GP[i],'re':GR[i],'n':GN[i],'amp':1}
            gals.append(SBModels.Sersic('galaxy%i'%i,var))

    return gals,galpars,galcov

def buildSource(x0,y0,components=1,gaussian=False,spherical=False,startpars=None,fixamp=False):
    from imageSim import SBModels
    SX={}
    SY={}
    SR={}
    SQ={}
    SP={}
    SN={}
    SA={}
    gals=[]
    for i in range(components):
        SX[i] = pymc.Uniform('SX%i'%i,x0-25.,x0+25.,value=x0)
        SY[i] = pymc.Uniform('SY%i'%i,y0-25.,y0+25.,value=y0)
        SP[i] = pymc.Uniform('SP%i'%i,-180.,180.,value=-23)
        if i==0:
            galpars=  [SX[i],SY[i],SP[i]]
            galcov =  list(numpy.array([0.1,0.1,1.])/1e0)
        else:
            galpars+= [SX[i],SY[i],SP[i]]
            galcov += list(numpy.array([0.05,0.05,1.])/1e0)
        SQ[i] = pymc.Uniform('SQ%i'%i,0.2,1,value=0.79)
        SR[i] = pymc.Uniform('SR%i'%i,0.1,100.,value=10)
        SN[i] = pymc.Uniform('SN%i'%i,0.25,8.,value=2.34)
        galpars +=[SQ[i],SR[i],SN[i]]
        galcov += list(numpy.array([0.03,0.03,0.07])/1e0)
        if fixamp!=True:
            SA[i] = pymc.Uniform('SA%i'%i,0,100.,value=0.1)#no good in longrun
            galpars +=[SA[i]]
            galcov += [0.01]
            var = {'x':SX[i],'y':SY[i],'q':SQ[i],'pa':SP[i],'re':SR[i],'n':SN[i],'amp':SA[i]}
            gals.append(SBModels.Sersic('galaxy%i'%i,var))
        else: 
            var = {'x':SX[i],'y':SY[i],'q':SQ[i],'pa':SP[i],'re':SR[i],'n':SN[i],'amp':1}
            gals.append(SBModels.Sersic('galaxy%i'%i,var))

    return gals,galpars,galcov
