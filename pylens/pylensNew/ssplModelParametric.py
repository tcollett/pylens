def ssplModelParametric(lenses,gals,srcs,fullimage,fullsig,imshape,mask,xc,yc,P,fullP,csub=11,psf=None,noResid=False,outputimages=False,interactive=False,name=None):
    import pixellatedTools
    import pixellatedTools as pT
    import pixellatedToolsMatt2 as pTM2
    import numpy
    from imageSim import convolve
    from scipy.sparse import diags
    import pylens
    from scipy import optimize

    maskA=mask.flatten()

    if len(gals)==0:
        galsubimage = fullimage*1.
        for gal in gals:
            gal.setPars()
            galsubimage -= fullP*gal.pixeval(xc,yc,csub=csub)
        galsum=(fullimage-galsubimage)
        sigA=fullsig[maskA]
        for lens in lenses:
            lens.setPars()
        xl,yl = pylens.getDeflections(lenses,[xc[maskA],yc[maskA]])
        xlout,ylout=None,None
    #now for the sources and the inversion:
        model=numpy.zeros((len(srcs),xl.size))
        import time
        for i in range(len(srcs)):
      #t0=time.clock()
      #for j in range(100):
            src=srcs[i]
            src.setPars()
            tmp=src.pixeval(xl,yl,csub=csub)
            if numpy.isnan(tmp).any():
                return -1e300

            model[i]=P*tmp
      #print time.clock()-t0,i

        rhs = (galsubimage[maskA]/fullsig[maskA])
        op = (model/fullsig[maskA]).T
        
        fit,chi = optimize.nnls(op,rhs)
        model=(fit*model.T).T
        model=model.sum(0)
        
        fullmodel=galsum.flatten()
        fullmodel[maskA]+=model

    elif len(gals)!=0:
        galsubimage = fullimage*1.
        for gal in gals:
            gal.setPars()
            galsubimage -= fullP*gal.pixeval(xc,yc,csub=csub)
        galsum=(fullimage-galsubimage)


        for lens in lenses:
            lens.setPars()
        xl,yl = pylens.getDeflections(lenses,[xc[maskA],yc[maskA]])
        xlout,ylout=None,None
    #now for the sources and the inversion:
        model=numpy.zeros((len(srcs),xl.size))
        import time
        for i in range(len(srcs)):
            src=srcs[i]
            src.setPars()
            tmp=src.pixeval(xl,yl,csub=csub)
            if numpy.isnan(tmp).any():
                return -1e300

            model[i]=P*tmp

        rhs = (galsubimage[maskA]/fullsig[maskA])
        op = (model/fullsig[maskA]).T
        
        fit,chi = optimize.nnls(op,rhs)
        model=(fit*model.T).T
        model=model.sum(0)
        
        fullmodel=galsum
        fullmodel[maskA]+=model




    resid = (fullmodel-fullimage)/fullsig
 
    print resid.sum(),resid[maskA].mean(),resid[maskA].std()


    if interactive:
        import pyfits
        pyfits.PrimaryHDU(fullmodel).writeto("%smodelParametric.fits"%name,clobber=True)
        pyfits.PrimaryHDU(resid).writeto("%sresidParametric.fits"%name,clobber=True)
        import pylab as plt
        plt.imshow((fullmodel.reshape(imshape)),origin='lower',interpolation="none")
        plt.figure()
        plt.imshow((resid.reshape(imshape)),origin='lower',interpolation="none")
 
        plt.figure()
        sxax,syax,sx,sy=pT.newAxes(xl,yl,100,100)
        tmp=numpy.zeros((100*100))
        for i in range(len(srcs)):
            src=srcs[i]
            src.setPars()
            tmp+=fit[i]*src.pixeval(sx,sy,csub=csub)
        tmp=tmp.reshape(100,100)
        plt.imshow((tmp),origin='lower',interpolation="none",extent=[sx.min(),sx.max(),sy.min(),sy.max()])
        plt.show(block=True)
        pyfits.PrimaryHDU(tmp).writeto("%ssrcParametric.fits"%name,clobber=True)

    return -0.5*(resid**2).sum()
