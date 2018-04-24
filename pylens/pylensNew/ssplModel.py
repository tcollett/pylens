def ssplModelParametric(lenses,gals,srcs,fullimage,fullsig,imshape,mask,xc,yc,P,fullP,csub=11,psf=None,noResid=False,outputimages=False,interactive=False,name=None):
    import numpy
    from imageSim import convolve
    import pylens
    from scipy import optimize

    maskA=mask.flatten()

    #this part of the code assumes the galaxy amplitudes are non-linear parameters,
    #however it's possible to solve for them linearly like the source parameters
    #but since you can't do this for the pixellated modelling, I've done it like this
    #for consistancy
    galsubimage = fullimage*1.
    for gal in gals:
        gal.setPars()
        galsubimage -= fullP*gal.pixeval(xc,yc,csub=csub)
    galsum=(fullimage-galsubimage)


    #Get the defleaction angles
    for lens in lenses:
        lens.setPars()
    xl,yl = pylens.getDeflections(lenses,[xc[maskA],yc[maskA]])


    #for the matrix that relates the data to the srcs
    # D = M v, where v is the amplitude of each src
    model=numpy.zeros((len(srcs),xl.size))
    for i in range(len(srcs)):
        src=srcs[i]
        src.setPars()
        tmp=src.pixeval(xl,yl,csub=csub)
        if numpy.isnan(tmp).any():
            return -1e300
        model[i]=P*tmp

    #now minimize the chisquared
    rhs = (galsubimage[maskA]/fullsig[maskA])
    op = (model/fullsig[maskA]).T
    fit,chi = optimize.nnls(op,rhs)

    #fit is the amplitude, so now we know the model
    model=(fit*model.T).T
    model=model.sum(0)
    
    fullmodel=galsum
    fullmodel[maskA]+=model

    resid = (fullmodel-fullimage)/fullsig

    if interactive:
        #save the fit, display it etc
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



def ssplModelPixellated(lenses,gals,sshape,fullimage,fullsig,imshape,cmat,mask,xc,yc,P,rmat,fullP,regconst,csub=11,psf=None,noResid=False,mapping='bilinear',outputimages=False,returnRegCoeffs=False,regmin=0.1,regacc=0.005,fixreg=False,interactive=False,removeedges=True,name=None):
    import pixellatedTools as pT
    import numpy
    from imageSim import convolve
    from scipy.sparse import diags
    import pylens

    maskA=mask.flatten()

    #first subtract the galaxy light
    galsubimage = fullimage*1.
    for gal in gals:
        gal.setPars()
        galsubimage -= fullP*gal.pixeval(xc,yc,csub=csub)
    galsum=(fullimage-galsubimage)

    galsubimage=galsubimage[maskA]
    sigA=fullsig[maskA]

    for lens in lenses:
        lens.setPars()
    xl,yl = pylens.getDeflections(lenses,[xc[maskA],yc[maskA]])

    sxax,syax,sx,sy=pT.newAxes(xl,yl,sshape[0],sshape[1])

    lm,isin=pT.getMatBilinear(xl,yl,sx,sy,sxax,syax)
    mat = P*lm

    model,fit,reg,Bmat = pT.getModel(galsubimage,sigA**2,mat,cmat,rmat,regconst,niter=30,fixreg=fixreg,removeedges=removeedges,isin=isin,regmin=regmin)


    Es=0.5*fit.dot(rmat[isin][:,isin]*fit)

    fullmodel=galsum*1
    fullmodel[maskA]+=model
    res=((fullimage-fullmodel)/fullsig)

    M=-(0.5*(res**2).sum()+reg*Es)

    if outputimages:
        import pyfits
        empty=fullimage.copy()*0
        modelarc=Bmat*fit
        if removeedges:
            src=numpy.zeros(sshape[0]*sshape[1])
            src[isin]=fit
            src=src.reshape(sshape)
        else:
            src=fit.copy()
            src=src.reshape(sshape)

        res=((fullimage-fullmodel)/fullsig).reshape(imshape)
        fullmodel=fullmodel.reshape(imshape)

        pyfits.PrimaryHDU(fullmodel).writeto("%smodel.fits"%name,clobber=True)
        pyfits.PrimaryHDU(res).writeto("%sresid.fits"%name,clobber=True)
        pyfits.PrimaryHDU(src).writeto("%ssrc.fits"%name,clobber=True)
        if interactive:
            import pylab as plt
            plt.figure()
            plt.imshow((fullmodel),origin='lower',interpolation="none")

            plt.figure()
            plt.imshow((res),origin='lower',interpolation="none")

            plt.figure()
            plt.imshow((src),origin='lower',interpolation="none",extent=[sx.min(),sx.max(),sy.min(),sy.max()])

            plt.show(block=True)

    return M,reg
