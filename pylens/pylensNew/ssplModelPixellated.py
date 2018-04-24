def ssplModelPixellated(lenses,gals,sshape,fullimage,fullsig,imshape,cmat,mask,xc,yc,P,rmat,fullP,regconst,csub=11,psf=None,noResid=False,mapping='bilinear',outputimages=False,returnRegCoeffs=False,regmin=0.1,regacc=0.005,fixreg=False,interactive=False,removeedges=False,galonly=False,forceoutofmasktobezero=False,name=None):
    import pixellatedTools
    import pixellatedTools as pT
    import pixellatedToolsMatt2 as pTM2
    import numpy
    from imageSim import convolve
    from scipy.sparse import diags
    import pylens

    maskA=mask.flatten()

    galsubimage = fullimage*1.
    for gal in gals:
        gal.setPars()
        galsubimage -= fullP*gal.pixeval(xc,yc,csub=csub)
    galsum=(fullimage-galsubimage)
    if galonly: 
        res=((galsubimage)/fullsig)
        M=-(0.5*(res**2).sum())
        return M,regconst


    galsubimage=galsubimage[maskA]
    sigA=fullsig[maskA]


    for lens in lenses:
        lens.setPars()
    if forceoutofmasktobezero:
        #really a  better second mask could be used here to make computations
        #faster. I'm currently just not taking the central pixels that are
        #highly demagnified 
        xlall,ylall = pylens.getDeflections(lenses,[xc,yc])

        innerreg=((xc-lenses[0].x)**2+(yc-lenses[0].y)**2<(0.05*lenses[0].b)**2)
        xlall[innerreg],ylall[innerreg]=-999,-999

        xl,yl=xlall[maskA],ylall[maskA]
        xlout,ylout=xlall[maskA==False],ylall[maskA==False]
        



    else:
        xl,yl = pylens.getDeflections(lenses,[xc[maskA],yc[maskA]])
        xlout,ylout=None,None

    sxax,syax,sx,sy=pT.newAxes(xl,yl,sshape[0],sshape[1])


    lm,isin=pTM2.getMatBilinear(xl,yl,sx,sy,sxax,syax,forceoutofmasktobezero=forceoutofmasktobezero,xout=xlout,yout=ylout)
    mat = P*lm

    model,fit,reg,Bmat = pTM2.getModel(galsubimage,sigA**2,mat,cmat,rmat,regconst,niter=30,fixreg=fixreg,isin=isin,regmin=regmin,removeedges=removeedges)


    if removeedges:
        Es=0.5*fit.dot(rmat[isin][:,isin]*fit)
    else:
        Es=0.5*fit.dot(rmat*fit)


    fullmodel=galsum*1
    fullmodel[maskA]+=model
    res=((fullimage-fullmodel)/fullsig)
    #print res[maskA].mean(),res[maskA].std()

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

        #empty[maskA]=modelarc
        res=((fullimage-fullmodel)/fullsig).reshape(imshape)
        fullmodel=fullmodel.reshape(imshape)


        pyfits.PrimaryHDU(fullmodel).writeto("%smodelRing.fits"%name,clobber=True)
        pyfits.PrimaryHDU(res).writeto("%sresidRing.fits"%name,clobber=True)
        pyfits.PrimaryHDU(src).writeto("%ssrcRing.fits"%name,clobber=True)
        if interactive:
            import pylab as plt
            #plt.imshow((fullimage.reshape(imshape)),origin='lower',interpolation="none")
            #plt.figure()
            #plt.imshow((galsum.reshape(imshape)),origin='lower',interpolation="none")

            plt.figure()
            plt.imshow((fullmodel),origin='lower',interpolation="none")

            #plt.figure()
            #plt.imshow((modelarc),origin='lower',interpolation="none")

            plt.figure()
            plt.imshow((res),origin='lower',interpolation="none")

            plt.figure()
            plt.imshow((src),origin='lower',interpolation="none",extent=[sx.min(),sx.max(),sy.min(),sy.max()])

            #plt.scatter(xl,yl,edgecolor=None,facecolor='r',s=3)

            #outmask=((xlout>sxax.min()) & (xlout<sxax.max()) & (ylout>syax.min()) & (ylout<syax.max()))
            #plt.scatter(xlout[outmask],ylout[outmask],edgecolor=None,facecolor='k',s=1)
            #plt.xlim([sx.min(),sx.max()])
            #plt.ylim([sy.min(),sy.max()])

            plt.show(block=True)



    return M,reg
