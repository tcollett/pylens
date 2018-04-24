import numpy

def getPSFMatrix(psf,imshape,mask=None):
    """
    Create a PSF matrix given the PSF model and image dimensions
    """
    import numpy
    from scipy.sparse import coo_matrix
    import indexTricks as iT


    imsize = imshape[0]*imshape[1]

    y,x = iT.coords(psf.shape)
    x -= int(x.mean())
    y -= int(y.mean())
    c = psf.ravel()!=0
    x = x.ravel().astype(numpy.int32)[c]
    y = y.ravel().astype(numpy.int32)[c]
    Y,X = iT.coords(imshape)
    X = X.ravel().astype(numpy.int32)
    Y = Y.ravel().astype(numpy.int32)

    cols = X.repeat(x.size)+numpy.tile(x,X.size)
    rows = Y.repeat(y.size)+numpy.tile(y,Y.size)

    C = (cols>=0)&(cols<imshape[1])&(rows>=0)&(rows<imshape[0])
    cols = cols[C]
    rows = rows[C]

    pvals = numpy.tile(psf.ravel()[c],imsize)[C]
    col = cols+rows*imshape[1]
    row = numpy.arange(imsize).repeat(c.sum())[C]


    cols=cols[pvals!=0]
    rows=rows[pvals!=0]
    pvals=pvals[pvals!=0]


    pmat = coo_matrix((pvals,(col,row)),shape=(imsize,imsize))
    if mask is not None:
        npnts = mask.sum()
        c = numpy.arange(imsize)[mask.ravel()]
        r = numpy.arange(npnts)
        smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
        pmat = smat.T*(pmat*smat)
    return pmat

def maskPSFMatrix(pmat,mask):
    import numpy
    from scipy.sparse import coo_matrix

    imsize = pmat.shape[0]
    npnts = mask.sum()
    c = numpy.arange(imsize)[mask.ravel()]
    r = numpy.arange(npnts)
    smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
    return smat.T*(pmat*smat)

def trimandmaskPSFMatrix(PSF,imshape,mask,size=None):
    import indexTricks as iT
    if size !=None:
        prad=size+0.0001
        pc=(PSF.shape[0]-1)/2
        PSF=PSF[pc-prad:pc+prad,pc-prad:pc+prad]
        pc=(PSF.shape[0]-1)/2
        l,m=iT.overSample(PSF.shape,1)
        l-=pc
        m-=pc
        l=l.flatten()
        m=m.flatten()
        f = ((l**2+m**2)<prad**2)
        f=f.reshape(PSF.shape)
        pmask = numpy.zeros(PSF.shape)
        pmask[f]=1
        pmask=pmask.reshape(PSF.shape)
        PSF[pmask!=1]=0        
    psf =PSF/ PSF.sum()
    psf /= psf.sum()
    Pfull = getPSFMatrix(psf,imshape)
    Pmasked = maskPSFMatrix(Pfull,mask)
    return Pmasked,Pfull


def getRegularizationMatrix(srcxaxis,srcyaxis,mode="new",kernel="c"):
    from scipy.sparse import diags,csc_matrix,lil_matrix
    import numpy
    #there are many plausible forms of regularization, but curvature
    #seems to be the default option in the literature. Use mode="new",
    # and kernel="c" to get this option.
    srcshape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)
    if mode[0:3]=="new":
        if kernel==None:
            kernel="c"
        elif kernel=="z":
            kernel=numpy.array([[1]])
        elif kernel=="g":
            kernel=numpy.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        elif kernel=="gc":
            kernel=numpy.array([[-1,-2,-1],[-2,12,-2],[-1,-2,-1]])
        elif kernel=="c":
            kernel=numpy.array([[ 0, 0, 1, 0, 0],
                                [ 0, 0,-4, 0, 0],
                                [ 1,-4,12,-4, 1],
                                [ 0, 0,-4, 0, 0],
                                [ 0, 0, 1, 0, 0]])
        elif kernel=="cc":
            kernel=2*numpy.array([[ 0, 0, 1, 0, 0],
                                  [ 0, 0,-4, 0, 0],
                                  [ 1,-4,12,-4, 1],
                                  [ 0, 0,-4, 0, 0],
                                  [ 0, 0, 1, 0, 0]])
            kernel+=numpy.array([[ 1, 0, 0, 0, 1],
                                 [ 0,-4, 0,-4, 0],
                                 [ 0, 0,12, 0, 0],
                                 [ 0,-4, 0,-4, 0],
                                 [ 1, 0, 0, 0, 1]])


        stringerr="regularization kernel not square symmetric"
        assert kernel.shape[0]==kernel.shape[1],stringerr
        assert numpy.sum(numpy.flipud(kernel)-kernel)==0,stringerr
        assert numpy.sum(numpy.fliplr(kernel)-kernel)==0,stringerr
        kc=(kernel.shape[0]-1)/2
        if kernel[kc,kc]<0:kernel=-kernel

        vals=[]
        poss=[]
        for i in range(kernel.shape[0]):
          for j in range(kernel.shape[1]):
            vals.append(kernel[i,j])
            poss.append((j-kc)+(i-kc)*srcyaxis.size)
            
        mat=diags(vals,poss,shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)) 
        mat=lil_matrix(mat)

        #now hammer out the glitches
        #(points where the regularization kernel overlaps the edges of the grid)
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        for i in range(kc):
            le=allcols[allcols%srcxaxis.size==i]
            re=allcols[allcols%srcxaxis.size==srcxaxis.size-1-i]
            for el in le:
                mat[el,el-1]=0
            for el in re:
                if el != allcols.max()-i:
                    mat[el,el+1]=0

    if mode=="zeroth":
        from scipy.sparse import identity
        return identity(srcxaxis.size*srcyaxis.size).tocsc()


    if mode=="gradient":
        mat = diags([-2,-2,8,-2,-2],[-srcxaxis.size,-1,0,1,srcxaxis.size],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size))
        mat=lil_matrix(mat)

        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        for el in leftedges:
            mat[el,el-1]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0

    elif mode=="curvatureOLD":
        mat=diags([2,2,-8,-8,24,-8,-8,2,2],[-2*srcxaxis.size,-2,-srcxaxis.size,-1,0,1,srcxaxis.size,2*srcxaxis.size,2],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)) 
        mat=lil_matrix(mat)
        
        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        leftedgesinone=allcols[allcols%srcxaxis.size==1]
        rightedgesinone=allcols[allcols%srcxaxis.size==srcxaxis.size-2]

        for el in leftedges:
            mat[el,el-1]=0
            mat[el,el-2]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0
                mat[el,el+2]=0
        for el in leftedgesinone:
            mat[el,el-2]=0
        for el in rightedgesinone:
            if el != allcols.max()-1:
                mat[el,el+2]=0

    elif mode=="curvature":
        I,J=srcxaxis.size,srcyaxis.size
        matrix=lil_matrix((I*J,I*J))
        for i in range(I-2):
            for j in range(J):
                ij=i+j*J
                i1j=ij+1
                i2j=ij+2
                matrix[ij,ij]+=1.
                matrix[i1j,i1j]+=4
                matrix[i2j,i2j]+=1
                matrix[ij,i2j]+=1
                matrix[i2j,ij]+=1
                matrix[ij,i1j]-=2
                matrix[i1j,ij]-=2
                matrix[i1j,i2j]-=2
                matrix[i2j,i1j]-=2
        for i in range(I):
            for j in range(J-2):
                ij=i+j*J
                ij1=ij+J
                ij2=ij+2*J
                matrix[ij,ij]+=1
                matrix[ij1,ij1]+=4
                matrix[ij2,ij2]+=1
                matrix[ij,ij2]+=1
                matrix[ij2,ij]+=1
                matrix[ij,ij1]-=2
                matrix[ij1,ij]-=2
                matrix[ij1,ij2]-=2
                matrix[ij2,ij1]-=2
        for i in range(I):
            iJ_1=i+(J-2)*J
            iJ=i+(J-1)*J
            matrix[iJ_1,iJ_1]+=1
            matrix[iJ,iJ]+=1
            matrix[iJ,iJ_1]-=1
            matrix[iJ_1,iJ]-=1
        for j in range(J):
            I_1j=(I-2)+j*J
            Ij=(I-1)+j*J
            matrix[I_1j,I_1j]+=1
            matrix[Ij,Ij]+=1
            matrix[Ij,I_1j]-=1
            matrix[I_1j,Ij]-=1
        for i in range(I):
            iJ=i+(J-1)*J
            matrix[iJ,iJ]+=1
        for j in range(J):
            Ij=(I-1)+j*J
            matrix[Ij,Ij]+=1
        mat=matrix

    return mat.tocsc()


class RegConst:
    def __init__(self,name,value):
        self.value = value
        self.logp=0
        self.__name__ = name



#==============================================================


def newAxes(x,y,srcxsize,srcysize):
    import numpy
    import indexTricks as iT

    xdif=x.max()-x.min()
    ydif=y.max()-y.min()

    difmax =numpy.max([xdif,ydif])

    scentre= ((x.max()+x.min())/2.,(y.max()+y.min())/2.)
    sshape=((srcysize,srcxsize))

    pscale=1.2*(difmax/((srcysize-3)))

    srcy,srcx = iT.coords(sshape)*pscale
    srcx -= srcx.mean()
    srcy -= srcy.mean()
    srcy += scentre[1]
    srcx += scentre[0]

    srcxaxis=srcx[0]
    srcyaxis=srcy[:,0]

    srcx = srcx.flatten()
    srcy = srcy.flatten()

    return srcxaxis,srcyaxis,srcx,srcy


def PixelNumber(x0,y0,xsrcaxes,ysrcaxes,mode='NearestCentre'):
    import numpy
    srcpixscale=xsrcaxes[1]-xsrcaxes[0]
    if mode=='NearestCentre':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)+0.5))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)+0.5))

    if mode=='NearestBottomLeft':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)))

    pixelnumber=ypixelnumber*len(xsrcaxes)+xpixelnumber

    pixelnumber[xpixelnumber<0]=-1
    pixelnumber[ypixelnumber<0]=-1
    pixelnumber[xpixelnumber>=len(xsrcaxes)]=-1
    pixelnumber[ypixelnumber>=len(ysrcaxes)]=-1

    if mode=='NearestBottomLeft': 
        pixelnumber[xpixelnumber==len(xsrcaxes)-1]=-1
        pixelnumber[ypixelnumber==len(ysrcaxes)-1]=-1

    return pixelnumber



def getMatBilinear(xin,yin,srcx,srcy,srcxaxis,srcyaxis,mask=None,forceoutofmasktobezero=False,xout=None,yout=None):
    #theres a better way of doing things now, but I want backward compatibility
    if forceoutofmasktobezero:
        return getMatBilinear2(xin,yin,srcx,srcy,srcxaxis,srcyaxis,mask=None,forceoutofmasktobezero=False,xout=xout,yout=yout)

    import numpy
    from scipy.sparse import coo_matrix
    if mask is None:
        mask = xin==xin

    size = xin.size
    scale = srcxaxis[1]-srcxaxis[0]
    spix = PixelNumber(xin,yin,srcxaxis,srcyaxis,mode='NearestBottomLeft')
    c = (spix>-1)&mask
    row = numpy.arange(spix.size)[c]
    xin,yin = xin[c],yin[c]
    spix = spix[c].astype(numpy.int32)

    # Create row, col, value arrays
    r = numpy.empty(row.size*4)
    c = numpy.empty(row.size*4)
    w = numpy.empty(row.size*4)

    # These are the lower-right pixel weights
    r[:row.size] = row
    c[:row.size] = spix
    w[:row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)

    # Now do the lower-left, upper-left, and upper-right pixels
    a = [1,srcxaxis.size,-1]
    for i in range(1,4):
        spix += a[i-1]
        r[i*row.size:(i+1)*row.size] = row
        c[i*row.size:(i+1)*row.size] = spix
        w[i*row.size:(i+1)*row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)


    nx=len(srcxaxis)
    ny=len(srcyaxis)
    inA=numpy.unique(c).astype(numpy.int32, copy=False)
    include=numpy.zeros(nx*ny)
    include[inA]=1
    include=include.reshape((nx,ny))

    for k in range(1):#iron out any wierd edge effects
        wh=numpy.where(include>0)
        betweenxlims=include*0
        for i in range(nx):
            whi=wh[1][wh[0]==i]
            if len(whi)>0:
                betweenxlims[i,whi.min()-1:whi.max()+2]=1       

        betweenylims=include*0
        for i in range(ny):
            whi=wh[0][wh[1]==i]
            if len(whi)>0:
                betweenylims[whi.min()-1:whi.max()+2,i]=1

        include=betweenylims*betweenxlims

    """
    #this code prettifies images but is really slow.
    for k in range(10):
     for i in range(nx):
      for j in range(ny):
        #print include[i,j]
        if include[i,j]==0:
         try: 
            if  include[i,j]+include[i,j-1]+include[i,j+1]+include[i+1,j]+include[i-1,j]>2: include[i,j]+=1
            if include[i,j-1]==1 and include[i,j+1]==1:
                include[i,j]=1
            if include[i-1,j]==1 and include[i+1,j]==1:
                include[i,j]=1
            if include[i,j-2]==1 and include[i,j+1]==1:
                include[i,j]=1
            if include[i,j-1]==1 and include[i,j+2]==1:
                include[i,j]=1
            if include[i-2,j]==1 and include[i+1,j]==1:
                include[i,j]=1
            if include[i-1,j-1]==1 and include[i+2,j]==1:
                include[i,j]=1
            if include[i,j-2]==1 and include[i,j+2]==1:
                include[i,j]=1
            if include[i-2,j]==1 and include[i+2,j]==1:
                include[i,j]=1
         except: continue
    """     
    include=include.ravel()

    isin=numpy.where(include==1)[0]
    return coo_matrix((w,(r,c)),shape=(size,srcx.size)),isin


def getModel(img,var,omat,cmat,rmat,reg,niter=10,fixreg=False,isin=None,removeedges=True,regmin=1e-5,regacc=0.001):
    from scikits.sparse.cholmod import cholesky
    from scikits.sparse.cholmod import CholmodError
    import numpy

    if ((removeedges) & (isin!=None)):
        omat=omat.tocsr()[:,isin]
        rmat=rmat[isin][:,isin]

    rhs = omat.T*(img/var)

    B = omat.T*cmat*omat

    res = 0.
    if fixreg==False:
        regs=[reg]
    else:
        regs = [fixreg]

    lhs = B+regs[-1]*rmat

    F = cholesky(lhs)
    fit = F(rhs)

    if fixreg!=False:
        finalreg=fixreg
    else:
        finalreg=solveSPregularization(B,rmat,reg,rhs,niter=niter,regmin=regmin,regacc=regacc)
        
    lhs = B+finalreg*rmat
    F = cholesky(lhs)
    fit = F(rhs)

    model = omat*fit


    return model,fit,finalreg,B




def solveSPregularization(B,rmat,reg,rhs,fixregI=False,fixregO=False,niter=10,regmin=1e-6,regacc=0.005):
    from scikits.sparse.cholmod import cholesky
    from scikits.sparse.cholmod import CholmodError
    from scipy.sparse import block_diag
    import numpy
    alpha=1e-5

    Ns=B.shape[0]

    Ranger=2
    regs=[reg]
    for i in range(niter):
      for j in range(Ranger): 
        lhs = (B+regs[-1]*rmat).tocsc()
        F = cholesky(lhs)
        fit = F(rhs)

        delta = regs[-1]*alpha
        lhs2 = (B+(regs[-1]+delta)*rmat).tocsc()
        logdetlhs2=numpy.log(F.cholesky(lhs2).L().diagonal()).sum()

        T = (2./(delta))*(logdetlhs2-numpy.log(F.L().diagonal()).sum())

        Es= 0.5*fit.dot(rmat*fit)

        newreg=Ns/(2*Es+T)

        Es=0.5*fit.dot(rmat*fit)        

        newreg=Ns/(2*Es+T)

        regs.append(newreg)

        if i>1:
            check=checkconvergence(regs,regmin,10000,regacc)
            if check!= "continue": 
                return check
      
      try:regs.append(Aitken(regs,regmin,10000))
      except FloatingPointError: pass

      if i > 5:
          Ranger+=1
          
      #attempt breakout hack incase stuck in a loop
      if i == 10:
          regs[-1]=regs[-1]*2.

    #eventually give up.
    return 1e10

def Aitken(regs,regmin,regmax):
    if regs[-1]<regmin:return regmin
    elif regs[-1]>regmax:return regmax
    else:
        x0=regs[-3]
        x1=regs[-2]
        x2=regs[-1]
        denominator = x2 - 2*x1 + x0
        aitkenX = x2 - ( (x2 - x1)**2 )/ denominator
        
        if (aitkenX<regmax) &(aitkenX>regmin):
            return aitkenX
        else:
            return x2

def checkconvergence(regs,regmin,regmax,regacc,wtfflag=False):
    import numpy

    #print regs[-1]
    if len(regs)<2:
        return "continue"
    if regs[-1]<regmin:
        #print regmin
        return regmin
    if regs[-1]>regmax:
        #print "regmax"
        return regmax
    if numpy.abs((regs[-1]-regs[-2])/(regs[-1]))<regacc:
        return regs[-1]

    if wtfflag:
        print numpy.abs((regs[-1]-regs[-2])/(regs[-1])),regacc


    return "continue"
