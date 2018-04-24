#!/usr/bin/env python
# ======================================================================

# Globally useful modules:

import matplotlib
# Force matplotlib to not use any Xwindows backend:
matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

import string,numpy,pylab,sys,getopt

# ======================================================================

def CornerPlotter(argv):

  USAGE = """
  NAME
    CornerPlotter.py

  PURPOSE
    Plot 2D projections of 5D parameter space as triangular 
    array of panels. Include 1D marginalised distributions
    on the diagonal

  COMMENTS
  
  Expected data format is plain text, header marked by # in column 1, 
  with header line listing parameter names separated by commas. These names
  will be used as labels on the plots

  USAGE
    CornerPlotter.py [flags]

  FLAGS
          -u           Print this message [0]
          --eps        Postscript output

  INPUTS
         file1,color1  Name of textfile containing data, and color to use
         file2,color2   etc.
         
  OPTIONAL INPUTS
         -w iw         Index of column containing weight of sample [NOT OPERATIONAL]
         -L iL         Index of column containing likelihood of sample [NOT OPERATIONAL]
         --plot-points Plot the samples themselves
 
  OUTPUTS
          stdout       Useful information
          pngfile      Output plot in png format

  EXAMPLES
  
    CornerPlotter.py -w 1 -L 2 --plot-points J2141-disk_bulge.txt,red

  BUGS
    - Tick labels overlap, cannot remove first and last tick mark
    - Figure has no legend
    - no overlay of 1-D priors

  HISTORY
    2010-05-06 started Marshall/Auger (UCSB)
    2011-06-24 generalized Marshall (Courmayeur/Bologna)
  """

  # --------------------------------------------------------------------

  try:
      opts, args = getopt.getopt(argv, "hvew:L:",["help","verbose","eps","plot-points"])
  except getopt.GetoptError, err:
      # print help information and exit:
      print str(err) # will print something like "option -a not recognized"
      print USAGE
      return

  vb = False
  wcol = -1
  Lcol = -1
  plotpoints = False
  eps = False
  # NB. wcol and Lcol are assumed to be entered indexed to 1! 
  for o,a in opts:
      if o in ("-v", "--verbose"):
          vb = True
      elif o in ("--plot-points"):
          plotpoints = True
      elif o in ("-w"):
          wcol = int(a) - 1
      elif o in ("-L"):
          Lcol = int(a) - 1
      elif o in ("--eps"):
          eps = True
      elif o in ("-h", "--help"):
          print USAGE
          return
      else:
          assert False, "unhandled option"
   
  # Check for datafiles in array args:  
  if len(args) > 0:
    datafiles = []
    colors = []
    for i in range(len(args)):
      bits = args[i].split(',')
      if len(bits) != 2:
        print "ERROR: supply input data as 'filename,color'"
        exit()
      datafiles = datafiles + [bits[0]]
      colors = colors + [bits[1]]
    if vb: 
      print "Making corner plot of data in following files:",datafiles
      print "using following colors:",colors
      if eps: "Output will be postscript"
  else :
    print USAGE
    return 
    
  # --------------------------------------------------------------------

  # Start figure, set up viewing area:
  figprops = dict(figsize=(8.0, 8.0), dpi=128)                                          # Figure properties
  fig = pylab.figure(**figprops)

  # Need small space between subplots to avoid deletion due to overlap...
  adjustprops = dict(\
    left=0.1,\
    bottom=0.1,\
    right=0.95,\
    top=0.95,\
    wspace=0.04,\
    hspace=0.08)
  fig.subplots_adjust(**adjustprops)

  # Font sizes:
  params = { 'axes.labelsize': 10,
              'text.fontsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8}
  pylab.rcParams.update(params)
  
  # No. of bins used:
  nbins = 81
  # Small tweak to prevent too many numeric tick marks
  tiny = 0.01

  # --------------------------------------------------------------------

  # Plot files in turn, using specified color scheme.
  
  for k in range(len(datafiles)):  

    datafile = datafiles[k]
    color = colors[k]
    if (color == 'black' or eps):
      style = 'outlines'
    else:
      style = 'shaded'  
    legend = datafile
   
    print "************************************************************"
    print "Plotting PDFs given in",datafile

    # Read in data:
    data = numpy.loadtxt(datafile)
    
    # Start figuring out how many parameters we have - index will be a 
    # list of column numbers containg the parameter values.
    # NB. ALL files must have same parameters/weights structure!
    if ( k == 0 ):
      npars = data.shape[1]
      index = numpy.arange(npars)

    # Some cols may not be data, adjust index accordingly. 
    
    # Weights/importances:
    if wcol >= 0 and wcol < data.shape[1]:
      if vb: print "using weight in column: ",wcol
      wht = data[:,wcol].copy()
      if ( k == 0 ):
        keep = (index != wcol)
        index = index[keep]
        npars = npars - 1
        if vb: print " index = ",index
    else:
      if (wcol >= data.shape[1]):
        print "WARNING: only ",data.shape[1]," columns are available"
        print "Setting weights to 1"
      wht = 0.0*data[:,0].copy() + 1.0
    
    # Likelihood values:
    if Lcol >= 0:
      Lhood = data[:,Lcol].copy()
      if ( k == 0 ):
        keep = (index != Lcol)
        index = index[keep]
        npars = npars - 1
      if vb: print "using lhood in column: ",Lcol
    else:
      Lhood = 0.0*data[:,0].copy() + 1.0
        
    # Now parameter list is in index - whic is fixed for other datasets
    if vb: print "using data in",npars,"columns: ",index
             
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # 1st data file defines labels and parameter ranges:

    if k == 0:

      limits = numpy.zeros([npars,2])
      hmax = numpy.zeros([npars])
    
      # NB: file MUST be of correct format!
    
      # Read comma-separated string axis labels from line 1:
      file = open(datafile)
      labelsline = file.readline().strip()
      # Read comma-separated axis limits from line 2
      limitsline = file.readline().strip()
      file.close()
      
      lineafterhash = labelsline.split('#')
      if len(lineafterhash) != 2:
        print 'ERROR: first line of file is not #-marked, comma-separated list of labels'
        exit()
      else:
        labels = lineafterhash[1].split(',')             
        if vb: print "Plotting",npars," parameters: ",labels

      lineafterhash = limitsline.split('#')
      if len(lineafterhash) != 2:
        if vb: print 'No axis limits found, using 5-sigma ranges'
        usedylimits = 1
      else:
        limitstrings = lineafterhash[1].split(',')             
        nlimits = len(limitstrings)
        if (nlimits/2 != data.shape[1]):
          print 'ERROR: found ',nlimits,'axis limits for',data.shape[1],'columns:'
          print limitstrings
          exit()
        else:
          ii = 0
          for i in index:
            for j in range(2):
              l = 2*i + j
              limits[ii,j] = float(limitstrings[l])*1.0
            ii = ii + 1
          if vb: print "Plot limits: ",limits
        usedylimits = 0
    
    # OK, back to any datafile.
                               
    # Set up dynamic axis limits, and smoothing scales:
    dylimits = numpy.zeros([npars,2])
    smooth = numpy.zeros(npars)
    for i in range(npars):
      col = index[i]
      # Get data subarray, and measure its mean and stdev:
      d = data[:,col].copy()
      mean,stdev = meansd(d)
      if vb: print "col = ",col," mean,stdev = ",mean,stdev
      # Set smoothing scale for this parameter, in physical units:
      smooth[i] = 0.1*stdev
      # Cf Jullo et al 2007, who use a bin size given by
      #  w = 2*IQR/N^(1/3)  for N samples, interquartile range IQR
      # For a Gaussian, IQR is not too different from 2sigma. 4sigma/N^1/3?
      # Also need N to be the effective number of parameters - return 
      # form meansd as sum of weights!
      # Set 5 sigma limits:
      dylimits[i,0] = mean - 5*stdev
      dylimits[i,1] = mean + 5*stdev

    # Now set up bin arrays based on dynamic limits, 
    # and convert smoothing to pixels:
    bins = numpy.zeros([npars,nbins])
    for i in range(npars):
      col = index[i]
      bins[i] = numpy.linspace(dylimits[i,0],dylimits[i,1],nbins)
      smooth[i] = smooth[i]/((dylimits[i,1]-dylimits[i,0])/float(nbins))
      if vb: print "col = ",col," smooth = ",smooth[i]
      if vb: print "binning limits:",dylimits[i,0],dylimits[i,1]
                  
    if (k == 0):
      # Finalise limits, again at 1st datafile:
      if (usedylimits == 1): limits = dylimits 

      for i in range(npars):
        limits[i,0] = limits[i,0] + tiny*abs(limits[i,0])
        limits[i,1] = limits[i,1] - tiny*abs(limits[i,1])

    # Good - limits are set.

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # Loop over plotting panels. Arrangement is bottom left hand corner,
    # most important parameter as x in first column, (row=0,col=0) is
    # top left hand corner panel.

    # panel = (col,row):

    # 1=0,0  |  2=1,0  |  3=2,0  |  4=3,0  |  5=4,0
    # ---------------------------------------------
    # 6=0,1  |  7=1,1  |  8=2,1  |  9=3,1  | 10=4,1
    # ---------------------------------------------
    #11=0,2  | 12=1,2  | 13=2,2  | 14=3,2  | 15=4,2
    # ---------------------------------------------
    #16=0,3  | 17=1,3  | 18=2,3  | 19=3,3  | 20=4,3
    # ---------------------------------------------
    #21=0,4  | 22=1,4  | 23=2,4  | 24=3,4  | 25=4,4

    for i in range(npars):
      col = index[i]

      # Get data subarray:
      d1 = data[:,col].copy()
      if vb: print "Read in ",col,"th column of data: min,max = ",min(d1),max(d1)

      for j in range(i,npars):
        row = index[j]

        # Move to next subplot:
        panel = j*npars+i+1
        pylab.subplot(npars,npars,panel)

        if j==i:

          # Report some statistcs:
          if vb:
            pc16,median,pc84 = percentiles(d1)
            print "  Percentiles (16,50,84th) =",pc16,median,pc84
            errplus = pc84 - median
            errminus = median - pc16
            print "  -> ",labels[col]+" = $",median,"^{",errplus,"}_{",errminus,"}$"

          # Plot 1D PDF, defined in subroutine below
          if vb: print "Calling pdf1d for col = ",col
          dummy = pdf1d(d1,wht,bins[i],smooth[i],color)
          if k == 0: hmax[i] = dummy

          # Force axes to obey limits:
          pylab.axis([limits[i,0],limits[i,1],0.0,1.2*hmax[i]])
          # Adjust axes of 1D plots:
          ax = pylab.gca()
          # Turn off the y axis tick labels for all 1D panels:
          ax.yaxis.set_major_formatter(pylab.NullFormatter())
          # Turn off the x axis tick labels for all but the last 1D panel:
          if j<(npars-1):
            ax.xaxis.set_major_formatter(pylab.NullFormatter())
          pylab.xticks(rotation=45)  
          # Label x axis, only on the bottom panels:
          if j==npars-1:
            pylab.xlabel(labels[col])
            
        else:

          # Get 2nd data set:
          d2 = data[:,row].copy()
          if vb: print "Read in ",row,"th column of data: min,max = ",min(d2),max(d2)

          # Plot 2D PDF, defined in subroutine below
          if vb: print "Calling pdf2d for col,row = ",col,row
          fwhm = 0.5*(smooth[i]+smooth[j])
          pdf2d(d1,d2,wht,bins[i],bins[j],fwhm,color,style)
          
          # If we are just plotting one file, overlay samples:
          if (len(datafiles) == 1 and plotpoints): 
            pylab.plot(d1,d2,'ko',ms=0.1)
          
          # Force axes to obey limits:
          pylab.axis([limits[i,0],limits[i,1],limits[j,0],limits[j,1]])
          # Adjust axes of 2D plots:
          ax = pylab.gca()
          if i>0:
            # Turn off the y axis tick labels
            ax.yaxis.set_major_formatter(pylab.NullFormatter())
          if j<npars-1:
            # Turn off the x axis tick labels
            ax.xaxis.set_major_formatter(pylab.NullFormatter())
          # Rotate ticks so that axis labels don't overlap
          pylab.xticks(rotation=45)  
          # Label x axes, only on the bottom panels:
          if j==npars-1:
            pylab.xlabel(labels[col])
          if i==0 and j>0:
          # Label y axes in the left-hand panels
            pylab.ylabel(labels[row])

        if vb: print "Done subplot", panel,"= (", i, j,")"
        if vb: print "  - plotting",labels[col],"vs",labels[row]
        if vb: print "--------------------------------------------------"

      # endfor
    # endfor

  # endfor
  print "************************************************************"

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
  # Plot graph to file:
  if eps:
    outfile = "cornerplot.eps"
  else:
    outfile = "cornerplot.png"
  # endif  
  pylab.savefig(outfile,dpi=300)
  print "Figure saved to file:",outfile

  exit()

# ======================================================================
# Subroutine to plot 1D PDF as histogram

def pdf1d(d,imp,bins,smooth,color):

  pylab.xlim([bins[0],bins[-1]])

  # Bin the data in 1D, return bins and heights
  # samples,positions,ax = pylab.hist(d,bins,fc='white')
  samples,positions = numpy.histogram(d,weights=imp,bins=bins,range=[bins[0],bins[-1]])

  # Normalise:
  norm = sum(samples)
  curve = samples/norm

  # Plot the PDF   
  pylab.plot(positions[:-1],curve,drawstyle='steps-mid',color=color)

  # # Smooth the PDF:
  # H = ndimage.gaussian_filter(H,smooth)

  # print "1D histogram: min,max = ",curve.min(),curve.max()
  hmax = curve.max()
  
#   print "Plotted 1D histogram with following axes limits:"
#   print "  extent =",(bins[0],bins[-1])
  
  return hmax

# ======================================================================
# Subroutine to plot 2D PDF as contours

def pdf2d(ax,ay,imp,xbins,ybins,smooth,color,style):

  from scipy import ndimage

  pylab.xlim([xbins[0],xbins[-1]])
  pylab.ylim([ybins[0],ybins[-1]])

  # npts = int((ax.size/4)**0.5)
  H,x,y = pylab.histogram2d(ax,ay,weights=imp,bins=[xbins,ybins])

  # Smooth the PDF:
  H = ndimage.gaussian_filter(H,smooth)
  
  sortH = numpy.sort(H.flatten())
  cumH = sortH.cumsum()
  # 1, 2, 3-sigma, for the old school:
  lvl00 = 2*sortH.max()
  lvl68 = sortH[cumH>cumH.max()*0.32].min()
  lvl95 = sortH[cumH>cumH.max()*0.05].min()
  lvl99 = sortH[cumH>cumH.max()*0.003].min()

#   print "2D histogram: min,max = ",H.min(),H.max()
#   print "Contour levels: ",[lvl00,lvl68,lvl95,lvl99]

  if style == 'shaded':
  
    # Plot shaded areas first:
    pylab.contourf(H.T,[lvl99,lvl95],colors=color,alpha=0.1,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl95,lvl68],colors=color,alpha=0.4,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl68,lvl00],colors=color,alpha=0.7,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
  # endif
  
  # Always plot outlines:  
  pylab.contour(H.T,[lvl68,lvl95,lvl99],colors=color,\
                  extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))



# ======================================================================
# Subroutine to return mean and stdev of numpy.array x

def meansd(x):

  N = len(x)
  mean = numpy.mean(x)
  stdev = numpy.sqrt(numpy.var(x)*float(N)/float(N-1))

  return mean,stdev

# ======================================================================
# Subroutine to return median and percentiles of numpy.array x

def percentiles(x):

  N = len(x)
  mark = numpy.array([int(0.16*N),int(0.50*N),int(0.84*N)],dtype=int)
  
  xx = x[numpy.argsort(x)]
  pc16 = xx[mark[0]]
  median = xx[mark[1]]
  pc84 = xx[mark[2]]

  return pc16,median,pc84

# ======================================================================

if __name__ == '__main__':
  CornerPlotter(sys.argv[1:])

# ======================================================================

"""
Commented out:

        if plotno%4!=1:
            print mlab,ilab
            bx.yaxis.set_major_formatter(pylab.NullFormatter())
        else:
            ticks = bx.get_yticks()
            labels = []
            for i in range(ticks.size):
                if i%2==1:
                    exp = int(("%1.e"%ticks[i]).split('e')[-1])
                    labels.append("$\\mathdefault{10^{%d}}$"%exp)
                else:
                    labels.append('')
            labels[-1] = ''
            bx.yaxis.set_ticklabels(labels)
        if y!=0.1:
            bx.xaxis.set_major_formatter(pylab.NullFormatter())
        else:
            bx.xaxis.set_major_formatter(pylab.ScalarFormatter())
            bx.set_xticks([2.,20.])
        plotno += 1

#fig = pylab.gcf()
#pylab.text(0.47,0.02,'$r$ (kpc)',transform=fig.transFigure)
#pylab.text(0.006,0.38,'$\\rho(r)$ (M$_\odot$ kpc$^{-3}$)',transform=fig.transFigure,rotation='vertical')
ax.set_xlabel('$r$ (kpc)')
ax.set_ylabel('$\\rho(r)$ (M$_\odot$ kpc$^{-3}$)')
"""
