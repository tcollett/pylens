
def getColor(f1,f2,z):
    import numpy
    from stellarpop import tools
    import spasmoid
    dir = spasmoid.__path__[0]+"/data"

    qso = numpy.loadtxt('%s/vandenBerk.dat'%dir)
    w = qso[:,0].copy()
    m = qso[:,1].copy()
    s = qso[:,2].copy()

    f1 = tools.filterfromfile(f1)
    f2 = tools.filterfromfile(f2)

    sed = [w,m]
    color = tools.ABFilterMagnitude(f1,sed,z)-tools.ABFilterMagnitude(f2,sed,z)
    return color
