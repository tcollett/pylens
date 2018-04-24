#!/home/mauger/bin/python -W ignore

import pyfits,scipy,glob
#import matplotlib.nxutils as nxutils
import matplotlib.path as path

import sys
if len(sys.argv)>1:
    coordfile = sys.argv[1]
else:
    coordfile = 'mask.coords' 
if len(sys.argv)>2:
    scifile = sys.argv[2]
else:
    scifile = 'image.fits'
if len(sys.argv)>3:
    maskfile = sys.argv[3]
else:
    maskfile = 'mask.fits'

d = pyfits.open(scifile)[0].data.shape
npts = d[0]*d[1]
points = scipy.indices(d)[::-1].reshape((2,npts)).T + 1

mask = scipy.zeros(d)
f = open(coordfile).readlines()[1:]
for poly in f:
    verts = scipy.asarray(poly.split('{')[1].split('}')[0].split())
    verts = verts.reshape((verts.size/2,2)).astype(scipy.float32)
    #IF NORMAL MODE
    mask += path.Path(verts).contains_points(points).T.reshape(d)
    #IF HOLLOW MODE
    #if poly==f[0]:
    #    mask -= path.Path(verts).contains_points(points).T.reshape(d)
    #else:
    #    mask += path.Path(verts).contains_points(points).T.reshape(d)


mask[mask>1] = 1
pyfits.PrimaryHDU(mask).writeto(maskfile,clobber=True)
