import numpy
import numpy as np
import numpy.random as rnd
from math import log,log10,pi
import emcee
import ensembleTEC
import time

global hackpars
global hackcosts
#global nregs

def optFunc(x,storemetadata=False):
    #print "opt"
    logp = 0.
    bad = False
    nvars = len(hackpars)
    metadata=[]
    for varIndx in xrange(nvars):
        hackpars[varIndx].value = x[varIndx]
        try:
            logp += hackpars[varIndx].logp
        except:
            logp = -1e200
            bad = True
            break
    if bad==False:
        for cost in hackcosts:
            #print cost,cost.logp
            if storemetadata==False:
                logp += cost.logp
            else:
                lp,meta = cost.logp,cost.meta
                metadata.append(meta)
                logp+=lp

    #rint logp

    if hackpars[varIndx].value != x[varIndx]:print "garbage"
    if storemetadata==False:
        return logp
    else:
        return logp,metadata

#pt needs a dummy prior...
def logp(x):
    return 0.0

class ParEmcee:
    def __init__(self,pars,cov,regconsts,nwalkers=None,nthreads=1):
        global hackpars,hackcosts,nregs
        self.nthreads=nthreads
        self.pars = []
        self.costs = []
        self.deterministics = []
        self.regs=regconsts
        self.nregs=len(self.regs)
        for par in pars:
            try:
                if par.observed==True:
                    self.costs.append(par)
                else: self.pars.append(par)
            except:
                self.deterministics.append(par)

        self.nvars = len(self.pars)
        self.ndim = self.nvars

        self.cov = cov

        if nwalkers==None:
            self.nwalkers = self.ndim*5
        else:
            self.nwalkers = nwalkers
        hackpars = self.pars
        hackcosts = self.costs
        nregs=self.nregs
        print emcee.__file__

        self.sampler = ensembleTEC.EnsembleSamplerTEC(self.nwalkers,self.ndim,optFunc,threads=nthreads)

    def sample(self,niter,burn=0,fixfirstwalker=True):
        print "preparing to sample"

        if burn!=0: print "burn isn't coded up"

        vals = numpy.array([p.value for p in self.pars])

        p0 = [vals]

        set = 1
        while set<self.nwalkers:
            trial = vals+numpy.random.randn(self.ndim)*self.cov
            #always start first walker at the peak
            if set==1 and fixfirstwalker: trial=vals
            ok = True
            for t in range(trial.size-self.nregs):
                self.pars[t].value = trial[t]
                try:
                    self.pars[t].logp
                    #print  self.pars[t],self.pars[t].value

                except:
                    #print  self.pars[t],self.pars[t].value
                    ok = False
                    break

            if ok==True:
                if set==1: trial=vals
                set += 1
                p0.append(trial)

        print "starpositions loaded"
        pos, prob, state = self.sampler.run_mcmc(p0, 1)
        opos,oprob,orstate = [],[],[]

        print "sampling"
        I=0
        import time
        self.t0=time.time()
        for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter,storechain=False):
            I+=1
            if I ==10 or I%50==0:
                print I,niter,time.time()-self.t0

            if niter*self.nwalkers>200*10000:
                if niter*self.nwalkers>200*100000: 
                    if I%100!=0: continue    
                if I%10!=0: continue            
                opos.append(pos.copy())
                oprob.append(prob.copy())

            else:
                opos.append(pos.copy())
                oprob.append(prob.copy())
            


        self.endpoint=pos
        self.endprob=prob
        self.endstate=rstate

        self.pos = numpy.array(opos)
        self.prob = numpy.array(oprob)


    def checkpoint(self):
        return [self.endpoint,self.endprob,self.endstate]


    def output(self):
        nstep,nwalk = numpy.unravel_index(self.prob.argmax(),self.prob.shape)
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.pos[:,:,i]
        #for i in range(len(self.deterministics)):
        #    d = []
        return self.pos,self.prob,result,self.pos[nstep,nwalk]



class TemperedEmcee:
    def __init__(self,pars,cov,regconsts,nwalkers=None,nthreads=1,ntemps=5):
        self.t0=time.time()
        global hackpars,hackcosts,nregs
        self.pars = []
        self.costs = []
        self.deterministics = []
        self.regs=regconsts
        self.nregs=len(self.regs)
        self.ntemps=ntemps
        self.nthreads=nthreads
        for par in pars:
            try:
                if par.observed==True:
                    self.costs.append(par)
                else: self.pars.append(par)
            except:
                self.deterministics.append(par)

        #now append the regconsts to the pars, and give them a dummy covariance
        #for reg in self.regs:
        #    #break
        #    self.pars.append(reg)
        #    if type(cov)==list:
        #        self.cov.append(reg.value/100.)
        #    else:
        #        cov=numpy.append(cov,reg.value/100.)

        self.nvars = len(self.pars)
        self.ndim = self.nvars

        self.cov = cov

        if nwalkers==None:
            self.nwalkers = self.ndim*8
        else:
            self.nwalkers = nwalkers
        hackpars = self.pars
        hackcosts = self.costs
        nregs=self.nregs
        print emcee.__file__


        try:
            self.sampler = emcee.PTSampler(self.ntemps,self.nwalkers,self.ndim,optFunc,logp,threads=nthreads)
        except AttributeError:
            "There is no Parallel Tempered Emcee sampler installed, try the untempered option?"
            
    def sample(self,niter,burn=0,fixfirstwalker=True):
        burnthreshold=burn
        print "sampling"
        vals = numpy.array([p.value for p in self.pars])

        p0=[]

        for k in range(self.ntemps):
          p0i = [vals]
          set = 1
          while set<self.nwalkers:
            trial = vals+numpy.random.randn(self.ndim)*self.cov
            #dont always start first walker at the peak
            if set==1 and fixfirstwalker: trial=vals
            ok = True
            for t in range(trial.size-self.nregs):
                self.pars[t].value = trial[t]
                try:
                    self.pars[t].logp
                    #print  self.pars[t],self.pars[t].value

                except:
                    #print  self.pars[t],self.pars[t].value
                    ok = False
                    break

            if ok==True:
                if set==1: trial=vals
                set += 1
                p0i.append(trial)

          p0.append(p0i)

        p0 = numpy.dstack(p0)
        p0 = numpy.rollaxis(p0,-1)
        #print p0.shape

        #print p0

        for pos, prob, state in self.sampler.sample(p0, iterations=1): 
            pass
        self.sampler.reset()
        

        print "starpositions loaded"

        self.t0=time.time()

        thinfac=1
        if niter*self.nwalkers*self.ntemps>100000000:
            thinfac=10
        if niter*self.nwalkers*self.ntemps>1000000000:
            thinfac=100
        if niter*self.nwalkers*self.ntemps>10000000000:
            thinfac=1000

        if (niter<burnthreshold) or (burnthreshold==0):
            I=0
            for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter,storechain=True,thin=thinfac):
                self.endpoint=pos
                self.endprob=prob
                self.endstate=rstate
                I+=1
                if I ==10 or I%50==0:
                    print I,niter,time.time()-self.t0
        else:
            I=0
            for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=burnthreshold,storechain=True,thin=thinfac):
                self.endpoint=pos
                self.endprob=prob
                self.endstate=rstate
                I+=1
                if I ==10 or I%50==0:
                    print I,niter,time.time()-self.t0


            colds=3
            pburnt=self.sampler.chain[0:colds,:,-1,:]
            print pburnt.shape
            print "burning start of chain, losing hot chains"
            self.sampler = emcee.PTSampler(colds,self.nwalkers,self.ndim,optFunc,logp,threads=self.nthreads)

            for pos, prob, state in self.sampler.sample(pburnt, iterations=1): 
                pass
            self.sampler.reset()

            I=burnthreshold
            for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter-burnthreshold,storechain=True,thin=thinfac):
                self.endpoint=pos
                self.endprob=prob
                self.endstate=rstate
                I+=1
                if niter<11:
                    print I,niter,time.time()-self.t0
                if I ==10 or I%50==0:
                    print I,niter,time.time()-self.t0

        #we're only interested in the 0 temperature samples 
                #strictly we could use the lower temperature samples if we 
                #re-weight them, but meh.      

        print thinfac
        self.pos=self.sampler.chain[0]
        self.prob=self.sampler.lnprobability[0]
        
        print self.prob.shape, self.pos.shape,self.sampler.chain.shape

        #print "max acor:",numpy.max(self.sampler.acor)
        

    def checkpoint(self):
        return [self.endpoint,self.endprob,self.endstate]


    def output(self):
        nstep,nwalk = numpy.unravel_index(self.prob.argmax(),self.prob.shape)
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.pos[:,:,i]
        #for i in range(len(self.deterministics)):
        #    d = []
        return (self.pos),(self.prob).T,result,self.pos[nstep,nwalk]
