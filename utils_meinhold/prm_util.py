"""set of utility functions for acquisition software"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import butter,filtfilt,iirdesign,correlate
from matplotlib import pyplot as plt
from matplotlib import mlab
import time
import os


rtd=180/np.pi

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    (from cookbook)
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)] 
    
def test_int_down(inarray):
    '''
    function to test if data set integrates down with increasing number of samples
    input is 1 dim array, output is a dictionary
    '''
    ncuts=np.int(np.fix(np.log2(len(inarray))))
    #first take max power of 2 subset of input array
    d=inarray[:2**ncuts]
    outstds=[]
    outnpts=[]
    out={}
    for i in range(ncuts):
        d=np.reshape(d,(2**i,2**(ncuts-i)))
        outstds.append(np.std(np.mean(d,axis=1)))
        outnpts.append(2**(ncuts-i))
    out['stdev']=np.array(outstds)
    out['npoints']=np.array(outnpts)
    return out

def bcd_to_int(int_1):
    #function to pull out bytes from int_1 and int_2 and reconstruct BCD decimal number
        b_1=[]
        for i1 in int_1:
            b_1.append(bin(i1))
        outinteger=[]
        ob1=[]
        ob2=[]
        ob3=[]
        ob4=[]
        ob5=[]
        bb1=0
        bb2=0
        bb3=0
        bb4=0
        bb5=0
        for b1 in b_1:
            if len(b1)==20:
                bb1=int(b1[18:20],2)
                bb2=int(b1[14:18],2)
                bb3=int(b1[10:14],2)
                bb4=int(b1[6:10],2)
                bb5=int(b1[4:6],2)
                ob1.append(bb1)
                ob2.append(bb2)
                ob3.append(bb3)
                ob4.append(bb4)
                ob5.append(bb5)
                outinteger.append( bb1+ bb2*10+100*bb3+1000*bb4+10000*bb5)
        return outinteger,ob1,ob2,ob3,ob4,ob5

def rev_to_int(rev):
    #function to pull out bytes from  revecounter input and reconstruct  number
        outinteger=[]
        out1=[]
        out2=[]
        for r in rev:
            int1=int(bin(int(r))[4:12],2)
            int2=int(bin(int(r))[12:20],2)
            out1.append(int1)
            out2.append(int2)
            outinteger.append(int1+256*int2)
        return outinteger,out1,out2


def thicklegendlines(legendname,thick=3):
    lglines=legendname.get_lines()
    for line in lglines:
        line.set_linewidth(thick)
    plt.draw()

def rebin(a, (m, n)):
    """
    Downsizes a 2d array by averaging, new dimensions must be integral factors of original dimensions
    Credit: Tim http://osdir.com/ml/python.numeric.general/2004-08/msg00076.html
    """
    M, N = a.shape
    ar = a.reshape((M/m,m,N/n,n))
    return np.sum(np.sum(ar, 2), 0) / float(m*n)


def psd_function(freqs,wnlevel,fknee,alpha):
    """ 
    function to generate model PSD function (for fitting).
    Inputs freqs(array), wnlevel (value), Fknee (value), Alpha (value, for
    simple 1/f give alpha = 1)
    result will be power spectrum for all freqs in freqs, with
    wnlevel (input should be units/sqrt(Hz)) alpha will be for Power spectrum
    slope, not amplitude spectrum
    """
    nf=len(freqs)
    psdoutput=(wnlevel**2)*(1.+ (freqs/fknee)**(-alpha))
    return(psdoutput)
    
def psd_fit_function(p,x):
    # Parameter values are passed in "p"
    # for PSD p=[wnlevel,fknee,alpha]
    # form is f(x)=(p[0]**2)*(1.+ (x/p[1])**(-p[2]))
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    # model = psd_function(x,p[0],p[1],p[2])
    model=(p[0]**2)*(1.+(x/p[1])**(-p[2]))
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return(model)
    
def psd_fit_function_resid(p,x,y,err):
    # Parameter values are passed in "p"
    # for PSD p=[wnlevel,fknee,alpha]
    # form is f(x)=(p[0]**2)*(1.+ (x/p[1])**(-p[2]))
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    # model = psd_function(x,p[0],p[1],p[2])
    model=(p[0]**2)*(1.+(x/p[1])**(-p[2]))
    status = 0
    return((y-psd_fit_function(p,x))/err)

def fit_fknee(psd,freqs):
    """
    function to call scipy optimize.leastsq to fit PSD function, assumed output of
    nps function above
    """
    if min(freqs) == 0:
        freqs=freqs[1:]
        psd=psd[1:]
    nfreq=len(freqs)
    topfreqs=np.where(freqs>.8*np.max(freqs))
    err=(np.zeros(nfreq,dtype=float)+np.std(np.sqrt(psd[topfreqs])))#*(1/freqs)
    p=np.array([np.sqrt(np.mean(psd[topfreqs])),.15,1.0])
    m=optimize.leastsq(psd_fit_function_resid,p,args=(freqs,psd,err),full_output=1)
    pfinal=m[0]
    print 'wnlevel',pfinal[0]
    print 'Fknee' ,pfinal[1]
    print 'alpha' ,pfinal[2]
    return(m)
    


def nps(s, Fs,minfreq=None):
    """
    returns two vectors, frequencies and PSD
    PSD is in units^s/Hz
    """
    if minfreq != None:
        nfft=np.min([len(s),np.int(2.*Fs/minfreq)])
        nfft=2**(np.int(np.log2(nfft)))
    elif minfreq == None:
        nfft=len(s)
        nfft=2**(np.int(np.log2(nfft)))
    #Pxx, freqs = plt.psd(s, NFFT=nfft, Fs = Fs)
    Pxx, freqs = mlab.psd(s, NFFT=nfft, Fs = Fs)
    #we hate zero frequency
    freqs=freqs[1:]
    Pxx=Pxx[1:]
    return freqs, Pxx 
    
def nonlinmodel(g_0,b,t_in):
    """
    function to calculate nonlinear Vout for given input 
    gain, bparameter, voltage
    """
    v_out=t_in*(g_0/(1+b*g_0*t_in))
    return vout
    
def spectrogram(indata,sampsperspec=1000):
    """
    make simple spectrogram, uses nps
    """
    n=len(indata)
    nspec=np.int(n/sampsperspec)
    ns=nspec*sampsperspec
    indata=np.reshape(indata[:ns],(nspec,sampsperspec))
    sgram=[]
    for i in range(nspec):
        z=nps(indata[i,:],1.)
        zlen=len(z[0])
        sgram.append(z[1])
    sgram=np.array(sgram)
    sgram=np.reshape(sgram.flatten(),(nspec,zlen))
    return sgram

def phasebin(nbins, az, signal,degrees=True):
    if degrees:
        az=(az*np.pi/180.)-np.pi
    ring_edges=np.where(np.diff(az) < -np.pi)
    nrings=len(ring_edges[0])
    phasebin_edges = np.linspace(-np.pi,np.pi, nbins+1)
    #SPLIT THE PHASE FOR EACH RING
    pseudomap = np.zeros([nbins,nrings-1],dtype=np.float32 ) 
    #plt.figure()
    #plt.hold(False)
    for ring in range(nrings-1):
        az_ring=az[ring_edges[0][ring]:ring_edges[0][ring+1]]
        signal_ring=signal[ring_edges[0][ring]:ring_edges[0][ring+1]]
        pseudomap[:,ring], edges = np.histogram(az_ring, bins=phasebin_edges, weights=signal_ring)
        #plt.plot(edges[:-1],pseudomap[:,ring])
        #plt.show()
        hits, edges = np.histogram(az_ring, bins=phasebin_edges)
        pseudomap[hits>0,ring] /= hits[hits>0]
    return pseudomap

def lowpass(d,sample_rate,cutoff):
    '''
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
    just stuff in an example from scipy to get this functional
    '''
    frac_cutoff=cutoff/(sample_rate/2.)
    print frac_cutoff
    b,a=butter(3,frac_cutoff)
    #b,a=iirdesign(frac_cutoff-.001,frac_cutoff+.1,.9,.1)
    filtered_d = filtfilt(b,a,d)
    return(filtered_d)

def highpass(d,sample_rate,cutoff):
    '''
    just do the lowpass above and subtract
    '''
    filtered_d=d-lowpass(d,sample_rate,cutoff)
    return(filtered_d)
    
def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]
    print ''.join(evList)
    return eval(''.join(evList))

def rebin_factor( a, newshape ):
        '''Rebin an array to a new shape.
        newshape must be a factor of a.shape.
        '''
        assert len(a.shape) == len(newshape)
        assert not np.sometrue(np.mod( a.shape, newshape ))

        slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
        return a[slices]


    
def oplot(*params):
    """function to emulate one good feature of IDL oplot with nice features of Python plot"""
    plt.hold(True)
    plot(*params)
    plt.hold(False)
    return
    
    
def linfit(x,y):
    """embed python lin algebra fitting to look like idl tool"""
    if len(y) != len(x):
        print 'inputs need to be same length arrays'
        
    a=np.vstack([x,np.ones(len(x))]).T
    m,b=np.linalg.lstsq(a,y)[0]
    return np.array([b,m])

  
 
def grab_x_from_plot(fig):

    print 'right button press selects xvalue to store. middle click to  end function'
    global startlist,stoplist,start
    global cid
    global ptnum
    ptnum=0
    start=True
    startlist=[]
    stoplist=[]
    
    def onclick(event):
        global ptnum,startlist,stoplist,start
        global cid
        print ptnum
        print ptnum%2
        if event.button == 3:
            if ptnum%2 == 0:
                startlist.append(event.xdata)
                print 'chose start point',event.xdata
                print 'select stop point'
            elif ptnum%2 == 1:
                stoplist.append(event.xdata)
                print 'chose stop point',event.xdata
                print 'select next startpoint'
            start=False
            ptnum+=1
        elif event.button ==2:
            print 'should quit now, was here: ',event.xdata
            print cid
            fig.canvas.mpl_disconnect(cid)
    cid=fig.canvas.mpl_connect('button_press_event',onclick)
    outlist=np.concatenate((np.array(startlist),np.array(stoplist)),axis=2)
    return(outlist)

def gaussian(p, x):
    # Parameter values are passed in "p"
    # for gaussian p=[offset,amplitude,xposition, sigma]
    # form is f(x)=p[0]+p[1]*exp(-((x-p[2])/p[3])^2)
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    p=np.array(p)
    model = p[0]+p[1]*np.exp(-((x-p[2])/p[3])**2)
    return(model)            
       
def gaussianresid(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # for gaussian p=[offset,amplitude,sigma]
    # form is f(x)=p[0]+p[1]*exp(-((x-p[2])/p[3])^2)
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    # p=np.array(p)
    model = p[0]+p[1]*np.exp(-((x-p[2])/p[3])**2)    
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return([status, (y-model)/err])
    
    
def fit_gaussian(x,y,yerr=None):
    import mpfit as mp
    if yerr==None:
        yerr=np.std(y)
    err=yerr*np.ones(x.size,dtype=float)
    fa={'x':x,'y':y,'err':err}
    peakval=x[y==np.max(y)][0]
    p=np.array([np.min(y),np.max(y)-np.min(y),peakval,.1])
    m=mp.mpfit(gaussianresid,p,functkw=fa,quiet=1,maxiter=10)
    if m.status<1:
        print(p)
        print(values)
        print(fa)
        print(parinfo)
        plt.plot(toi)
        plt.plot(toierr)
    
#    sigma=abs(m.params[3])
#    center=m.params[2]
#    redchi=m.fnorm
    return m
    

def linfit(x,y):
    """embed python lin algebra fitting to look like idl tool"""
    if len(y) != len(x):
        print 'inputs need to be same length arrays'
        
    a=np.vstack([x,np.ones(len(x))]).T
    m,b=np.linalg.lstsq(a,y)[0]
    return np.array([b,m])
    
  
        
    
        
        