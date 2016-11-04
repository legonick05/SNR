# -*- coding: utf-8 -*-
"""
module SNR.py

CalcSNR Master File

Here's all the functions used in calcSNR. Documentation can be found in the
individual files.
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as FT

class ImgPar:
    def __init__(self):
        self.filenm = 'test'
        self.scale = 1
        self.same_scale = 0
        self.cbonoff = 'on'
        self.nrimin = 0
        self.ncimin = 0
        self.pos = [1, 1, 256, 384]
        self.fig = 0
        self.titles = []
        self.title = 'test'
        self.xaxis = ''
        self.yaxis = ''
        self.axis = 0
        self.cmap = 'jet'
        self.__doc__ = '''ImgPar is an object for image parameters in dspsubimgpar()'''
        
def dspsubimgpar(imgar, dpar):
    '''Plots an input array of images 'imgar', according to \nthe parameters dpar. Also edits dpar for future use.'''
    if str(type(imgar)) != "<class 'numpy.ndarray'>":
        dpar = ImgPar()
    else:
        sztitle = len(dpar.titles)
        szxaxis = len(dpar.xaxis)
        dotitles = 0
        doxaxis = 0
        cbon = (dpar.cbonoff == 'on')
        if sztitle != 0:
            dotitles = 1
        if szxaxis != 0:
            doxaxis = 1
        [nr,nc,nch] = imgar.shape
        nrim = dpar.nrimin
        ncim = dpar.ncimin
        if (dpar.nrimin == 0) or (dpar.ncimin == 0):
            nrim = int(np.sqrt(nch*nc/nr))
            ncim = np.ceil(nch/nrim)
        nexpc = 1
        nexpr = 1
        if nc <= 64:
            nexpc = int(200/nc)
        if nr <= 64:
            nexpr = int(200/nr)
        nexp = min(nexpc, nexpr)
        dpar.pos[2] = nexp*ncim*nc + 40*cbon
        dpar.pos[3] = nexp*nrim*nr + 20*dotitles
        maximgar = np.ma.max(abs(imgar[:]))
        xoff = 0.05
        delta = 0.025
        ydelta = (1-cbon)*delta
        yoff = 0.05
        width = (0.95 - xoff + delta)/ncim - delta    # = (0.925/ncim) - 0.025
        height = (0.95 - yoff +ydelta)/nrim - ydelta  # = ((0.9 + 0.025(1-cbon))/nrim) - 0.025(1-cbon)

        fig = plt.figure()
        wm = plt.get_current_fig_manager()
        wm.window.raise_()
        for jim in range(1,nch+1):
            jr = int((jim-1)/ncim)
            jc = jim - ncim*jr
            fig.add_subplot(nrim, ncim, jim)
            if dpar.same_scale == 0:
                im = fig.axes[2*jim-2].imshow(imgar[:,:,jim-1], aspect='equal', interpolation='None')
            else:
                im = fig.axes[2*jim-2].imshow(imgar[:,:,jim-1], aspect='equal', interpolation='None')
                im.set_clim(0, maximgar/dpar.scale)
            if jim == 1:
                fig.axes[0].set_title(dpar.filenm)
            if cbon == 1:
                cb = fig.colorbar(im, ax=fig.axes[2*jim-2], fraction=0.0461, pad=0.04, orientation='vertical')
                cb.formatter.set_powerlimits((0,0))
                cb.update_ticks()
            if dotitles == 1:
                fig.axes[2*jim-2].set_title(dpar.titles[jim-1])
            if doxaxis == 1:
                fig.axes[2*jim-2].set_xlabel(dpar.xaxis[jim-1])
            else:
                fig.axes[2*jim-2].xaxis.set_visible(False)
        dpar.pos[0] = dpar.pos[0] + dpar.pos[2]
        if dpar.pos[0] > 1500:
            dpar.pos[0] = 1
            dpar.pos[1] = dpar.pos[1] + dpar.pos[3]
        fig.tight_layout()
        plt.show()
                       
    return dpar
    
def cellorder(vect, want):      
    '''want is a str, vect is a list of strs'''
    ndim = len(want)
    ndimv = len(vect)   
    vectind = [0]
    for i in range(ndim):
        vectind[i] = i
        if i < ndim-1:
            vectind.append(0)
    
    if ndimv <= ndim:
        nofit = ndimv
        jfit = 0
        for jdim in range(1,ndim+1):
            for kdim in range(1,ndimv+1):
                if want[jdim-1] == vect[kdim-1]:
                    jfit += 1
                    vectind[jfit-1] = kdim-1
                    nofit = nofit - 1
    else:
        nofit = 1
    
    
    return [vectind, nofit]
    
def was(prompt, numin):
    numout = input(prompt + '['+str(numin) +'] = ')
    if str(numout) == '':
        numout = numin
    numout = int(numout)
    return numout
    
def findNoise(imgalln, inscale):
    [nrall, ncn, nchn] = imgalln.shape
    nr = nrall/2
    alpha = 0.5
    weights = [0]
    for i in range(int(nr)):
        weights[i] = (1-alpha) + alpha*np.sin(np.pi*(i+1)/nr)
        if i < nr-1:
            weights.append(0)

    scalethresh = 0.8  #Average std * 1.6
    if (str(type(inscale)) == "<class 'int'>") or (str(type(inscale)) == "<class 'float'>"):
        scalethresh = inscale        
    nrn = int(nr/2)
    nrn2 = int(nrn/2)
    imnoise = np.ndarray(shape=(nrn,ncn,nchn), dtype=complex, order='F')
    imnoise[0:nrn2,:,:] = imgalln[(nrn-nrn2):nrn,:,:]
    imnoise[nrn2:nrn,:,:] = imgalln[int(nrn+nr):int(nrn+nr+nrn2),:,:]
    wgtnoise = weights[int(nrn-nrn2):int(nrn-nrn2 + nrn)] 
    for jr in range(int(nrn)):
        imnoise[jr,:,:] = imnoise[jr,:,:]/wgtnoise[jr]
        
    nP = 8
    stdm = np.ndarray(shape=(nP,nP,nchn), dtype=float, order='F')
    nrp = int(nrn/nP)
    ncp = int(ncn/nP)    #patches
    for jp in range(nP):
        for jc in range(nP):
            for jch in range(nchn):
                xx = np.std(imnoise[(jp*nrp):(jp+1)*nrp,(jc*ncp):(jc+1)*ncp,jch], axis=0, ddof=1)
                stdm[jp,jc,jch] = np.std(xx[:], axis=0, ddof=1)  
    
    stdv = stdm.reshape(nP*nP, nchn, order='F').copy()
    stdvsort = stdv.copy()
    usevmat = np.zeros(shape=(nP,nP,nchn), dtype=float, order='F')
    stdth = np.zeros(shape=(1,nchn), dtype=float, order='F')
    for jch in range(nchn):
        val = np.ma.sort(stdv[:,jch], axis=0).copy()
        stdvsort[:,jch] = val[:]
        stdth[0,jch] = val[0]*(1+scalethresh)
        usevmat[:,:,jch] = (stdm[:,:,jch] < stdth[0,jch])
    usemat = (nchn == np.sum(usevmat, axis=2))
    
    nmat = np.sum(usemat)
    imnew = np.zeros(shape=(nrp*nmat,ncp,nchn), dtype=complex, order='F')
    js = 0
    imnoise2 = imnoise.copy()    
    for jp in range(int(nP)):
        for jc in range(int(nP)):
            if usemat[jp,jc] == 1:
                js += 1
                imnew[(js*nrp):(js+1)*nrp,:,:] = imnoise[(jp*nrp):(jp+1)*nrp,(jc*ncp):(jc+1)*ncp,:]
            else:
                imnoise2[(jp*nrp):(jp+1)*nrp,(jc*ncp):(jc+1)*ncp,:] = 0
    covmat = np.cov(np.reshape(imnew, (nrp*ncp*nmat, nchn), order='F'), rowvar=False, ddof=1)
    cormat = np.zeros(shape=(covmat.shape), dtype=complex, order='F')
    for i in range(8):
        for j in range(8):
            cormat[i,j] = covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j]) 

    jtst = 0
    if jtst == 1:
        plt.plot(stdvsort[:,1])
        plt.hold(True)
        input('')
        for jch in range(2,nchn+1):
            plt.plot(stdvsort[:,jch])
            input('')
            
    return [imnew, imnoise, imnoise2, covmat, cormat]
    
def run(ksdata, dimvect):
    plt.rcParams['figure.figsize'] = [12.0,9.0]
    #%% File-naming section goes here
    #%%
    print('Signal file')
    jread = 1
    do3d = was('1=do 3D FFT, other = 2D', 0)
    ndim = ksdata.ndim
    if do3d == 0:
        if ndim == 5:
            [vectind,nofit] = cellorder(dimvect, ['col','lin','slc','rep','cha'])
            if nofit != 0:
                [vectind,nofit] = cellorder(dimvect, ['col','lin','slc','acq','cha'])
            ks_all = np.transpose(ksdata, vectind)  #[1,2,3,4]
            ks_all = ksdata[:,:,:,1,:].squeeze()
            [nr,nc,nrep,nch] = ks_all.shape
        elif ndim == 4:
            [vectind,nofit] = cellorder(dimvect, ['col','lin','slc','rep','cha'])
            if nofit != 0:
                [vectind,nofit] = cellorder(dimvect, ['col','lin','slc','acq','cha'])
            ks_all = np.transpose(ksdata, vectind)
            [nr,nc,nrep,nch] = ks_all.shape
        elif ndim == 3:
            [vectind,nofit] = cellorder(dimvect, ['col','lin','cha'])
            ks_all = np.transpose(ksdata, vectind)
            [nr,nc,nch] = ks_all.shape
            ks_all = ks_all.reshape(nr,nc,1,nch, order='F')
        elif ndim == 2:
            [vectind,nofit] = cellorder(dimvect, ['col','lin'])
            ks_all = np.transpose(ksdata, vectind)
    if do3d == 1:
        ndim = ksdata.ndim
        if ndim == 5:
            [vectind,nofit] = cellorder(dimvect.lower(), ['col','lin','par','rep','cha','eco','acq'])
            if nofit != 0:
                [vectind, nofit] = cellorder(dimvect, ['col','lin','par','acq','cha'])
            ks_all = np.transpose(ksdata, vectind)
            [nrall,nc,ns,ntime,nch] = ks_all.shape
            jtime = was('time in sequence to process: ',int(ntime/2))
            ks_all = ks_all[:,:,:,jtime,:].squeeze()
        elif ndim == 4:
            [vectind,nofit] = cellorder(dimvect, ['col','lin','par','cha'])
            ks_all = np.transpose(ksdata, vectind)
            [nrall,nc,ns,nch] = ks_all.shape
        else:
            print('must be dimension 4 or 5, not: ' + str(ndim))
            import sys
            sys.exit()
        jslice = was('slice number to use for noise covmat: ',int(ns/2))
    #%%
    [nrall,nc,ns,nch] = ks_all.shape
    nr = int(nrall/2)
    
    imgall = np.ndarray(shape=(ks_all.shape), dtype=np.complex64)
    imgall = FT.fftshift(FT.fft(FT.ifftshift(ks_all, axes=0),axis=0),axes=0)
    imgall = imgall.astype(np.complex64)
    imgall = FT.fftshift(FT.fft(FT.ifftshift(imgall, axes=1),axis=1),axes=1)
    imgall = imgall.astype(np.complex64)
    if do3d == 1:
        imgall = FT.fftshift(FT.fft(FT.ifftshift(imgall, axes=2),axis=2),axes=2)
        
    #if ndim == 4:
    [nrall,nc,ns,nch] = imgall.shape
    jslice = 1
    if ns > 1:
        jslice = int(was('slice number to use: ',int(ns/2)))
    #imgalln = np.reshape(imgall, (nrall,nc*ns,nch), order='F')
    imgalln = imgall[:,:,jslice-1,:].squeeze()
        
    [nrall,nc,nch] = imgalln.shape    #[nrall,ncs,nch] = imgalln.shape
    nr = int(nrall/2)
    
    filenoise = filenm   #From 1st section, or input filenm manually
    
    removelines = 1
    [imnew, imnoise, immnoise2, covmat, cormat] = findNoise(imgalln,0.9)
    #Separate out the noise from the image
    imsignal = imgall[int(nr/2):int(nr+(nr/2)),:,jslice-1,:]
    imsignal = np.reshape(imsignal, (nr, nc, jslice, nch))
    ns = 1
    [nrn,ncn,nchn] = imnew.shape
    #Combine channels using only diagonal terms - to form SNR image
    imsos = np.zeros(shape=(nr,nc,ns), dtype=complex, order='F')
    imSNR = np.zeros(shape=(nr,nc,ns,nch), dtype=complex, order='F')
    imsumopt = np.zeros(shape=(nr,nc,ns), dtype=complex, order='F')
    imsumnot = imsumopt
    imsosdenom = imsumopt
    imbaddenom = imsumopt
    invcov = np.linalg.inv(covmat)
    badcov = np.conj(invcov) * covmat * np.conj(invcov)
    
    for ich in range(nch):
        imSNR[:,:,:,ich] = abs(imsignal[:,:,:,ich]/np.sqrt(covmat[ich,ich]))
        imsos = imsos + np.multiply(np.conj(imsignal[:,:,:,ich-1]), imsignal[:,:,:,ich-1])/covmat[ich-1,ich-1]
        for jch in range(nch):
            imsosdenom = imsosdenom + np.multiply(imsignal[:,:,:,ich]/covmat[ich,ich], covmat[ich,jch]*np.conj(imsignal[:,:,:,jch])/covmat[jch,jch])
            imbaddenom = imbaddenom + np.multiply(imsignal[:,:,:,ich], badcov[ich,jch]*np.conj(imsignal[:,:,:,jch]))
            imsumopt = imsumopt + np.multiply(imsignal[:,:,:,ich], invcov[ich,jch]*np.conj(imsignal[:,:,:,jch]))
            imsumnot = imsumnot + np.multiply(np.conj(imsignal[:,:,:,jch]), invcov[ich,jch]*imsignal[:,:,:,jch])
    imSNR = imSNR.astype(np.float32)
           
    absimsumopt = np.sqrt(abs(imsumopt))   #The optimal
    absimsumnot = np.sqrt(abs(imsumnot))
    absimsos = abs(imsos/np.sqrt(abs(imsosdenom)))  #See paper by Keil and Wald, 2013
    absimnot = abs(imsumnot)/np.sqrt(abs(imbaddenom))   #The true SNR of the non-optimal combination
    
    #Individual channel analysis
    nrow = int(np.sqrt(nch*nc/nr))
    nrown = int(np.sqrt(nch*ncn/nrn))
    ncol = np.ceil(nch/nrow)
    if nrown != 0:
        ncoln = np.ceil(nch/nrown)
    else:
        ncoln = np.inf
    dpar = ImgPar()
    dpar.filenm = filenm
    dpar.irimin = nrown
    dpar.icimin = ncoln
    dpar.scale = 0
    dpar.title = filenm
    dpar = dspsubimgpar(imSNR.squeeze(), dpar)   #dspsubimg3(imSNR,nrow,ncol,filenm)
    dpar = dspsubimgpar(immnoise2.real, dpar)   #dspsubimg3(imnew,nrown,ncoln,filenoise)
    
    imgdif = abs(imsumopt) - abs(imsos)
    dofullcomp = 0
    if dofullcomp == 1:
        imgarcomp = np.zeros(shape=(nr,nc,6), dtype=complex, order='F')
        imgarcomp[:,:,1] = absimsos[:,:,jslice]
        imgarcomp[:,:,2] = absimsumopt[:,:,jslice]
        imgarcomp[:,:,3] = absimnot[:,:,jslice]
        imgarcomp[:,:,4] = absimsumopt[:,:,jslice] - absimsos[:,:,jslice]
        imgarcomp[:,:,5] = absimnot[:,:,jslice] - absimsos[:,:,jslice]
        imgarcomp[:,:,6] = absimsumopt[:,:,jslice] - absimnot[:,:,jslice]
        dpar.titles = ['SNR SoS','optimal combine','not optimal comb','optimal-sos','notopt - sos','opt-not']
        dpar = dspsubimgpar(imgarcomp, dpar)
    imgarcomb = np.zeros(shape=(nr,nc,3), dtype=float, order='F')
    imgarcomb[:,:,0] = absimsos[:,:,0]      #jslice
    imgarcomb[:,:,1] = absimsumopt[:,:,0]
    imgarcomb[:,:,2] = absimsumopt[:,:,0] - absimsos[:,:,0]
    
    dpar.titles = ['SNR SoS','SNR optimal combine','SNR optimal-SoS']
    dpar.cbonoff = 'on'
    dpar.scale = 1
    dpar.xaxis = ''
    dpar = dspsubimgpar(imgarcomb, dpar)     #dspsubimgs(imgarcomb,titles)
    
    plt.rcParams['figure.figsize'] = [8.0,6.0]
    imgarcov = np.zeros(shape=(nch,nch,2), dtype=float, order='F')
    imgarcov[:,:,1] = abs(cormat)
    imgarcov[:,:,0] = abs(covmat)
    titlecov = ['covariance','correlation']
    dpar.titles = titlecov
    dpar.xaxis = ['coil','coil']
    dpar.cbonoff = 'on'
    dpar.scale = 1
    dpar = dspsubimgpar(imgarcov, dpar)    #dspsubimgs(imgarcov,titlecov)