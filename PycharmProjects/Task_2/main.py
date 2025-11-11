from PIL import Image
import numpy as np
import sys

def makeArray(image):
    im = Image.open(image)

    arr = np.array(im.getdata())
    if arr.ndim == 1:  # grayscale
        arr = arr.reshape(im.size[1], im.size[0])
    else:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(im.size[1], im.size[0], numColorChannels)
    return arr

def makeImage(arr,name):
    newIm = Image.fromarray(arr.astype(np.uint8))
    #newIm.show()

    newIm.save(name)

def doHistogram(filenamen,param):
    hist = np.zeros(256)

    arr = makeArray(filenamen)

    print(f"Function doHistogram invoked for {filenamen}")
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr.ndim == 2:
                val = int(arr[x,y])
            else:
                val = int(arr[x, y, param])
            hist[val] += 1

    hist_raw = np.copy(hist)
    hist = hist/hist.max() * 255
    hist = hist.astype(np.uint8)
    hist = np.clip(hist, 0, 255)

    histimg = np.zeros((256, 256))
    for x in range(256):
        intens = hist[x]
        for y in range(intens):
            histimg[255-y,x] = 255


    makeImage(histimg,f"histogram_{filenamen}")
    return hist_raw

def doHPower(filenamen):
    col = 0
    hist = np.zeros(256)
    histr = np.zeros(256)
    histg = np.zeros(256)
    histb = np.zeros(256)
    arr = makeArray(filenamen)

    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    print(f"Function doHPower invoked")
    if(col==0):
        hist = doHistogram(filenamen,None)
    else:
        histr = doHistogram(filenamen,0)
        histg = doHistogram(filenamen,1)
        histb = doHistogram(filenamen,2)
    gmin = np.cbrt(0)
    gmax = np.cbrt(255)
    pixelcount = arr.shape[0] * arr.shape[1]
    result = np.zeros_like(arr)
    if(col==0):
        cumulative = np.cumsum(hist) / pixelcount
        mapping = np.power(gmin + (gmax - gmin) * cumulative, 3)
        mapping = np.clip(mapping, 0, 255).astype(np.uint8)
        result = mapping[arr]
    else:
        cumr = np.cumsum(histr) / pixelcount
        mapr = np.power(gmin + (gmax - gmin) * cumr, 3)
        mapr = np.clip(mapr, 0, 255).astype(np.uint8)
        result[..., 0] = mapr[arr[..., 0]]
        cumg = np.cumsum(histg) / pixelcount
        mapg = np.power(gmin + (gmax - gmin) * cumg, 3)
        mapg = np.clip(mapg, 0, 255).astype(np.uint8)
        result[..., 1] = mapg[arr[..., 1]]
        cumb = np.cumsum(histb) / pixelcount
        mapb = np.power(gmin + (gmax - gmin) * cumb, 3)
        mapb = np.clip(mapb, 0, 255).astype(np.uint8)
        result[..., 2] = mapb[arr[..., 2]]


    makeImage(result,f"improved_{filenamen}")

def doMean(filenamen):
    hist = doHistogram(filenamen,None)
    m = np.arange(256)
    N = np.sum(hist)
    mean = np.sum(hist*m) / N
    print(f"Mean = {mean}")
    return mean

def doVariance(filenamen):
    hist = doHistogram(filenamen, None)
    m = np.arange(256)
    N = np.sum(hist)
    mean = doMean(filenamen)
    varrr = np.sum(((m - mean)**2) * hist) / N
    print(f"Variance = {varrr}")
    return varrr

def doStandDev(filenamen):
    varrr = doVariance(filenamen)
    stddev = np.sqrt(varrr)
    print(f"Standard Deviation = {stddev}")
    return stddev

def doVariationCoeffOne(filenamen):
    stddev = doStandDev(filenamen)
    mean = doMean(filenamen)
    varcoi = stddev/mean
    print(f"Variation Coefficient I = {varcoi}")
    return varcoi

def doAsymCoeff(filenamen):
    stddev = doStandDev(filenamen)
    mean = doMean(filenamen)
    hist = doHistogram(filenamen, None)
    m = np.arange(256)
    N = np.sum(hist)
    asymco = (1/(stddev**3*N)) * np.sum(((m-mean)**3)*hist)
    print(f"Asymmetric Coefficient = {asymco}")
    return asymco

def doFlatCoeff(filenamen):
    stddev = doStandDev(filenamen)
    mean = doMean(filenamen)
    hist = doHistogram(filenamen, None)
    m = np.arange(256)
    N = np.sum(hist)
    flatco = (1/(stddev**4*N)) * np.sum(((m-mean)**4)*hist) - 3
    print(f"Flattening Coefficient = {flatco}")
    return flatco

def doVariationCoeffTwo(filenamen):
    hist = doHistogram(filenamen, None)
    N = np.sum(hist)
    varcoii = (1 / N**2) * np.sum(hist**2)
    print(f"Variation Coefficient II = {varcoii}")
    return varcoii

def doEntropy(filenamen):
    hist = doHistogram(filenamen, None)
    N = np.sum(hist)
    p = hist / N
    p = p[p>0]
    ent = -1*np.sum(p*np.log2(p))
    print(f"Entropy = {ent}")
    return ent

if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if (len(sys.argv) == 2) & (sys.argv[1] != '--help'):
    print("Too few command line parameters given.\n")
    sys.exit()
param = None

command = sys.argv[1]
filenamen = sys.argv[2]
if len(sys.argv) > 3:
    param = int(sys.argv[3])


if command == '--histogram':
   doHistogram(filenamen,param)
elif command == '--hpower':
   doHPower(filenamen)
elif command == '--cmean':
    doMean(filenamen)
elif command == '--cvariance':
    doVariance(filenamen)
elif command == '--cstdev':
    doStandDev(filenamen)
elif command == '--cvarcoi':
    doVariationCoeffOne(filenamen)
elif command == '--casyco':
    doAsymCoeff(filenamen)
elif command == '--cflaco':
    doFlatCoeff(filenamen)
elif command == '--cvarcoii':
    doVariationCoeffTwo(filenamen)
elif command == '--centropy':
    doEntropy(filenamen)