from PIL import Image
import numpy as np
import sys

def makeArray(image):
    im = Image.open(image)

    arr = np.array(im.getdata(), complex)
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

def bit_reverse_indices(N):
    bits = int(np.log2(N))
    rev = np.zeros(N, dtype=int)

    for i in range(N):
        b = format(i, f'0{bits}b')
        rev[i] = int(b[::-1], 2)

    return rev

def frequency_grid(N, M):
    u = np.arange(-N//2, N//2)
    v = np.arange(-M//2, M//2)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    return U, V, D

def low_pass_filter(F, D0):
    N, M = F.shape
    _, _, D = frequency_grid(N, M)
    H = (D <= D0).astype(float)
    return F * np.fft.fftshift(H)

def high_pass_filter(F, D0):
    N, M = F.shape
    _, _, D = frequency_grid(N, M)
    H = (D > D0).astype(float)
    return F * np.fft.fftshift(H)

def band_pass_filter(F, D1, D2):
    N, M = F.shape
    _, _, D = frequency_grid(N, M)
    H = ((D >= D1) & (D <= D2)).astype(float)
    return F * np.fft.fftshift(H)

def band_cut_filter(F, D1, D2):
    N, M = F.shape
    _, _, D = frequency_grid(N, M)
    H = ((D < D1) | (D > D2)).astype(float)
    return F * np.fft.fftshift(H)

def directional_high_pass(F, angle, width=0.3):
    N, M = F.shape
    U, V, D = frequency_grid(N, M)

    theta = np.arctan2(V, U)
    H = np.abs(np.sin(theta - angle)) < width

    return F * np.fft.fftshift(H.astype(float))

def phase_modifying_filter(F, phase_shift):
    return F * np.exp(1j * phase_shift)


def dft1d(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        s = 0.0 + 0.0j
        for n in range(N):
            s += x[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = s

    return X

def fft_dif_1d(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if (N & (N - 1)) != 0:
        raise ValueError("Size must be power of 2")

    X = x.copy()
    step = N

    while step > 1:
        half = step // 2
        W = np.exp(-2j * np.pi * np.arange(half) / step)

        for k in range(0, N, step):
            for n in range(half):
                a = X[k + n]
                b = X[k + n + half]

                X[k + n] = a + b
                X[k + n + half] = (a - b) * W[n]

        step //= 2

    return X[bit_reverse_indices(N)]

def ifft_dif_1d(X):
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]

    if (N & (N - 1)) != 0:
        raise ValueError("Size must be power of 2")

    x = X.copy()
    step = N

    while step > 1:
        half = step // 2
        W = np.exp(+2j * np.pi * np.arange(half) / step)

        for k in range(0, N, step):
            for n in range(half):
                a = x[k + n]
                b = x[k + n + half]

                x[k + n] = a + b
                x[k + n + half] = (a - b) * W[n]

        step //= 2

    x = x[bit_reverse_indices(N)]
    return x / N

def fft2_dif(img):
    img = np.asarray(img, dtype=complex)
    N, M = img.shape

    if (N & (N - 1)) or (M & (M - 1)):
        raise ValueError("Image dimensions must be powers of 2")

    F = img.copy()

    for i in range(N):
        F[i, :] = fft_dif_1d(F[i, :])

    for j in range(M):
        F[:, j] = fft_dif_1d(F[:, j])

    return F

def ifft2_dif(F):
    F = np.asarray(F, dtype=complex)
    N, M = F.shape

    img = F.copy()

    for i in range(N):
        img[i, :] = ifft_dif_1d(img[i, :])

    for j in range(M):
        img[:, j] = ifft_dif_1d(img[:, j])

    return img


def doDFT(filename):
    x = makeArray(filename)
    N, M = x.shape

    temp = np.zeros((N, M), dtype=complex)

    for m in range(M):
        temp[:, m] = dft1d(x[:, m])
        print("forend"+str(m))

    F = np.zeros((N, M), dtype=complex)

    for p in range(N):
        F[p, :] = dft1d(temp[p, :])
        print("forendp"+str(p))

    return F

def doIFT(F):
    N, M = F.shape
    img = np.zeros((N, M), dtype=complex)

    for n in range(N):
        for m in range(M):
            s = 0.0 + 0.0j
            for p in range(N):
                for q in range(M):
                    s += F[p, q] * np.exp(2j * np.pi * ((p * n) / N +(q * m) / M))
            img[n, m] = s / (N * M)
            print(str(n) + " -- " + str(m))
    return img

def saveDFTSpectrum(F, name="spectrum.bmp"):
    img = np.abs(F)
    img = np.log(1 + img)
    img = np.fft.fftshift(img)
    img = 255 * img / img.max()

    makeImage(img, name)

def save_fft_spectrum(F, name="fft_spectrum.bmp"):
    mag = np.abs(F)
    mag = np.log(1 + mag)
    mag = np.fft.fftshift(mag)
    mag = 255 * mag / mag.max()

    Image.fromarray(mag.astype(np.uint8)).save(name)


command = sys.argv[1]
filenamen = sys.argv[2]
if len(sys.argv) > 3:
    param = sys.argv[3]
if len(sys.argv) > 4:
    param2 = int(sys.argv[4])

if command == '--DFT':
    F = doDFT(filenamen)
    saveDFTSpectrum(F,"spectrumResult.bmp")
if command == '--IFT':
    F = doDFT(filenamen)
    img = doIFT(F)
    makeImage(np.real(img),"resultIFT.bmp")
if command == '--FFT':
    img = makeArray(filenamen)
    F = fft2_dif(img)
    save_fft_spectrum(F, "fftSpectrum.bmp")
if command == '--IFFT':
    img = makeArray(filenamen)
    F = fft2_dif(img)
    img_rec = ifft2_dif(F)
    makeImage(np.real(img_rec), "fftReconstructed.bmp")