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

def makeImage(arr):
    newIm = Image.fromarray(arr.astype(np.uint8))
    newIm.show()

    newIm.save("result.bmp")


def doBrightness(param,filenamen):
    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')

    print(f"Function doBrightness invoked with param: {param}")

    if col == 0:
        for x in range(arr.shape[0]):
            for y in range (arr.shape[1]):
                if (arr[x,y] + param < 0):
                    arr[x,y] = 0
                elif (arr[x,y] + param > 255):
                    arr[x,y] = 255
                else:
                    arr[x,y] = arr[x,y] + param

    if col == 1:
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    if (arr[x, y, z] + param < 0):
                        arr[x, y, z] = 0
                    elif (arr[x, y, z] + param > 255):
                        arr[x, y, z] = 255
                    else:
                        arr[x, y, z] += param

    makeImage(arr)

def doContrast(param,filenamen):
    print(f"Function doContrast invoked with param: {param}")

    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')

    if col == 0:
        for x in range(arr.shape[0]):
            for y in range (arr.shape[1]):
                change = (arr[x,y] - (255/2))*(param/100)
                if (arr[x,y] + change < 0):
                    arr[x,y] = 0
                elif (arr[x,y] + change > 255):
                    arr[x,y] = 255
                else:
                    arr[x,y] = arr[x,y] + change

    if col == 1:
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    change = (arr[x, y, z] - (255 / 2)) * (param / 100)
                    if (arr[x, y, z] + change < 0):
                        arr[x, y, z] = 0
                    elif (arr[x, y, z] + change > 255):
                        arr[x, y, z] = 255
                    else:
                        arr[x, y, z] += change

    makeImage(arr)

def doNegative(filenamen):
    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')

    print(f"Function doNegative invoked")

    if col == 0:
        for x in range(arr.shape[0]):
            for y in range (arr.shape[1]):
                arr[x,y] = 255 - arr[x,y]

    if col == 1:
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    arr[x,y,z] = 255 - arr[x,y,z]

    makeImage(arr)

def doHorFlip(filenamen):
    print(f"Function doHorizontalFlip invoked")
    arr = makeArray(filenamen)

    arr = np.flip(arr, 1)

    makeImage(arr)

def doVerFlip(filenamen):
    print(f"Function doVerticalFlip invoked")
    arr = makeArray(filenamen)

    arr = np.flip(arr, 0)

    makeImage(arr)

def doDiagFlip(filenamen):
    print(f"Function doDiagonalFlip invoked")
    arr = makeArray(filenamen)

    arr = np.flip(arr, 0)
    arr = np.flip(arr, 1)

    makeImage(arr)

def doEnlarge(param, filenamen, big):
    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    if (big):
        print(f"Function doEnlarge invoked with param: {param}")
    else:
        print(f"Function doShrink invoked with param: {param}")

    size1 = arr.shape[0]
    scale = param/100
    size2 = int(round(size1*scale))
    newpixelsP = [int(i * (1/scale)) for i in range(size2)]
    newpixels = np.array(newpixelsP)
    newpixels = np.clip(newpixels,0,size1-1)
    if col == 0:
        newarr = arr[newpixels[:,None],newpixels]
    else:
        newarr = arr[newpixels[:, None], newpixels, :]

    makeImage(newarr)

def doMid(filenamen):
    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    print(f"Function doMid invoked")

    newarr = np.copy(arr)
    arrnew = np.copy(arr)
    if col == 0:
        for x in range(1, arr.shape[0] - 1):
            for y in range(1, arr.shape[1] - 1):
                window = arrnew[x-1:x+2, y-1:y+2]
                newarr[x, y] = int((np.min(window) + np.max(window)) / 2)
    else:
        for x in range(1, arr.shape[0] - 1):
            for y in range(1, arr.shape[1] - 1):
                for c in range(3):
                    window = arrnew[x-1:x+2, y-1:y+2, c]
                    newarr[x, y, c] = int((np.min(window) + np.max(window)) / 2)

    makeImage(newarr)


def doAMean(filenamen):
    col = 0
    arr = makeArray(filenamen)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    print(f"Function doAMean invoked")

    newarr = np.copy(arr)
    arrnew = np.copy(arr)
    if col == 0:
        for x in range(1, arr.shape[0] - 1):
            for y in range(1, arr.shape[1] - 1):
                window = arrnew[x-1:x+2, y-1:y+2]
                newarr[x, y] = int((np.sum(window) / 9))
    else:
        for x in range(1, arr.shape[0] - 1):
            for y in range(1, arr.shape[1] - 1):
                for c in range(3):
                    window = arrnew[x-1:x+2, y-1:y+2, c]
                    newarr[x, y, c] = int((np.sum(window) / 9))

    makeImage(newarr)

def doMeanSquarerror(filenamen,filenamen2,filenamen3,peak):
    col = 0
    arr = makeArray(filenamen)
    arr2 = makeArray(filenamen2)
    arr3 = makeArray(filenamen3)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    if (peak == False):
        print(f"Function doMeanSquareError invoked")
    else:
        print(f"Function doPeakMeanSquareError invoked")
    print(f"Reference image: {filenamen}, Noisy image: {filenamen2}, Denoised image: {filenamen3}")

    sum1 = 0
    sum2 = 0
    divnum = arr.shape[0] * arr.shape[1]
    maxvalsqr = np.max(arr) ** 2
    if (col != 0):
        divnum = divnum*3

    if (col == 0):
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                diff = arr[x, y] - arr2[x, y]
                sum1 += diff ** 2
        mse1 = sum1/divnum
        if (peak == False):
            print(f"MSE between original and noisy images: {mse1}")
        else:
            pmse1 = mse1/maxvalsqr
            print(f"PMSE between original and noisy images: {pmse1}")

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                diff = arr[x, y] - arr3[x, y]
                sum2 += diff ** 2
        mse2 = sum2 / divnum
        if (peak == False):
            print(f"MSE between original and denoised images: {mse2}")
        else:
            pmse2 = mse2 / maxvalsqr
            print(f"PMSE between original and denoised images: {pmse2}")
        if (mse1>mse2):
            print("Denoising succesful")
        else:
            print("Denoising failed")
    else:
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for c in range(3):
                    diff = arr[x, y, c] - arr2[x, y, c]
                    sum1 += diff ** 2
        mse1 = sum1/divnum
        if (peak == False):
            print(f"MSE between original and noisy images: {mse1}")
        else:
            pmse1 = mse1 / maxvalsqr
            print(f"PMSE between original and noisy images: {pmse1}")

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for c in range(3):
                    diff = arr[x, y, c] - arr3[x, y, c]
                    sum2 += diff ** 2
        mse2 = sum2 / divnum
        if (peak == False):
            print(f"MSE between original and denoised images: {mse2}")
        else:
            pmse2 = mse2 / maxvalsqr
            print(f"PMSE between original and denoised images: {pmse2}")
        if (mse1>mse2):
            print("Denoising succesful")
        else:
            print("Denoising failed")

def doSignalNoiseRatio(filenamen,filenamen2,filenamen3,peak):
    col = 0
    arr = makeArray(filenamen)
    arr2 = makeArray(filenamen2)
    arr3 = makeArray(filenamen3)
    if arr.ndim == 2:  # grayscale
        print('grayscale')
    else:
        col = 1
        print('color')
    if (peak == False):
        print(f"Function doSignalToNoiseRatio invoked")
    else:
        print(f"Function doPeakSignalToNoiseRatio invoked")
    print(f"Reference image: {filenamen}, Noisy image: {filenamen2}, Denoised image: {filenamen3}")

    sum1 = 0
    sum2 = 0
    sum3 = 0
    divnum = arr.shape[0] * arr.shape[1]
    maxvalsqr = np.max(arr) ** 2
    if (col != 0):
        divnum = divnum * 3

    sum1 = np.sum(arr ** 2)
    sum2 = np.sum((arr - arr2) ** 2)
    sum3 = np.sum((arr - arr3) ** 2)

    if (peak == False):
        snr1 = 10 * np.log10(sum1/sum2)
        snr2 = 10 * np.log10(sum1/sum3)
        print(f"SNR between original and noisy images: {snr1}")
        print(f"SNR between original and denoised images: {snr2}")
        if(snr2>snr1):
            print("Denoising succesful")
        else:
            print("Denoising failed")
    else:
        psnr1 = 10 * np.log10(maxvalsqr*divnum/sum2)
        psnr2 = 10 * np.log10(maxvalsqr*divnum/sum3)
        print(f"PSNR between original and noisy images: {psnr1}")
        print(f"PSNR between original and denoised images: {psnr2}")
        if (psnr2 > psnr1):
            print("Denoising succesful")
        else:
            print("Denoising failed")

def doMaxDiff(filenamen,filenamen2,filenamen3):
    print(f"Function doMaxDifference invoked")
    print(f"Reference image: {filenamen}, Noisy image: {filenamen2}, Denoised image: {filenamen3}")
    arr = makeArray(filenamen)
    arr2 = makeArray(filenamen2)
    arr3 = makeArray(filenamen3)
    max1 = np.max(np.abs(arr - arr2))
    max2 = np.max(np.abs(arr - arr3))
    print(f"Maximum difference between original and noisy images: {max1}")
    print(f"Maximum difference between original and denoised images: {max2}")
    if (max1 >= max2):
        print("Denoising succesful")
    else:
        print("Denoising failed")

def doHelp():
    print("Available commands:")
    print("")
    print("Elementary operations:")
    print("--brightness [image_filename] [int (-255,255)]")
    print("--contrast [image_filename] [int (in percentage)]")
    print("--negative [image_filename]")
    print("")
    print("Geometric operations:")
    print("Horizontal flip:")
    print("--hflip [image_filename]")
    print("Vertical flip:")
    print("--vflip [image_filename]")
    print("Diagonal flip:")
    print("--dflip [image_filename]")
    print("--enlarge [image_filename] [int (in percentage)(min 100)]")
    print("--shrink [image_filename] [int (in percentage)(max 100)]")
    print("")
    print("Noise removal:")
    print("Midpoint filter:")
    print("--mid [image_filename]")
    print("Arithmetic mean filter:")
    print("--amean [image_filename]")
    print("")
    print("Noise comparison:")
    print("Mean square error:")
    print("--mse [original_image_filename] [noisy_image_filename] [denoised_image_filename]")
    print("Peak mean square error:")
    print("--pmse [original_image_filename] [noisy_image_filename] [denoised_image_filename]")
    print("Signal to noise ratio:")
    print("--snr [original_image_filename] [noisy_image_filename] [denoised_image_filename]")
    print("Peak signal to noise ratio:")
    print("--psnr [original_image_filename] [noisy_image_filename] [denoised_image_filename]")
    print("Maximum difference:")
    print("--md [original_image_filename] [noisy_image_filename] [denoised_image_filename]")

if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if (len(sys.argv) == 2) & (sys.argv[1] != '--help'):
    print("Too few command line parameters given.\n")
    sys.exit()

command = sys.argv[1]
param = None
filenamen = None
filenamen2 = None

if len(sys.argv) > 2:
    filenamen = sys.argv[2]
if len(sys.argv) > 3:
    param = sys.argv[3]
if len(sys.argv) > 4:
    filenamen2 = sys.argv[4]


if command == '--brightness':
    doBrightness(int(param),filenamen)
elif command == '--contrast':
    doContrast(int(param),filenamen)
elif command == '--negative':
    doNegative(filenamen)
elif command == '--hflip':
    doHorFlip(filenamen)
elif command == '--vflip':
    doVerFlip(filenamen)
elif command == '--dflip':
    doDiagFlip(filenamen)
elif command == '--enlarge':
    doEnlarge(int(param),filenamen,True)
elif command == '--shrink':
    doEnlarge(int(param),filenamen,False)
elif command == '--mid':
    doMid(filenamen)
elif command == '--amean':
    doAMean(filenamen)
elif command == '--mse':
    doMeanSquarerror(filenamen,param,filenamen2,False)
elif command == '--pmse':
    doMeanSquarerror(filenamen,param,filenamen2,True)
elif command == '--snr':
    doSignalNoiseRatio(filenamen,param,filenamen2,False)
elif command == '--psnr':
    doSignalNoiseRatio(filenamen,param,filenamen2,True)
elif command == '--md':
    doMaxDiff(filenamen,param,filenamen2)
elif command == '--help':
    doHelp()
else:
    print("Unknown command: " + command)
print("")
