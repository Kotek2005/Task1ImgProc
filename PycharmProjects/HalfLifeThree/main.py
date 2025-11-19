from PIL import Image
import numpy as np
import sys

def makeArray(image):
    im = Image.open(image).convert('1')
    arr = np.array(im.getdata())
    arr = arr.reshape(im.size[1], im.size[0])
    return arr

def makeImage(arr, name):

    import numpy as np
    from PIL import Image

    arr_bool = arr > 0
    newIm = Image.fromarray(arr_bool.astype(np.uint8) * 255).convert("1")
    newIm.save(name)

def doDilation(filename):
    if isinstance(filename, str):
        print(f"Function doDilation invoked for {filename}")
        arr = makeArray(filename)
    else:
        arr = filename
    arrnew = np.pad(arr, 1, "edge")
    SE = np.ones((3, 3), dtype=np.uint8)
    SE = SE*255
    out = np.zeros_like(arr)
    for i in range(1, arrnew.shape[0]-1):
        for j in range(1, arrnew.shape[1]-1):
            region = arrnew[i-1:i+2, j-1:j+2]
            if np.any(region & SE):
                out[i-1,j-1] = 255
            else:
                out[i-1,j-1] = 0
    if isinstance(filename, str):
        makeImage(out, f"dilated_{filename}")
    return out

def doDilationTwo(arr,x,y,iter):
    arrnew = np.pad(arr, 1, "edge")
    arrnew = arrnew > 0
    SE = np.ones((3, 3), dtype=np.uint8)
    out = np.copy(arr)
    for i in range(x-iter, x+1+iter):
        for j in range(y-iter, y+1+iter):
            region = arrnew[i-1:i+2, j-1:j+2]
            if np.any(region & SE):
                out[i-1,j-1] = 1
            else:
                out[i-1,j-1] = 0

    return out

def doErosion(filename):
    if isinstance(filename, str):
        print(f"Function doErosion invoked for {filename}")
        arr = makeArray(filename)
    else:
        arr = filename
    arrnew = np.pad(arr, 1, "edge")
    SE = np.ones((3, 3), dtype=np.uint8)
    SE = SE*255
    out = np.zeros_like(arr)
    for i in range(1, arrnew.shape[0]-1):
        for j in range(1, arrnew.shape[1]-1):
            region = arrnew[i-1:i+2, j-1:j+2]
            if np.all(region & SE):
                out[i-1,j-1] = 255
            else:
                out[i-1,j-1] = 0

    if isinstance(filename, str):
        makeImage(out, f"eroded_{filename}")
    return out

def doOpening(filename):
    print(f"Function doOpening invoked for {filename}")
    eroded = doErosion(filename)
    out = doDilation(eroded)

    makeImage(out, f"opened_{filename}")
    return out

def doClosing(filename):
    print(f"Function doClosing invoked for {filename}")
    dilated = doDilation(filename)
    out = doErosion(dilated)

    makeImage(out, f"closed_{filename}")
    return out

def doHitOrMiss(filename):
    print(f"Function doHitOrMiss invoked for {filename}")
    arr = makeArray(filename)
    arrbool = arr > 0
    arrnew = np.pad(arrbool, 1, "edge")
    outh = np.zeros_like(arr)
    outm = np.zeros_like(arr)
    SEh = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]], dtype=bool)

    SEm = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]], dtype=bool)

    for i in range(1, arrnew.shape[0]-1):
        for j in range(1, arrnew.shape[1]-1):
            region = arrnew[i-1:i+2, j-1:j+2]
            if np.all(region[SEh]):
                outh[i-1,j-1] = 1
            if np.all(~region[SEm]):
                outm[i-1,j-1] = 1

    out = outh & outm
    outim = out*255
    makeImage(outim, f"hit_or_miss_{filename}")
    return out

def doEmThree(filename,x,y):
    print(f"Function doM3 invoked for {filename}")
    arr = makeArray(filename)
    arrnew = arr > 0
    arrnew[x,y] = True
    iter = 1

    while True:

        arrold = arrnew.copy()
        arrdil = doDilationTwo(arrnew,x,y,iter)
        arrnew = arrdil & (arr>0)
        iter += 1
        if np.array_equal(arrnew, arrold):
            break

    print(iter)
    makeImage(arrnew, f"emThree_{filename}")



if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if (len(sys.argv) == 2) & (sys.argv[1] != '--help'):
    print("Too few command line parameters given.\n")
    sys.exit()
param = None
param2 = None

command = sys.argv[1]
filenamen = sys.argv[2]
if len(sys.argv) > 3:
    param = sys.argv[3]
if len(sys.argv) > 4:
    param2 = int(sys.argv[4])

if command == '--dil':
    doDilation(filenamen)
elif command == '--ero':
    doErosion(filenamen)
elif command == '--open':
    doOpening(filenamen)
elif command == '--close':
    doClosing(filenamen)
elif command == '--hmt':
    doHitOrMiss(filenamen)
elif command == '--m3':
    doEmThree(filenamen,int(param),int(param2))

