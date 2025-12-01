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

#def doDilationTwo(arr,x,y,iter):
    arr = arr > 0
    SE = np.ones((3, 3), dtype=np.uint8)
    out = np.copy(arr)
    #print(iter)
    for i in range(max(1,x-iter), min(511,x+1+iter)):
        for j in range(max(1,y-iter), min(511,y+1+iter)):
            if (i==x-iter or i==x+iter or j==y-iter or j==y+iter):
                region = arr[i-1:i+2, j-1:j+2]
                #print(f"{j},{i}")
                if np.any(region & SE):
                    out[i,j] = 1
    return out

def doDilationFrontier(arr, frontier):
    new_pixels = set()
    y, x = arr.shape

    for (i, j) in frontier:
        for k in range(max(0, i-1), min(y, i+2)):
            for l in range(max(0, j-1), min(x, j+2)):
                if not arr[k, l]:
                    new_pixels.add((k, l))

    return new_pixels

def doDilationFrontierFlex(arr, frontier):
    SE = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

    new_pixels = set()

    height, width = arr.shape

    for (cx, cy) in frontier:

        for k in range(-1, 2):
            for l in range(-1, 2):


                if SE[k + 1, l + 1] == 0:
                    continue

                nx = cx + l
                ny = cy + k

                if 0 <= nx < width and 0 <= ny < height:
                    if not arr[ny, nx]:
                        new_pixels.add((nx, ny))

    return new_pixels

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
    SEh = np.array([[1, 0, 0],
                    [1, 0, 0],
                    [1, 1, 1]], dtype=bool)

    SEm = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], dtype=bool)

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

#def doEmThree(filename,x,y):
    print(f"Function doM3 invoked for {filename}")
    arr = makeArray(filename)
    arr = arr > 0
    arrnew = np.zeros_like(arr)
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

def doEmThreeO(filename, x, y):
    print(f"Function doM3O invoked for {filename}")
    arr = makeArray(filename) > 0
    arrnew = np.zeros_like(arr, dtype=bool)
    arrnew[x, y] = True

    frontier = {(x, y)}

    while frontier:
        new_frontier = doDilationFrontier(arrnew, frontier)

        new_frontier = {p for p in new_frontier if arr[p]}

        for p in new_frontier:
            arrnew[p] = True

        frontier = new_frontier

    makeImage(arrnew, f"emThree_{filename}")


def doEmThreeCheckEd(arr, x1, y1, x2, y2):
    check_mask = np.zeros_like(arr, dtype=bool)

    if not arr[y1, x1]:
        return False

    check_mask[y1, x1] = True

    frontier = {(x1, y1)}

    while frontier:
        new_frontier = doDilationFrontierFlex(check_mask, frontier)

        valid_frontier = set()

        for p in new_frontier:
            px, py = p

            if arr[py, px] and not check_mask[py, px]:
                valid_frontier.add(p)

        for p in valid_frontier:
            px, py = p
            check_mask[py, px] = True  # Set [y, x]

        frontier = valid_frontier

    if check_mask[y2, x2]:
        return True
    else:
        return False


#def doRegionGrowingMultiSeeds(filename):
    print(f"Function doRegionGrowing invoked for {filename}")
    arr = makeArray(filename) > 0
    arrnew = np.zeros_like(arr, dtype=bool)
    frontier = set()

    seed_points = {(400, 200), (300, 500)}

    for (y, x) in seed_points:
        arrnew[x, y] = True
        frontier.add((x, y))

    while frontier:
        new_frontier = doDilationFrontierFlex(arrnew, frontier)
        new_frontier = {p for p in new_frontier if arr[p] and not arrnew[p]}

        for p in new_frontier:
            arrnew[p] = True

        frontier = new_frontier

    makeImage(arrnew, f"regionGrown_{filename}")


def doRegionGrowingMultiSeedsCheck(filename):
    print(f"Function doRegionGrowing invoked for {filename}")

    arr = makeArray(filename) > 0
    arrnew = np.zeros_like(arr, dtype=bool)

    s1x = 100
    s1y = 100
    s2x = 450
    s2y = 450
    seed_points = [(s1x, s1y), (s2x, s2y)]


    frontier = set()

    for (x, y) in seed_points:

        if 0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]:
            arrnew[y, x] = True
            frontier.add((x, y))
    while frontier:

        new_frontier = doDilationFrontierFlex(arrnew, frontier)

        valid_frontier = set()
        for (nx, ny) in new_frontier:
            if arr[ny, nx] and not arrnew[ny, nx]:
                valid_frontier.add((nx, ny))

        for (nx, ny) in valid_frontier:
            arrnew[ny, nx] = True


        frontier = valid_frontier

    makeImage(arrnew, f"regionGrownCh_{filename}")

    merged = doEmThreeCheckEd(arrnew, s1x, s1y, s2x, s2y)

    if merged:
        print("The regions merged")
    else:
        print("The regions cannot merge")

    return merged

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
#elif command == '--m3':
    #doEmThree(filenamen,int(param2),int(param))
elif command == '--m3o':
    doEmThreeO(filenamen,int(param2),int(param))
#elif command == '--regrow':
    #doRegionGrowingMultiSeeds(filenamen)
elif command == '--regrow2':
    doRegionGrowingMultiSeedsCheck(filenamen)
