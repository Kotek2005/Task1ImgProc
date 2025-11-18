from PIL import Image
import numpy as np
import sys

def makeArray(image):
    im = Image.open(image)
    arr = np.array(im.getdata())
    arr = arr.reshape(im.size[1], im.size[0])
    return arr

def makeImage(arr,name):
    newIm = Image.fromarray(arr.astype(np.uint8))
    newIm.save(name)

def doDilation(filename):
    print(f"Function doDilation invoked for {filename}")
    arr = makeArray(filename)
    arrnew = np.pad(arr, 1, "edge")
    SE = np.ones((3, 3), dtype=np.uint8)
    out = np.zeros_like(arr)





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

if command == '--dilation':
    doDilation(filenamen)