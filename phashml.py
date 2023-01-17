from pyphashml.phashml import phashmlctx
from pyphashml.phashml import phashml_distance

import os
import itertools
import pandas as pd
from bitstring import BitArray


def hdist(a, b):
    return str(a**b).count(1)


print(phashml_distance(phashmlctx.image_hash("./images/" + "0044500073225_1.400.jpg"),
      phashmlctx.image_hash("./images/" + "0017082876362_2.400.jpg")))

im = os.listdir("./images/")
if os.path.isfile("./hashes.csv"):
    hashes = pd.read_csv("./hashes.csv")['hashes']
    hashes = [h for h in hashes[:1000]]
else:

    hashes = [phashmlctx.image_hash("./images/" + ima) for ima in im]
    pd.DataFrame({"file": im, "hashes": hashes}).to_csv('hashes.csv')


dist = [[x, y, phashml_distance(BitArray(x), BitArray(y))]
        for x, y in itertools.combinations(hashes, 2)]

pd.DataFrame(dist).to_csv("phashmldist.csv")
