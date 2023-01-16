from pyphashml.phashml import phashmlctx
from pyphashml.phashml import phashml_distance

import os
import itertools
import pandas as pd

im = os.listdir("./images/")

hashes = [phashmlctx.image_hash("./images/" + ima) for ima in im]
pd.DataFrame({"file": im, "hashes": hashes}).to_csv('hashes.csv')
print((len(im)**2 - len(im)) / 2)
df = pd.DataFrame({"x": [], "y": [], "d": []})
for n, (x, y) in enumerate(itertools.combinations(hashes, 2)):

    d = (phashml_distance(x, y))
    df.append({"x": x, "y": y, "d": d}, ignore_index=True)

    if n % 50 == 0:
        print(n)

df.to_csv("phashmldist.csv")
