import pandas as pd
from sqlalchemy import create_engine
from scipy.spatial import distance
import itertools
from multiprocessing import Pool

engine = create_engine(
    "postgresql+psycopg2://admin:admin@localhost:5432/postgres")


df = pd.DataFrame(pd.read_pickle("reduced_features.pkl")).T
eti = pd.read_sql("select eti_list from labels_list", engine)

df = df[~df.index.isin(eti.eti_list)]
print(df.shape)


def cosine(a, b):
    return (distance.cosine(a, b)-1) * -1


def parallel_cosine(x, y):
    return x, y, cosine(df.loc[x, :], df.loc[y, :])


with Pool() as p:
    l = list(p.starmap(parallel_cosine,
             itertools.combinations(df.index, 2)))

# pd.DataFrame(l, columns=["x", "y", "cosine"]).to_sql(
#     "DCT_noetiquette", engine, if_exists='replace')

pd.DataFrame(l, columns=["x", "y", "cosine"]).to_csv(
    "DCT_noetiquette.csv")
