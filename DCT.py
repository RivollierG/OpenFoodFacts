import pandas as pd
from scipy.spatial import distance
import itertools
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://admin:admin@localhost:5432/postgres")

# sudo docker run -it -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin -p 5432:5432 -v postgres:/var/lib/postgresql/data postgres


def cosine(a, b):
    return (distance.cosine(a, b)-1) * -1


def parallel_cosine(x, y):
    return x, y, cosine(df.loc[x, :], df.loc[y, :])


df = pd.DataFrame(pd.read_pickle("reduced_features.pkl")).T

# print(cosine(df.iloc[0, :], df.iloc[1, :]))

# print(cosine([0, 1], [0, -1]))

# l = [[x, y, cosine(df.loc[x, :], df.loc[y, :])]
#      for x, y in itertools.combinations(df.index[:1000], 2)]


with Pool() as p:
    l = list(p.starmap(parallel_cosine,
             itertools.combinations(df.index, 2)))

pd.DataFrame(l, columns=["x", "y", "cosine"]).to_sql(
    "DCT", engine, if_exists='replace')
