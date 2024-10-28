import lancedb
from lancedb.pydantic import vector, LanceModel


class CatsAndDogs(LanceModel):
    vector: vector(2)
    species: str
    breed: str
    weight: float


db = lancedb.connect("~/.lancedb")
table = db.create_table("cats_and_dogs", schema=CatsAndDogs, mode="create", exist_ok=True)

data_cats = [
    CatsAndDogs(
        vector=[1., 0.],
        species="cat",
        breed="shorthair",
        weight=12.,
    ),
    CatsAndDogs(
        vector=[-1., 0.],
        species="cat",
        breed="himalayan",
        weight=9.5,
    ),
]

table.add([dict(d) for d in data_cats])

table.head().to_pandas()

data_dogs = [
    CatsAndDogs(
        vector=[0., 10.],
        species="dog",
        breed="samoyed",
        weight=47.5,
    ),
    CatsAndDogs(
        vector=[0, -1.],
        species="dog",
        breed="corgi",
        weight=26.,
    )
]

table.add([dict(d) for d in data_dogs])
table.search([10.5, 10.]).limit(2).to_pandas()

result = table.search([10.5, 10.]).metric("cosine").limit(1).to_pandas()
print(result)
