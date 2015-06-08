

def load_ids(fname):
    raw = open(fname).readlines()
    ids = map(lambda x: int(x), raw)
    return ids

allids = load_ids("../ImageId.txt")
trainids = load_ids("trainid.txt")
testids = load_ids("testid.txt")
dataids = load_ids("databaseid.txt")

ddict = dict(zip(allids, range(len(allids))))

files = [
"Labels_animal.txt",
"Labels_buildings.txt",
"Labels_clouds.txt",
"Labels_grass.txt",
"Labels_lake.txt",
"Labels_person.txt",
"Labels_plants.txt",
"Labels_sky.txt",
"Labels_water.txt",
"Labels_window.txt",
]

cx = []
for name in files:
    cx.append(load_ids("./../Concept/" + name))

def calc_relation(i,j):
    i,j = ddict[i], ddict[j]
    for k in range(len(cx)):
        if (cx[k][i] == 1 and cx[k][j] == 1):
            return 1
    return 0

for i in range(10):
    xx = [calc_relation(testids[i], did) for did in dataids]
    print reduce(lambda x,y: x+y, xx, 0)

def is_label(i,j): return cx[j][i]
