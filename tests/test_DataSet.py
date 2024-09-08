from src.DataSet.DataSet import TiledDataset

def test_len():
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[0])
    
    # the length of the dataset should be the number of tiles
    assert(len(ds) == len(ds.tiles))
    
    # There should be exactly 64=(2048/256)**2 tiles given the test data
    assert(len(ds) == 64)
    
def test_getitem():
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[0])
    
    # last tile should be 63 and it should have shape (256,256)
    assert(ds[63].shape == (256,256))

def test_rotation():
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[15,30])
    
    # the should be fewer than 64 tiles per rotation,
    # as some on the edges need to be dropped
    assert(len(ds) < 64*2)
    
    