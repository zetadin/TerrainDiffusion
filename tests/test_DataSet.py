from src.DataSet.DataSet import TiledDataset
from torchvision.transforms import v2

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
    
    # last tile should be 63 and it should have shape (1,256,256)
    assert(ds[63].shape == (1,256,256))

def test_rotation():
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[15,30])
    
    # the should be fewer than 64 tiles per rotation,
    # as some on the edges need to be dropped
    assert(len(ds) < 64*2)
    

import torch
from torchvision.transforms import v2
def test_Transform_and_Normalize():
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[30],
                       transform = v2.Compose([
                            v2.RandomHorizontalFlip(),
                            v2.RandomVerticalFlip(),
                            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                        ]))
    
    
    
    print("Dataset size:", len(ds))

    # get a batch of all the tiles and figure out the mean and standard deviation
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_tiles = next(iter(dataloader))
    mean = torch.mean(all_tiles).item()
    std = torch.std(all_tiles).item()

    print("mean:", torch.mean(all_tiles))
    print("std:", torch.std(all_tiles))

    # what should the std be?
    target_std = 0.5

    # add a Normalize transform for production ready data
    ds.transform = v2.Compose([
                            v2.RandomHorizontalFlip(),
                            v2.RandomVerticalFlip(),
                            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                            v2.Normalize(mean=[mean], std=[std/target_std]),
                        ])
    
    # rebuild dataloader and resample the data
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_tiles = next(iter(dataloader))

    print("num tiles:", len(all_tiles))
    print("tiles shape:", all_tiles.shape)
    print("mean:", torch.mean(all_tiles))
    print("std:", torch.std(all_tiles))
    
    assert(torch.abs(torch.mean(all_tiles)-0.0).item() < 0.001)
    assert(torch.abs(torch.std(all_tiles)-target_std).item() < 0.001)    
    