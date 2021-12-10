import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairdata", type=str)
    args = parser.parse_args()

    for p in Path(args.pairdata).iterdir():
        if p.is_file(): continue
        cleans = sorted([str(p) for p in p.glob("description/clean/[a-z]*.pkl")])
        images = sorted([str(p) for p in p.glob("train_number[1-3]/*.jpg")])
        tiled_cleans = sorted(cleans * len(images)) #sph1, sph1, ..., sph1, sph2, sph2, ...
        tiled_images = images * len(cleans)         #img1, img2, ..., imgN, img1, img2, ...
        assert len(cleans) == 4, "mismatch in len(cleans)"
        assert len(images) == 90, "mismatch in len(images)"
        for a, i in zip(tiled_cleans, tiled_images):
            print(a, i)