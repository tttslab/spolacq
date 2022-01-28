import argparse
from glob import glob

import yaml


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str)
    parser.add_argument("audio_image_pair", type=str, help="Audio-image pairs used to train front-end.")
    parser.add_argument("train_imgs", type=str, help="Images used in observation phase.")
    parser.add_argument("test_imgs", type=str, help="Images used in dialogue phase.")
    parser.add_argument("conf", type=str)
    args = parser.parse_args()
    args = update_args(args.conf, args)
    
    with open(args.audio_image_pair, "w") as f:
        for food in args.obj_name_list:
            cleans = sorted(glob(f"{args.datadir}/{food}/description/clean/[a-z]*.pkl"))
            images = sorted(glob(f"{args.datadir}/{food}/train_number[1-3]/*.jpg"))
            tiled_cleans = sorted(cleans * len(images)) #sph1, sph1, ..., sph1, sph2, sph2, ...
            tiled_images = images * len(cleans)         #img1, img2, ..., imgN, img1, img2, ...
            assert len(cleans) == 4, "mismatch in len(cleans)"
            assert len(images) == 90, "mismatch in len(images)"
            for a, i in zip(tiled_cleans, tiled_images):
                f.write(f"{a} {i}\n")
    
    with open(args.train_imgs, "w") as f:
        for food in args.obj_name_list:
            image_paths = sorted(glob(f"{args.datadir}/{food}/train_number[1-3]/*.jpg"))
            assert len(image_paths) == 90, "mismatch in test image dataset"
            f.write("\n".join(image_paths))
            f.write("\n")
    
    with open(args.test_imgs, "w") as f:
        for food in args.obj_name_list:
            image_paths = sorted(glob(f"{args.datadir}/{food}/test_number[1-3]/*.jpg"))
            assert len(image_paths) == 30, "mismatch in test image dataset"
            f.write("\n".join(image_paths))
            f.write("\n")