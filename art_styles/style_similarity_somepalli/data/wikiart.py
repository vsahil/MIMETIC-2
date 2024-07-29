import pathlib
import os
import sys
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import vaex as vx
import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))


class WikiArt(object):
    def __init__(self, root_dir):
        assert osp.exists(osp.join(root_dir, 'wikiart.csv'))
        self.root_dir = root_dir
        annotations = vx.from_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(set(annotations[annotations['split'] == 'database']['artist'].tolist()))
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.query_images = temprepo[temprepo['split'] == 'query']['name'].tolist()
        self.val_images = temprepo[temprepo['split'] == 'database']['name'].tolist()
        self.query_db = annotations[annotations['name'].isin(self.query_images)]
        self.val_db = annotations[annotations['name'].isin(self.val_images)]
        self.query_db['name'] = self.query_db['name'].apply(lambda x: '.'.join(x.split('.')[:-1]))
        self.val_db['name'] = self.val_db['name'].apply(lambda x: '.'.join(x.split('.')[:-1]))

    def get_query_col(self, col):
        return np.asarray(self.query_db[col].tolist())

    def get_val_col(self, col):
        return np.asarray(self.val_db[col].tolist())


class WikiArtD(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        assert osp.exists(osp.join(root_dir, 'wikiart.csv'))
        annotations = vx.from_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(set(annotations[annotations['split'] == 'database']['artist'].tolist()))
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.pathlist = temprepo[temprepo['split'] == split]['fps'].tolist()
        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]  # os.path.join(self.root_dir, self.split,self.artists[idx] ,self.pathlist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx


class SelectedWikiArt(Dataset):
    def __init__(self, args, artist_name, transform=None, generated_images_root=None):
        self.transform = transform

        if args.which_images == 'wikiart_images':
            from unidecode import unidecode
            replacement_list = {'apollinary goravsky': 'apollinariy-goravskiy', 'petro kholodny': 'petro-kholodny-elder', 'alexei korzukhin': 'aleksey-ivanovich-korzukhin', 'jérôme-martin langlois': 'jerome-martin-langlois'}
            if artist_name in replacement_list:
                artist_name_downloaded_folder = replacement_list[artist_name]
            else:
                artist_name_downloaded_folder = unidecode(artist_name.strip().lower().replace("'", ' ').replace(".", ' ').replace('   ', '-').replace('   ', '-').replace('  ', '-').replace(' ', '-'))
            downloaded_images_folder = f"/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/wikiart_images_downloaded/{artist_name_downloaded_folder}"
            
            ## take all the images inside the downloaded_images_folder, these must end with .jpg
            images = [f for f in os.listdir(downloaded_images_folder) if f.endswith(".jpg")]
            assert len(images) >= 8, f"artist: {artist_name}, len(images): {len(images)}"
            ## now get all images of this person and store them in a list, the image name is computed by combining downloaded_images_folder / name
            self.pathlist = [f"{downloaded_images_folder}/{name}" for name in images]
            self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))
        
        elif args.which_images == "generated_images":
            generated_images_folder = f"{generated_images_root}/generated_images_wikiart_artists/{artist_name}/{args.image_generation_prompt}"
            ## take all the images that end with .png except grid_images.png
            images = [f for f in os.listdir(generated_images_folder) if f.endswith(".png") and f != "grid_images.png"]
            assert len(images) == 200, f"artist: {artist_name}, len(images): {len(images)}"
            ## now get all images of this person and store them in a list, the image name is computed by combining downloaded_images_folder / name
            self.pathlist = [f"{generated_images_folder}/{name}" for name in images]
            self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))
        
        elif args.which_images == "laion_images":
            laion_images_folder = f"/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images/{artist_name}"
            ## take all the files inside the images folder. They are already filtered to be images, but with various extensions, except .csv
            images = [f for f in os.listdir(laion_images_folder) if not f.endswith(".csv") and not f.endswith(".txt") and not f.endswith(".json") and not f.endswith(".npy")]
            ## we cannot say anything about the length of images, as it can be any number upto 100K
            assert len(images) <= 100000, f"artist: {artist_name}, len(images): {len(images)}"
            ## now get all images of this person and store them in a list, the image name is computed by combining downloaded_images_folder / name
            self.pathlist = [f"{laion_images_folder}/{name}" for name in images]
            self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))
        
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]  # os.path.join(self.root_dir, self.split,self.artists[idx] ,self.pathlist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx

class WikiArtTrain(Dataset):
    def __init__(self, root_dir, split='database', transform=None, maxsize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        assert os.path.exists(os.path.join(root_dir, 'wikiart.csv'))
        annotations = pd.read_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(
            set(annotations[annotations['split'] == 'database']['artist'].tolist())
        )
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.pathlist = temprepo[temprepo['split'] == split]['fps'].tolist()
        self.labels = temprepo[temprepo['split'] == split]['artist'].tolist()

        self.artist_to_index = {artist: i for i, artist in enumerate(acceptable_artists)}
        self.index_to_artist = acceptable_artists

        # Convert labels to one-hot
        self.labels = list(map(lambda x: self.artist_to_index[x], self.labels))
        self.labels = np.eye(len(acceptable_artists))[self.labels].astype(bool)
        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

        # Select maxsize number of images
        if maxsize is not None:
            ind = np.random.randint(0, len(self.namelist), maxsize)
            self.namelist = [self.namelist[i] for i in ind]
            self.pathlist = [self.pathlist[i] for i in ind]
            self.labels = self.labels[ind]

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):

        img_loc = self.pathlist[idx]
        image = Image.open(img_loc).convert("RGB")

        if self.transform:
            images = self.transform(image)

        artist = self.labels[idx]
        return images, artist, idx
