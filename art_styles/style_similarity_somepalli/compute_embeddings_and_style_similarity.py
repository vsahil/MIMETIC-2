import argparse
import os
import pathlib
import random
import sys
import warnings
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# from torchvision import transforms
# import torchvision.models as torchvision_models
# from torchvision.models import VGG16_Weights

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import utils
# from utils import extract_features_pca
# from models import dino_vits, moco_vits

best_acc1 = 0
prompts = [("a painting in the style of ", 1), ("an artwork in the style of ", 1), ("a sketch in the style of ", 1), ("a fine art piece in the style of ", 1),  ("an illustration in the style of ", 1)]


def compute_image_style_embeddings(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # utils.init_distributed_mode(args)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # compute_image_style_embeddings_worker process function
        mp.spawn(compute_image_style_embeddings_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call compute_image_style_embeddings_worker function
        compute_image_style_embeddings_worker(args.gpu, ngpus_per_node, args)


def compute_image_style_embeddings_worker(gpu, ngpus_per_node, args):
    from data.wikiart import SelectedWikiArt
    global best_acc1
    args.gpu = gpu
    assert args.gpu is not None
    print("Use GPU: {} for training".format(args.gpu))
    
    assert args.pt_style.startswith('csd')
    
    assert args.model_path is not None, "Model path missing for CSD model"
    from CSD.model import CSD_CLIP
    from CSD.utils import has_batchnorms, convert_state_dict
    from CSD.loss_utils import transforms_branch0

    args.content_proj_head = "default"
    model = CSD_CLIP(args.arch, args.content_proj_head)
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint with msg {msg}")
    preprocess = transforms_branch0
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    ## get the GPu type
    gpu_type = torch.cuda.get_device_name(args.gpu)
    print(f"GPU type: {gpu_type}")
    if "A100" in gpu_type:
        args.batch_size = 64 * 16
    elif "A40" in gpu_type or "L40" in gpu_type or "L40s" in gpu_type:
        args.batch_size = 64 * 8
    elif "RTX" in gpu_type:
        args.batch_size = 64 * 4
    else:
        args.batch_size = 64 * 2
    
    print(f"Batch size: {args.batch_size}")
    
    ret_transform = preprocess
    
    # artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv")   ## artist_name,counts_in_laion2b-en,counts_in_wikiart,count_paintings
    if args.art_group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.art_group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError

    artists = artist_df['artist_name'].tolist()[::-1]
    ## randomly shuffle the artists
    # random.shuffle(artists)
    model.eval()

    for artist in artists:
        os.makedirs(args.embed_dir, exist_ok=True)
        # embsavepath = os.path.join(args.embed_dir, f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{str(args.layer)}', f'{artist}')
        if args.which_images == 'wikiart_images':
            embsavepath = os.path.join(args.embed_dir, f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        elif args.which_images == 'generated_images':
            embsavepath = os.path.join(args.embed_dir, f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}', args.image_generation_prompt)
        elif args.which_images == 'laion_images':
            embsavepath = os.path.join(args.embed_dir, f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        else:
            raise ValueError(f"Invalid value for which_images: {args.which_images}")
        
        if os.path.isfile(os.path.join(embsavepath, 'embeddings_0.pkl')) or args.skip_val:
            valexist = True
        else:
            valexist = False
        
        if valexist:
            print(f'Embeddings for {artist} already exist at {embsavepath}')
            continue
        
        ## construct a data loader for this artists and compute the embeddings using the model, and save it in the wikiart_images_embeddings folder
        generated_images_root = None
        if args.which_images == 'generated_images':
            if args.stable_diffusion_version == "1":
                generated_images_root = "/gscratch/h2lab/vsahil/vlm-efficiency"
            elif args.stable_diffusion_version == "5":
                generated_images_root = "/gscratch/h2lab/vsahil/vlm_efficiency_second_model/vlm-efficiency"
            elif args.stable_diffusion_version == "2.1":
                generated_images_root = "/gscratch/h2lab/vsahil/vlm_efficiency_stable_diffusion2/vlm-efficiency/"
            else:
                raise ValueError(f"Invalid value for stable_diffusion_version: {args.stable_diffusion_version}")
        
        dataset = SelectedWikiArt(args, artist, ret_transform, generated_images_root=generated_images_root)
        
        print(f"artist: {artist} / {len(dataset)} imgs")
        
        if len(dataset) == 0:
            print(f"No images found for {artist}")
            continue
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        from CSD.utils import extract_features
        values_features = extract_features(model, data_loader, use_cuda=True, use_fp16=True, eval_embed=args.eval_embed)
        from search.embeddings import save_chunk
        # l_query_features = list(np.asarray(query_features.cpu().detach(), dtype=np.float16))
        # save_chunk(l_query_features, dataset_query.namelist, 0, f'{embsavepath}/{args.qsplit}')
        l_values_features = list(np.asarray(values_features.cpu().detach(), dtype=np.float16))
        # save_chunk(l_values_features, dataset_values.namelist, 0, f'{embsavepath}/database')
        save_chunk(l_values_features, dataset.namelist, 0, f'{embsavepath}')
        print(f'Embeddings for {artist} saved at {embsavepath} that has {len(l_values_features)} images')


def compute_similarity_between_art_images_of_same_artist(args):
    # artist_df = pd.read_csv("artists_to_analyze.csv")   ## artist_name,counts_in_laion2b-en,counts_in_wikiart,count_paintings
    if args.art_group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.art_group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError
    
    artists = artist_df['artist_name'].tolist()
    
    cosine_similarities = {}
    
    for artist in artists:
        embsavepath = os.path.join('wikiart_images_embeddings', f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        assert os.path.isfile(os.path.join(embsavepath, 'embeddings_0.pkl')), f"Embeddings for {artist} do not exist at {embsavepath}"
        ## load the embeddings and compute the similarity between the images of the same artist
        from search.embeddings import Embeddings
        emb = Embeddings(embsavepath, embsavepath, file_ext='.pkl', chunked=True, chunk_size=5000)
        this_artist_embeddings = emb.embeddings
        this_artist_embeddings = this_artist_embeddings / (np.linalg.norm(this_artist_embeddings, axis=1, keepdims=True, ord=2) + 1e-16)
        # assert that the embeddings have norm of 1 along the axis 1
        assert np.all(np.abs(np.linalg.norm(this_artist_embeddings, axis=1, ord=2) - 1) < 1e-6), f"max {np.max(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} min {np.min(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} for {artist}"
        cosine_similarity_between_images_of_same_artist = np.dot(this_artist_embeddings, this_artist_embeddings.T)
        ## assert that diagonal will be close to 1
        if artist not in ['joseph wright', 'janet fish', 'edward hopper']:
            assert np.all(np.abs(np.diag(cosine_similarity_between_images_of_same_artist) - 1) < 1e-6), f"max {np.max(np.diag(cosine_similarity_between_images_of_same_artist) - 1)} min {np.min(np.diag(cosine_similarity_between_images_of_same_artist) - 1)} for {artist}"
            assert np.all(cosine_similarity_between_images_of_same_artist >= -1 - 1e-8) and np.all(cosine_similarity_between_images_of_same_artist <= 1 + 1e-8), f"max {np.max(cosine_similarity_between_images_of_same_artist)} min {np.min(cosine_similarity_between_images_of_same_artist)} for {artist}"
        print(f"The avg cosine similarity between the images of the same artist {artist} is {np.mean(cosine_similarity_between_images_of_same_artist)}")
        cosine_similarities[artist] = np.mean(cosine_similarity_between_images_of_same_artist)
    
    ## save the cosine similarities in a csv file
    import csv
    with open(f"cosine_similarities_images_same_artist_{args.art_group}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "avg_cosine_similarity"])
        for artist, avg_cosine_similarity in cosine_similarities.items():
            writer.writerow([artist, avg_cosine_similarity])
    
    avg_cosine_similarities_same_artist = list(cosine_similarities.values())
    ## make a histogram of the cosine similarities for all artists in one plot. 
    import matplotlib.pyplot as plt
    plt.hist(avg_cosine_similarities_same_artist, bins=50)
    plt.xlabel("Average Cosine Similarity images of same artist", fontsize=12)
    plt.ylabel("Number of Art Styles", fontsize=12)
    ## add the minimum and maximum values of the cosine similarity as texts on the top left of the plot
    plt.text(0.55, 22.5, f"Min similarity: {round(np.min(avg_cosine_similarities_same_artist), 2)}", fontsize=12)
    plt.text(0.55, 21, f"Max similarity: {round(np.max(avg_cosine_similarities_same_artist), 2)}", fontsize=12)
    # plt.title("Average Cosine Similarity for the ")
    plt.tight_layout()
    plt.savefig(f"histogram_avg_cosine_similarity_images_same_artist_{args.art_group}.png")


def compute_similarity_between_art_images_of_different_artists(args):
    ## we got the positive examples of the same art images in the previous function. Now we need to get the negative examples of the different art images.
    # artist_df = pd.read_csv("artists_to_analyze.csv")   ## artist_name,counts_in_laion2b-en,count_paintings_in_laion_images,counts_in_wikiart_original_dataset,downloaded_images_from_wikiart_website
    if args.art_group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.art_group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError
    ## randomly sample 30 artists whose number of downloaded_images_from_wikiart_website is greater than 25
    artists = artist_df[artist_df['downloaded_images_from_wikiart_website'] >= 25]['artist_name'].tolist()
    num_artists_to_sample = 150
    artists = random.sample(artists, num_artists_to_sample)
    assert len(artists) >= num_artists_to_sample, f"Expected {num_artists_to_sample} artists, got {len(artists)} artists"
    
    normalized_embeddings = {}
    for artist in artists:
        embsavepath = os.path.join('wikiart_images_embeddings', f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        assert os.path.isfile(os.path.join(embsavepath, 'embeddings_0.pkl')), f"Embeddings for {artist} do not exist at {embsavepath}"
        ## load the embeddings and compute the similarity between the images of the same artist
        from search.embeddings import Embeddings
        emb = Embeddings(embsavepath, embsavepath, file_ext='.pkl', chunked=True, chunk_size=5000)
        this_artist_embeddings = emb.embeddings
        this_artist_embeddings = this_artist_embeddings / (np.linalg.norm(this_artist_embeddings, axis=1, keepdims=True, ord=2) + 1e-16)
        # assert that the embeddings have norm of 1 along the axis 1
        assert np.all(np.abs(np.linalg.norm(this_artist_embeddings, axis=1, ord=2) - 1) < 1e-6), f"max {np.max(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} min {np.min(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} for {artist}"
        normalized_embeddings[artist] = this_artist_embeddings
    
    ## Compute the cosine similarity between the images of each pair of these selected artists -- it will give us a total of 30*29/2 = 435 pairs
    cosine_similarities = {}
    for i in range(len(artists)):
        for j in range(i+1, len(artists)):
            artist1 = artists[i]
            artist2 = artists[j]
            cosine_similarity_between_images_of_different_artists = np.dot(normalized_embeddings[artist1], normalized_embeddings[artist2].T)
            cosine_similarities[(artist1, artist2)] = np.mean(cosine_similarity_between_images_of_different_artists)

    assert len(cosine_similarities) == num_artists_to_sample * (num_artists_to_sample - 1) // 2, f"Expected {num_artists_to_sample * (num_artists_to_sample - 1) // 2} pairs, got {len(cosine_similarities)} pairs"
    
    ## save the cosine similarities in a csv file
    import csv
    with open(f"cosine_similarities_images_different_artists_{args.art_group}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["artist1", "artist2", "avg_cosine_similarity"])
        for (artist1, artist2), avg_cosine_similarity in cosine_similarities.items():
            writer.writerow([artist1, artist2, avg_cosine_similarity])
    
    print(f"Saved the cosine similarities between the images of different artists in cosine_similarities_images_different_artists_{args.art_group}.csv")


def investigate_cosine_similarity_and_find_threshold(args):
    # df = pd.read_csv("cosine_similarities_images_same_artist.csv")
    df = pd.read_csv(f"cosine_similarities_images_same_artist_{args.art_group}.csv")
    different_artist_cosine_sim = pd.read_csv(f"cosine_similarities_images_different_artists_{args.art_group}.csv")
    ## print the name of all artists whose cosine similarity is less than 0.25
    df = df[df['avg_cosine_similarity'] < -2]
    for artist in df['artist']:
        embsavepath = os.path.join('wikiart_images_embeddings', f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        assert os.path.isfile(os.path.join(embsavepath, 'embeddings_0.pkl')), f"Embeddings for {artist} do not exist at {embsavepath}"
        ## load the embeddings and compute the similarity between the images of the same artist
        from search.embeddings import Embeddings
        emb = Embeddings(embsavepath, embsavepath, file_ext='.pkl', chunked=True, chunk_size=5000)
        this_artist_embeddings = emb.embeddings
        this_artist_embeddings = this_artist_embeddings / (np.linalg.norm(this_artist_embeddings, axis=1, keepdims=True, ord=2) + 1e-16)
        # assert that the embeddings have norm of 1 along the axis 1
        assert np.all(np.abs(np.linalg.norm(this_artist_embeddings, axis=1, ord=2) - 1) < 1e-6), f"max {np.max(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} min {np.min(np.linalg.norm(this_artist_embeddings, axis=1, ord=2)) - 1} for {artist}"
        cosine_similarity_between_images_of_same_artist = np.dot(this_artist_embeddings, this_artist_embeddings.T)
        ## assert that diagonal will be close to 1
        if artist not in ['joseph wright']: #, 'janet fish', 'edward hopper']:
            assert np.all(np.abs(np.diag(cosine_similarity_between_images_of_same_artist) - 1) < 1e-6), f"max {np.max(np.diag(cosine_similarity_between_images_of_same_artist) - 1)} min {np.min(np.diag(cosine_similarity_between_images_of_same_artist) - 1)} for {artist}"
            assert np.all(cosine_similarity_between_images_of_same_artist >= -1 - 1e-8) and np.all(cosine_similarity_between_images_of_same_artist <= 1 + 1e-8), f"max {np.max(cosine_similarity_between_images_of_same_artist)} min {np.min(cosine_similarity_between_images_of_same_artist)} for {artist}"
        print(f"The avg cosine similarity between the images of the same artist {artist} is {np.mean(cosine_similarity_between_images_of_same_artist)}")
        print(cosine_similarity_between_images_of_same_artist.shape)
        continue
        import matplotlib.pyplot as plt
        # Create the plot
        fig, ax = plt.subplots()
        cax = ax.matshow(cosine_similarity_between_images_of_same_artist, cmap='viridis')  # Choose colormap, e.g., 'viridis', 'plasma', 'inferno', 'magma'

        # Adding color bar
        fig.colorbar(cax)

        # Adding titles and labels
        plt.title('Cosine Similarity Matrix')
        ax.set_xlabel('Item Index')
        ax.set_ylabel('Item Index')
        plt.savefig(f"cosine_similarity_matrix_{artist}.png")
    
    # different_artist_cosine_sim = pd.read_csv("cosine_similarities_images_different_artist.csv")
    avg_cosine_similarities_different_artists = different_artist_cosine_sim['avg_cosine_similarity'].tolist()
    ## for the cosine similarities of different artists filter out the ones that are less than 0.2
    # avg_cosine_similarities_different_artists = [x for x in avg_cosine_similarities_different_artists if x > 0.2]
    cosine_sims = pd.read_csv(f"cosine_similarities_images_same_artist_{args.art_group}.csv")
    avg_cosine_similarities_same_artist = cosine_sims['avg_cosine_similarity'].tolist()
    ## since we want to make a histogram, we want the two lists to have almost the same length, to do that we will just repeat avg_cosine_similarities_same_artist 5 times. Make sure to not multiply the elements by 5, but to just repeat the list 5 times.
    avg_cosine_similarities_same_artist = avg_cosine_similarities_same_artist * (len(avg_cosine_similarities_different_artists) // len(avg_cosine_similarities_same_artist))
    avg_cosine_similarities_different_artists = np.array(avg_cosine_similarities_different_artists)
    avg_cosine_similarities_same_artist = np.array(avg_cosine_similarities_same_artist)
    print(len(avg_cosine_similarities_different_artists), len(avg_cosine_similarities_same_artist))
    
    from sklearn.metrics import roc_curve
    scores = np.concatenate((avg_cosine_similarities_different_artists, avg_cosine_similarities_same_artist))
    true_labels = np.concatenate( ( np.zeros(avg_cosine_similarities_different_artists.shape[0]), np.ones(avg_cosine_similarities_same_artist.shape[0]) ) )
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    
    ## ## find the threshold that maximizes the F1 score
    f1_score = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr))
    # for threshold_index in range(400, 500):
    #     print(f"Threshold: {thresholds[threshold_index]}, FPR: {fpr[threshold_index]}, TPR: {tpr[threshold_index]}, F1 score: {f1_score[threshold_index]}")
        
    threshold_for_max_f1_score = thresholds[np.argmax(f1_score)]
    print(f"Threshold for max F1 score: {threshold_for_max_f1_score}")
    ## at this threshold, print the FPR and TPR
    threshold_index = np.where(thresholds == threshold_for_max_f1_score)[0][0]
    print(f"Threshold: {thresholds[threshold_index]}, FPR: {fpr[threshold_index]}, TPR: {tpr[threshold_index]}, F1 score: {f1_score[threshold_index]}")
    
    import matplotlib.pyplot as plt
    ## set figure size
    plt.figure(figsize=(10, 7))
    bins = 50
    plt.hist(avg_cosine_similarities_same_artist, bins=bins, color='b', alpha=0.8, label='Images by same artist') 
    if args.art_group == "artist_group1":
        plt.hist(np.repeat(avg_cosine_similarities_different_artists, 1), bins=bins, color='r', alpha=0.8, label='Images by different artists')     ## repeated to make it similar size as avg_cosiine_similarities_same_artist
    else:
        plt.hist(np.repeat(avg_cosine_similarities_different_artists, 2), bins=bins, color='r', alpha=0.8, label='Images by different artists')
    ## add a vertical black line at the threshold = 0.1849365234375. Also add the point on the x-axis where the threshold is.
    plt.axvline(x=threshold_for_max_f1_score, color='k', linestyle='--', label='Threshold', ymax=0.9)
    ## label the point where the threshld line touches the x-axis -- this is the point where the threshold is.
    if args.art_group == "artist_group1":
        plt.text(threshold_for_max_f1_score-0.08, 730, f"Threshold: {round(threshold_for_max_f1_score, 3)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
    else:
        plt.text(threshold_for_max_f1_score-0.03, 1150, f"Threshold: {round(threshold_for_max_f1_score, 3)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
    plt.xlabel("Average Cosine Similarity", fontsize=20)
    plt.ylabel("Number of Art Styles", fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"wikiart_images/histogram_avg_cosine_similarity_images_of_same_and_different_artist_{args.art_group}.pdf")

    # import ipdb; ipdb.set_trace()
    ## get the pair of (different) artists whose cosine similarity is higher than the threshold
    # different_artist_cosine_sim = different_artist_cosine_sim[different_artist_cosine_sim['avg_cosine_similarity'] > threshold_for_max_f1_score]
    # print(f"Number of pairs of different artists whose cosine similarity is higher than the threshold {threshold_for_max_f1_score}: {different_artist_cosine_sim.shape[0]}")
    # ## sort different_artist_cosine_sim by descending order of avg_cosine_similarity
    # different_artist_cosine_sim = different_artist_cosine_sim.sort_values(by='avg_cosine_similarity', ascending=False)
    # ## print the top 5
    # print(different_artist_cosine_sim.head(5))
    
    # ## get the pair of (same) artists whose cosine similarity is lower than the threshold
    # cosine_sims = cosine_sims[cosine_sims['avg_cosine_similarity'] < threshold_for_max_f1_score]
    # print(f"Number of pairs of same artists whose cosine similarity is lower than the threshold {threshold_for_max_f1_score}: {cosine_sims.shape[0]}")
    # print(cosine_sims['artist'].tolist())


def count_painting_images_for_artists(args):
    artist_list_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv")
    painting_embedding_threshold = 0.1849365234375
    for artist in artist_list_df['artist_name'].tolist():
        total_number_captions = artist_list_df[artist_list_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]
        label_similarities = os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images", artist, f"text_probs_all_artists_images_{artist}.npy")
        num_paintings = artist_list_df[artist_list_df['artist_name'] == artist]['count_paintings_in_laion_images'].values[0]
        if not os.path.exists(label_similarities):
            print(f"Label similarities not found for {artist}")
            if num_paintings != 0:  assert False
            continue
        
        label_similarities = np.load(label_similarities)
        assert label_similarities.shape[1] == 2, f"Expected 15 labels, got {label_similarities.shape[1]}"
        paintings = np.sum(label_similarities[:, 0] > painting_embedding_threshold)

        ## now get the names of the images that are paintings. 
        artist_images = []
        for root, dirs, files in os.walk(os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images", artist)):
            for file in files:
                if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv"):
                    artist_images.append(os.path.join(root, file))
        
        number_of_total_downloaded_images = len(artist_images)
        assert len(artist_images) == label_similarities.shape[0], f"Expected {label_similarities.shape[0]} images, got {len(artist_images)} for {artist}"
        assert len(artist_images) <= artist_list_df[artist_list_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0], f"Expected {artist_list_df[artist_list_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]} paintings, got {len(artist_images)} for {artist}"
        
        new_num_paintings = (paintings * total_number_captions) // len(artist_images)     ## this is the estimated paintings of this person in the entire laion images based on the number of images I downloaded.
        # assert new_num_paintings >= num_paintings, f"Expected atleast {num_paintings} paintings, got {new_num_paintings} for {artist}"
        assert len(artist_images) >= paintings, f"Expected atleast {paintings} paintings, got {len(artist_images)} for {artist}"
        ## print the new number of paintings in the csv file. Keep all other columns the same.
        artist_list_df.loc[artist_list_df['artist_name'] == artist, 'count_paintings_in_laion_images'] = new_num_paintings
        artist_list_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv", index=False)
        
        print(f"Artist: {artist}, Paintings: {paintings}, Total images: {label_similarities.shape[0]}, Percentage: {paintings/label_similarities.shape[0]*100:.2f}")


def _get_embeddings_of_all_laion_paintings_(args, artist, artist_list_df):
    # painting_embedding_threshold = 0.1849365234375
    threshold_an_artwork_group1 = 0.1815185546875
    threshold_an_artwork_group2 = 0.1768798828125
    if args.art_group == "artist_group1":
        painting_embedding_threshold = threshold_an_artwork_group1
    elif args.art_group == "artist_group2":
        painting_embedding_threshold = threshold_an_artwork_group2
    else:
        raise NotImplementedError

    ## get the embeddings for all the laion images of the artist
    embsavepath = os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/laion_images_embeddings", f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
    num_total_laion_images = len(os.listdir(f"/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images/{artist}"))
    if num_total_laion_images > 1:
        assert os.path.isfile(os.path.join(embsavepath, 'embeddings_0.pkl')), f"Embeddings for {artist} do not exist at {embsavepath}"
    else:
        print(f"No LAION images downloaded found for {artist}")
        return None, None, 0, 0, 0

    ## load the embeddings and compute the similarity between the images of the same artist
    from search.embeddings import Embeddings
    emb = Embeddings(embsavepath, embsavepath, file_ext='.pkl', chunked=True, chunk_size=5000)
    this_artist_embeddings = emb.embeddings
    file_names_of_saved_embeddings = emb.filenames
    
    ## for the laion images, first get the images that are paintings
    total_number_captions = artist_list_df[artist_list_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]
    label_similarities = os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images", artist, f"text_probs_all_artists_images_{artist}.npy")
    # num_paintings = artist_list_df[artist_list_df['artist_name'] == artist]['count_paintings_in_laion_images'].values[0]
    num_artworks = artist_list_df[artist_list_df['artist_name'] == artist]['count_artworks_in_laion_images'].values[0]
    if not os.path.exists(label_similarities):
        print(f"Label similarities not found for {artist}")
        # if num_paintings != 0:
        if num_artworks != 0:
            assert False
        return None, None, 0, total_number_captions, 0
    
    label_similarities = np.load(label_similarities)
    labels_list = ["a painting", "a drawing", "an artwork", "an illustration", "a sketch", "a sculpture", "a mural", "a portrait", "a landscape", "a canvas art", "an abstract art", "a modern art", "a pop art", "a surreal art", "a textile artwork"]
    label_index = labels_list.index("an artwork")
    assert label_similarities.shape[0] == this_artist_embeddings.shape[0], f"Expected {label_similarities.shape[0]} images, got {this_artist_embeddings.shape[0]} for {artist}"
    # assert label_similarities.shape[1] == 2, f"Expected 2 labels, got {label_similarities.shape[1]}"
    assert label_similarities.shape[1] == len(labels_list), f"Expected {len(labels_list)} labels, got {label_similarities.shape[1]}"
    all_paintings = np.sum(label_similarities[:, label_index] > painting_embedding_threshold)

    # if total_number_captions > 100000:
    #     all_paintings_estimated = int(all_paintings * total_number_captions / 100000 + 0.5)
    # else:
    #     all_paintings_estimated = all_paintings
    all_paintings_estimated = all_paintings * total_number_captions // this_artist_embeddings.shape[0]      ## this is the estimated paintings of this person in the entire laion images based on the number of images I downloaded.
    
    # assert all_paintings_estimated == num_paintings, f"Expected {all_paintings_estimated} paintings, got {num_paintings} for {artist}"
    assert all_paintings_estimated == num_artworks, f"Expected {all_paintings_estimated} paintings, got {num_artworks} for {artist}"
    
    ## now get the embeddings of the paintings
    painting_embeddings = this_artist_embeddings[label_similarities[:, label_index] > painting_embedding_threshold]
    ## get the file_names that are paintings
    file_names_of_saved_embeddings = file_names_of_saved_embeddings[label_similarities[:, label_index] > painting_embedding_threshold]
    assert painting_embeddings.shape[0] == file_names_of_saved_embeddings.shape[0], f"Expected {file_names_of_saved_embeddings.shape[0]} paintings, got {painting_embeddings.shape[0]} for {artist}"
    assert painting_embeddings.shape[0] == all_paintings, f"Expected {all_paintings} paintings, got {painting_embeddings.shape[0]} for {artist}"
    assert painting_embeddings.shape[0] <= this_artist_embeddings.shape[0], f"Expected less than {this_artist_embeddings.shape[0]} paintings, got {painting_embeddings.shape[0]} for {artist}"
    return painting_embeddings, file_names_of_saved_embeddings, all_paintings, total_number_captions, all_paintings_estimated
        

def _get_embeddings_of_this_artists_paintings_(args, artist, artist_list_df, artist_painting_threshold, return_this_artist_painting_embeddings=True):
    from search.embeddings import Embeddings
    ## now get the embeddings of the wikiart paintings (reference images) for this artist
    wikiart_images_save_path = os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/wikiart_images_embeddings", f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
    assert os.path.isfile(os.path.join(wikiart_images_save_path, 'embeddings_0.pkl')), f"Embeddings for wikiart images of {artist} do not exist at {wikiart_images_save_path}"
    
    emb = Embeddings(wikiart_images_save_path, wikiart_images_save_path, file_ext='.pkl', chunked=True, chunk_size=5000)
    wikiart_images_embeddings = emb.embeddings
    # import ipdb; ipdb.set_trace()
    painting_embeddings, file_names_of_saved_embeddings, all_paintings, total_number_captions, all_paintings_estimated = _get_embeddings_of_all_laion_paintings_(args, artist, artist_list_df)
    
    if painting_embeddings is None:
        print(f"No paintings found for {artist}")
        if return_this_artist_painting_embeddings:
            return 0, 0
        return 0
    
    assert painting_embeddings.shape[0] == all_paintings, f"Expected {all_paintings} paintings, got {painting_embeddings.shape[0]} for {artist}"
    ## now compute the similarity between the laion images and the wikiart images, after normalizing the embeddings
    painting_embeddings = torch.tensor(painting_embeddings).to(args.device)
    wikiart_images_embeddings = torch.tensor(wikiart_images_embeddings).to(args.device)

    # Normalize the embeddings
    painting_embeddings /= (torch.norm(painting_embeddings, p=2, dim=1, keepdim=True) + 1e-16)
    wikiart_images_embeddings /= (torch.norm(wikiart_images_embeddings, p=2, dim=1, keepdim=True) + 1e-16)

    # Compute cosine similarity
    cosine_similarity_between_laion_paintings_and_wikiart_images = torch.mm(painting_embeddings, wikiart_images_embeddings.t())

    # Assert that all values are between -1 and 1 (almost)
    assert torch.all(cosine_similarity_between_laion_paintings_and_wikiart_images >= -1 - 1e-8) and torch.all(cosine_similarity_between_laion_paintings_and_wikiart_images <= 1 + 1e-8), f"max {torch.max(cosine_similarity_between_laion_paintings_and_wikiart_images)} min {torch.min(cosine_similarity_between_laion_paintings_and_wikiart_images)} for {artist}"

    # Count all the paintings that belong to this artist
    max_similarity_to_any_wikiart_image = torch.max(cosine_similarity_between_laion_paintings_and_wikiart_images, dim=1).values
    assert max_similarity_to_any_wikiart_image.shape == (all_paintings,), f"Expected shape {(all_paintings,)}, got {max_similarity_to_any_wikiart_image.shape}"

    num_paintings_this_artist = torch.sum(max_similarity_to_any_wikiart_image > artist_painting_threshold)
    print(f"Artist: {artist}, Total LAION images: {total_number_captions}, all artworks: {all_paintings}, artworks this artist: {num_paintings_this_artist}, Percentage of all artworks: {num_paintings_this_artist/all_paintings*100:.2f}")
    
    if return_this_artist_painting_embeddings:
        ## get the embeddings of this artists paintings, not all the paintings
        this_artist_painting_embeddings = painting_embeddings[max_similarity_to_any_wikiart_image > artist_painting_threshold]
        if file_names_of_saved_embeddings.shape == (1,):        ## np.ndarry
            # [max_similarity_to_any_wikiart_image.cpu() > artist_painting_threshold] is creating an error: index 1 is out of bounds for axis 0 with size 1. Write code to select the file if its max_similarity_to_any_wikiart_image > artist_painting_threshold
            if max_similarity_to_any_wikiart_image.cpu() > artist_painting_threshold:
                this_artist_painting_file_names = file_names_of_saved_embeddings
            else:
                this_artist_painting_file_names = torch.tensor([])
        else:
            this_artist_painting_file_names = file_names_of_saved_embeddings[max_similarity_to_any_wikiart_image.cpu() > artist_painting_threshold]
        assert this_artist_painting_embeddings.shape[0] == this_artist_painting_file_names.shape[0], f"Expected {this_artist_painting_file_names.shape[0]} paintings, got {this_artist_painting_embeddings.shape[0]} for {artist}"
        assert this_artist_painting_embeddings.shape[0] == num_paintings_this_artist, f"Expected {num_paintings_this_artist} paintings, got {this_artist_painting_embeddings.shape[0]} for {artist}"
        assert this_artist_painting_embeddings.shape[0] <= painting_embeddings.shape[0], f"Expected less than {painting_embeddings.shape[0]} paintings, got {this_artist_painting_embeddings.shape[0]} for {artist}"
        return this_artist_painting_embeddings, this_artist_painting_file_names
    else:
        ## these are the paintings of this artist among all the laion images that were classified as paintings. But we need to scale to to the number of estimated paintings.
        if all_paintings != 0:
            return round((num_paintings_this_artist * all_paintings_estimated / all_paintings).cpu().item())
        else:
            return round(num_paintings_this_artist.cpu().item())


def count_laion_images_for_artists(args, artist_painting_threshold):
    ## here we will compare the similarity between the laion images of an artist and the reference images of that artist. If the similarity is greater than a threshold, then we will consider that image to be a painting.
    # artist_list_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv")
    if args.art_group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.art_group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError
    
    artist_actual_paintings = {}
    for artist in artist_df['artist_name'].tolist():
        paintings_this_artist = _get_embeddings_of_this_artists_paintings_(args, artist, artist_df, artist_painting_threshold, return_this_artist_painting_embeddings=False)
        artist_actual_paintings[artist] = paintings_this_artist
    
    ## save the actual number of paintings for each artist in a csv file
    assert len(artist_df) == len(artist_actual_paintings), f"Expected {len(artist_df)} artists, got {len(artist_actual_paintings)} artists"
    artist_df['count_this_artist_artworks_in_laion_images'] = artist_df['artist_name'].map(artist_actual_paintings)
    # artist_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze2.csv", index=False)
    if args.art_group == "artist_group1":
        artist_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv", index=False)
    elif args.art_group == "artist_group2":
        artist_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv", index=False)
    else:
        raise NotImplementedError


def compute_similarity_between_generated_and_training_images(args, artist_painting_threshold):
    ## here we will take the embeddings of the generated and training images, and compute the cosine similarity between them, and then take the top-10 closest training images to generated images on average. 
    output_filename = f"average_cosine_similarity_wikiart_images_and_generated_images_prompt_{args.image_generation_prompt}_top_top_10_similar_to_average_generated_images_using_style_clip_wikiart_{args.art_group}.csv"
    
    ## if this file does not exist or has no lines in it, then we will write the header to it.
    if not os.path.exists(output_filename) or os.stat(output_filename).st_size == 0:
        with open(output_filename, "w") as f:
            f.write("artist_name,total_matching_captions,total_laion_paintings_this_artist,average_cosine_similarity\n")
    
    # artist_list_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv")   ## aartist_name,counts_in_laion2b-en,count_paintings_in_laion_images,counts_in_wikiart_original_dataset,downloaded_images_from_wikiart_website
    if args.art_group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.art_group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError
    
    if args.stable_diffusion_version == "2.1":
        df_laion5b = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/wikiart_artists_sorted_laion5b.csv", header=0)        ## artist_name,counts_in_laion5b
    
    from search.embeddings import Embeddings
    save_closest_images = False
    for artist in artist_df['artist_name'].tolist():
        print("processing", artist)
        ## GENERATED IMAGES EMBEDDINGS
        generated_images_save_path = os.path.join('generated_images_embeddings_stable_diffusion_version_' + str(args.stable_diffusion_version), f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}', args.image_generation_prompt)
        assert os.path.isfile(os.path.join(generated_images_save_path, 'embeddings_0.pkl')), f"Embeddings for generated images of {artist} do not exist at {generated_images_save_path}"
        emb = Embeddings(generated_images_save_path, generated_images_save_path, file_ext='.pkl', chunked=True, chunk_size=5000)
        generated_images_embeddings = emb.embeddings
        generated_images_embeddings = generated_images_embeddings / (np.linalg.norm(generated_images_embeddings, axis=1, keepdims=True, ord=2) + 1e-16)
        
        ## WIKIART IMAGES EMBEDDINGS
        wikiart_images_save_path = os.path.join("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/wikiart_images_embeddings", f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}', f'{artist}')
        assert os.path.isfile(os.path.join(wikiart_images_save_path, 'embeddings_0.pkl')), f"Embeddings for wikiart images of {artist} do not exist at {wikiart_images_save_path}"
        emb = Embeddings(wikiart_images_save_path, wikiart_images_save_path, file_ext='.pkl', chunked=True, chunk_size=5000)
        wikiart_images_embeddings_ = emb.embeddings
        wikiart_images_embeddings_ = wikiart_images_embeddings_ / (np.linalg.norm(wikiart_images_embeddings_, axis=1, keepdims=True, ord=2) + 1e-16)
        
        ## TRAINING IMAGES EMBEDDINGS: here we want to get the embeddings of the training images that are considered as art of this artist.
        this_artist_painting_embeddings, this_artist_painting_file_names = _get_embeddings_of_this_artists_paintings_(args, artist, artist_df, artist_painting_threshold, return_this_artist_painting_embeddings=True)
        using_laion_images = False
        if isinstance(this_artist_painting_embeddings, int) and this_artist_painting_embeddings == 0:
            print(f"No laion paintings found for {artist}")
            training_images_embedddings = wikiart_images_embeddings_
        elif isinstance(this_artist_painting_embeddings, torch.Tensor):
            if this_artist_painting_embeddings.shape[0] <= 10:
                training_images_embedddings = wikiart_images_embeddings_
            else:
                using_laion_images = True
                training_images_embedddings = this_artist_painting_embeddings
        else:
            raise ValueError(f"Expected 0 or torch.Tensor, got {this_artist_painting_embeddings}")
        
        print(f"Comparing generated images of {artist} (size: {generated_images_embeddings.shape[0]}) with their training images of size {training_images_embedddings.shape[0]}")
        
        ## load the generated_image embedding and the training_images_embedddings, compute the cosine similarity between them (after normalizing the embeddings)
        generated_images_embeddings = torch.tensor(generated_images_embeddings).to(args.device)
        training_images_embeddings = torch.tensor(training_images_embedddings).to(args.device)

        # Compute cosine similarity
        cosine_similarity_between_generated_and_training_images = torch.mm(generated_images_embeddings, training_images_embeddings.t())

        # Assert that all values are between -1 and 1 (almost)
        assert torch.all(cosine_similarity_between_generated_and_training_images >= -1 - 1e-8) and torch.all(cosine_similarity_between_generated_and_training_images <= 1 + 1e-8), f"max {torch.max(cosine_similarity_between_generated_and_training_images)} min {torch.min(cosine_similarity_between_generated_and_training_images)} for {artist}"

        # Find the 10 images that are closest to the generated images on average
        top_10_reference_images_closest_to_the_generated_images_on_average = torch.mean(cosine_similarity_between_generated_and_training_images, dim=0)
        assert top_10_reference_images_closest_to_the_generated_images_on_average.shape == (training_images_embeddings.shape[0],), f"Expected shape {(training_images_embeddings.shape[0],)}, got {top_10_reference_images_closest_to_the_generated_images_on_average.shape}"
        top_10_closest_images = torch.argsort(top_10_reference_images_closest_to_the_generated_images_on_average, descending=True)[:10]
        
        if using_laion_images and save_closest_images:
            ## we want to store the top-5 closest images to the generated images on average.
            os.makedirs("human_eval/top_5_closest_artist_images", exist_ok=True)
            os.makedirs(os.path.join("human_eval/top_5_closest_artist_images", artist), exist_ok=True)
            top_5_closest_images = torch.argsort(top_10_reference_images_closest_to_the_generated_images_on_average, descending=True)[:5]
            ## now get the names of the files of the top-5 closest images
            top_5_closest_images_file_names = this_artist_painting_file_names[top_5_closest_images.cpu()]
            from PIL import Image
            base_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/all_artists_images/{artist}"
            ## we only get the file names without the extension, we we need to add the extension to the file names which can be in various formats. So first look at the extension of the first file name and then add the extension to all the file names.
            for i, image_index in enumerate(top_5_closest_images):
                # image_file_name_without_extension = f"{base_directory}/{top_5_closest_images_file_names[i]}"
                ## read the files in the home directory of the images and take the full name will be the one whose first name matches the image_file_name_without_extension
                for root, dirs, files in os.walk(base_directory):
                    for file in files:
                        if file.startswith(top_5_closest_images_file_names[i]):
                            this_file = file
                            break
                image = Image.open(f"{base_directory}/{this_file}")
                image.save(f"human_eval/top_5_closest_artist_images/{artist}/{this_file}")
        
        cosine_similarity_between_generated_and_training_images = cosine_similarity_between_generated_and_training_images[:, top_10_closest_images]
        assert cosine_similarity_between_generated_and_training_images.shape == (generated_images_embeddings.shape[0], min(10, training_images_embedddings.shape[0])), f"Expected shape {(generated_images_embeddings.shape[0], min(10, training_images_embedddings.shape[0]))}, got {cosine_similarity_between_generated_and_training_images.shape}"
    
        average_similarity = torch.mean(cosine_similarity_between_generated_and_training_images)
        
        print(f"Average cosine similarity between the generated images of {artist} and the top-10 closest images of the wikiart images is {average_similarity}")
        
        if args.stable_diffusion_version in ["1", "5"]:
            # celeb|total_matching_captions|total_laion_paintings|average_cosine_similarity
            this_artist_total_matching_captions = artist_df[artist_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]
            # this_artist_laion_images_all_paintings = artist_df[artist_df['artist_name'] == artist]['count_paintings_in_laion_images'].values[0]
            this_artist_laion_images_all_artworks = artist_df[artist_df['artist_name'] == artist]['count_artworks_in_laion_images'].values[0]
            # this_artist_total_laion_their_paintings = artist_df[artist_df['artist_name'] == artist]['count_this_artist_paintings_in_laion_images'].values[0]
            this_artist_total_laion_their_artworks = artist_df[artist_df['artist_name'] == artist]['count_this_artist_artworks_in_laion_images'].values[0]
            # assert this_artist_total_laion_their_paintings <= this_artist_laion_images_all_paintings <= this_artist_total_matching_captions
            assert this_artist_total_laion_their_artworks <= this_artist_laion_images_all_artworks <= this_artist_total_matching_captions
        
        elif args.stable_diffusion_version == "2.1":
            this_artist_total_matching_captions_2b = artist_df[artist_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]
            this_artist_total_matching_captions_5b = df_laion5b[df_laion5b['artist_name'] == artist]['counts_in_laion5b'].values[0]
            
            this_artist_laion_images_all_paintings = artist_df[artist_df['artist_name'] == artist]['count_paintings_in_laion_images'].values[0]       ## this figure needs to be scaled according to this ratio of the captions in laion5b and laion2b-en
            this_artist_total_laion_their_artworks = artist_df[artist_df['artist_name'] == artist]['count_this_artist_artworks_in_laion_images'].values[0]      ## this number needs to be scaled according to the ratio of the number of new estimated paintings. 
            if this_artist_total_matching_captions_2b != 0:
                this_artist_laion_images_all_paintings = this_artist_laion_images_all_paintings * this_artist_total_matching_captions_5b / this_artist_total_matching_captions_2b
                this_artist_total_laion_their_artworks = this_artist_total_laion_their_artworks * this_artist_total_matching_captions_5b / this_artist_total_matching_captions_2b
            
            assert this_artist_total_laion_their_artworks <= this_artist_laion_images_all_paintings <= this_artist_total_matching_captions_5b, f"Expected {this_artist_total_laion_their_artworks} <= {this_artist_laion_images_all_paintings} <= {this_artist_total_matching_captions_5b} for {artist}"
            this_artist_total_matching_captions = this_artist_total_matching_captions_5b
        
        with open(output_filename, "a") as f:    
            f.write(f"{artist},{this_artist_total_matching_captions},{this_artist_total_laion_their_artworks},{average_similarity}\n")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('dynamicDistances-Embedding Generation Module')
    parser.add_argument('--compute_image_style_embeddings', action='store_true', help='Compute image style embeddings')
    parser.add_argument('--compute_similarity_between_art_images_of_same_artist', action='store_true', help='Compute similarity between art images of same artist')
    parser.add_argument('--compute_similarity_between_art_images_of_different_artists', action='store_true', help='Compute similarity between art images of different artists')
    parser.add_argument('--investigate_cosine_similarity', action='store_true', help='Investigate cosine similarity between images of same artist')
    parser.add_argument('--count_painting_images_for_artists', action='store_true', help='Count the number of paintings in the laion images for each artist using the CLIP classifier of similarity to "a painting" prompt.')
    parser.add_argument('--count_laion_images_for_artists', action='store_true', help='Count the number of laion images for each artist using the similarity to the reference images of that artist. We only do this for the images that the CLIP says are paintings.')
    parser.add_argument('--compute_similarity_between_generated_and_training_images', action='store_true', help='Compute similarity between generated and training images')
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset", choices=['wikiart'])
    parser.add_argument('--stable_diffusion_version', choices=["1", "5", "2.1"], default=None)
    parser.add_argument("--art_group", choices=["artist_group1", "artist_group2"], help="Group of artists to download images for which group", default=None, required=True)

    parser.add_argument('--qsplit', default='query', choices=['query', 'database'], type=str, help="The inferences")
    parser.add_argument('--data-dir', type=str, default=None, help='The directory of concerned dataset')
    parser.add_argument('--pt_style', default='csd', type=str)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128), this is the total '
                            'batch size of all GPUs on all nodes when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)

    # additional configs:
    parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')
    parser.add_argument('--num_loss_chunks', default=1, type=int)
    parser.add_argument('--isvit', action='store_true')
    parser.add_argument('--layer', default=1, type=int, help="layer from end to create descriptors from.")
    parser.add_argument('--feattype', default='normal', type=str, choices=['otprojected', 'weighted', 'concated', 'gram', 'normal'])
    parser.add_argument('--projdim', default=256, type=int)

    parser.add_argument('-mp', '--model_path', type=str, default=None)
    parser.add_argument('--gram_dims', default=1024, type=int)
    parser.add_argument('--query_count', default=-1, type=int, help='Number of queries to consider for final evaluation. Works only for domainnet')

    ## Additional config for CSD
    parser.add_argument('--eval_embed', default='head', choices=['head', 'backbone'], help="Which embed to use for eval")
    parser.add_argument('--skip_val', action='store_true')
    
    parser.add_argument('--which_images', default=None, choices=['wikiart_images', 'laion_images', 'generated_images'])
    parser.add_argument('--image_generation_prompt_id', type=int, default=0, help="Prompt id for image generation", choices=[0, 1, 2, 3, 4])
    
    args = parser.parse_args() 
    
    assert sum([args.compute_image_style_embeddings, args.compute_similarity_between_art_images_of_same_artist, args.compute_similarity_between_art_images_of_different_artists, 
                args.investigate_cosine_similarity, args.count_painting_images_for_artists, args.compute_similarity_between_generated_and_training_images, args.count_laion_images_for_artists]) == 1, "Exactly one of compute_image_style_embeddings, compute_similarity_between_art_images_of_same_artist, investigate_cosine_similarity, count_painting_images_for_artists must be True"   
    
    if args.model_path == "style_checkpoint.pth":
        args.model_path = "/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/" + args.model_path
    
    ## as result of running --investigate_cosine_similarity
    threshold_between_artwork_same_artist_and_different_artist_group1 = 0.2783
    threshold_between_artwork_same_artist_and_different_artist_group2 = 0.2883
        
    if args.which_images == 'wikiart_images':
        args.embed_dir = 'wikiart_images_embeddings'
    elif args.which_images == 'laion_images':
        args.embed_dir = 'laion_images_embeddings'
    elif args.which_images == 'generated_images':
        args.stable_diffusion_version is not None, "stable_diffusion_version must be provided"
        args.embed_dir = 'generated_images_embeddings_stable_diffusion_version_' + str(args.stable_diffusion_version)
        args.image_generation_prompt = prompts[args.image_generation_prompt_id][0]
    else:
        args.embed_dir = None
        print("You have not provided any specific images to compute embeddings for")
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    if args.compute_image_style_embeddings:
        assert args.which_images is not None, "which_images must be provided"
        compute_image_style_embeddings(args)
    
    elif args.compute_similarity_between_art_images_of_same_artist:
        compute_similarity_between_art_images_of_same_artist(args)
    
    elif args.compute_similarity_between_art_images_of_different_artists:
        compute_similarity_between_art_images_of_different_artists(args)
    
    elif args.investigate_cosine_similarity:
        investigate_cosine_similarity_and_find_threshold(args)
    
    elif args.count_painting_images_for_artists:
        count_painting_images_for_artists(args)
        
    elif args.count_laion_images_for_artists:
        if args.art_group == "artist_group1":
            count_laion_images_for_artists(args, artist_painting_threshold=threshold_between_artwork_same_artist_and_different_artist_group1)
        elif args.art_group == "artist_group2":
            count_laion_images_for_artists(args, artist_painting_threshold=threshold_between_artwork_same_artist_and_different_artist_group2)
        else:
            raise NotImplementedError
        # count_laion_images_for_artists(args, artist_painting_threshold=0.2698)
    
    elif args.compute_similarity_between_generated_and_training_images:
        assert args.which_images is None, "which_images must be None"
        args.stable_diffusion_version is not None, "stable_diffusion_version must be provided"
        args.image_generation_prompt = prompts[args.image_generation_prompt_id][0]
        if args.art_group == "artist_group1":
            compute_similarity_between_generated_and_training_images(args, artist_painting_threshold=threshold_between_artwork_same_artist_and_different_artist_group1)
        elif args.art_group == "artist_group2":
            compute_similarity_between_generated_and_training_images(args, artist_painting_threshold=threshold_between_artwork_same_artist_and_different_artist_group2)
        else:
            raise NotImplementedError
        # compute_similarity_between_generated_and_training_images(args, artist_painting_threshold=0.2698)
        
    else:
        raise NotImplementedError
