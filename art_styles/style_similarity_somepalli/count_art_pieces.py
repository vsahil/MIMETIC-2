## The goal of this file is to create a script that will take in a list of images and find the art pieces among them. We will use CLIP model to do this.
## For the CLIP model to select the art pieces, we will use the following prompts: 
import torch, os
import open_clip
import torch
import warnings
from PIL import Image
warnings.simplefilter("ignore")
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
random.seed(1)

cache_dir = "/gscratch/h2lab/vsahil/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ["DEEPFACE_HOME"] = cache_dir


def generate_preprocess(args, save_file, all_images, preprocess):
    if os.path.exists(save_file):
        return None
        preprocessed_images = torch.load(save_file)
    elif os.path.exists(save_file.replace(".pt", "_chunk_0.pt")):
        return None
        preprocessed_images = []
        for i in range(100):
            save_file_chunk = save_file.replace(".pt", f"_chunk_{i}.pt")
            if not os.path.exists(save_file_chunk):
                break
            preprocessed_images.append(torch.load(save_file_chunk))
        preprocessed_images = torch.cat(preprocessed_images, dim=0)
        print(f"Loaded the preprocessed images from {save_file}")
    else:        
        preprocessed_images = []
        for image_path in tqdm(all_images):
            img = Image.open(image_path)
            image = preprocess(img).unsqueeze(0)
            preprocessed_images.append(image)
        if len(preprocessed_images) == 0:
            ## there were no images for this artist
            return None
        
        ## break all_images into chunks of 4000 images and store the chunks as .pt files
        preprocessed_images = [preprocessed_images[i:i+4000] for i in range(0, len(preprocessed_images), 4000)]
        for i, preprocessed_images_chunk in enumerate(preprocessed_images):
            preprocessed_images_chunk = torch.cat(preprocessed_images_chunk, dim=0)
            save_file_chunk = save_file.replace(".pt", f"_chunk_{i}.pt")
            torch.save(preprocessed_images_chunk, save_file_chunk)
            print(f"Saved the preprocessed images to {save_file_chunk}")
        
    return preprocessed_images


def compute_label_similarties(args, text_features, model):
    if args.image_root == "wikiart_images":
        root_directory = args.image_root
        preprocessed_images_start = f"preprocessed_images_{args.image_root}"
    elif args.image_root == "all_artists_images":
        root_directory = os.path.join(args.image_root, args.this_artist)
        preprocessed_images_start = f"preprocessed_images_{args.image_root}_{args.this_artist}"
        
    ## in the root directory get all the .pt files that start with preprocessed_images_start
    preprocessed_images_file = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if not file.startswith(preprocessed_images_start):
                continue
            assert file.endswith(".pt"), f"Expected a .pt file, got {file}"
            preprocessed_images_file.append(os.path.join(root, file))
    
    if args.image_root == "wikiart_images":
        print(f"Found {len(preprocessed_images_file)} set of preprocessed images for {args.image_root}")
        save_file = f"wikiart_images/text_probs_{args.image_root}.npy"
    elif args.image_root == "all_artists_images":
        print(f"Found {len(preprocessed_images_file)} set of preprocessed images for {args.this_artist}")
        save_file = f"all_artists_images/{args.this_artist}/text_probs_{args.image_root}_{args.this_artist}.npy"
    
    if os.path.exists(save_file):
        print(f"Text_probs already exists for {args.image_root} or {args.this_artist}")
        return
    
    if len(preprocessed_images_file) == 0:
        if args.image_root == "wikiart_images":
            print(f"No preprocessed images found for {args.image_root}")
        elif args.image_root == "all_artists_images":
            print(f"No preprocessed images found for {args.this_artist}")
        return
    
    image_predictions = []
    for file in preprocessed_images_file:
        preprocessed_images = torch.load(file)
        preprocessed_images = preprocessed_images.to(args.device)
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            ## batch size is 1024 if the GPU is A100 and 512 if the GPU is A40
            batch_size = 1024 if "A100" in torch.cuda.get_device_name(0) else 512 if "A40" in torch.cuda.get_device_name(0) else 256
            print("Batch size:", batch_size)
            all_image_features = []
            for i in range(0, preprocessed_images.size(0), batch_size):
                all_image_features.append(model.encode_image(preprocessed_images[i:i+batch_size]))
            all_image_features = torch.cat(all_image_features, dim=0)
            all_image_features /= all_image_features.norm(dim=-1, keepdim=True)
            text_probs = (all_image_features @ text_features.T)
    
        print("Label similarities:", text_probs.cpu().numpy().shape, "for", file)
        image_predictions.append(text_probs.cpu().numpy())

    if args.save_pred:
        image_predictions = np.concatenate(image_predictions, axis=0)
        np.save(save_file, image_predictions)
        print(f"Saved the text_probs to {save_file}")


def compute_label_similarities_without_preprocessed_images(args, text_features, model, all_images, preprocess):
    if args.image_root == "wikiart_images":
        save_file = f"wikiart_images/text_probs_{args.image_root}.npy"
    elif args.image_root == "all_artists_images":
        save_file = f"all_artists_images/{args.this_artist}/text_probs_{args.image_root}_{args.this_artist}.npy"
    elif args.image_root == "mscoco_dataset":
        save_file = f"mscoco_dataset/text_probs_{args.image_root}.npy"
    elif args.image_root == "wikiart_website_images":
        save_file = f"wikiart_images/text_probs_{args.image_root}.npy"
    else:
        raise NotImplementedError
    
    if os.path.exists(save_file):
        print(f"Text_probs already exists for {args.image_root} or {args.this_artist}")
        return
    
    if len(all_images) == 0:
        if args.image_root == "wikiart_images":
            print(f"No images found for {args.image_root}")
        elif args.image_root == "all_artists_images":
            print(f"No images found for {args.this_artist}")
        return
    
    batch_size = 1024 if "A100" in torch.cuda.get_device_name(0) else 512 if "A40" in torch.cuda.get_device_name(0) else 256
    image_predictions = []
    for batch in range(0, len(all_images), batch_size):
        images = []
        for image_path in all_images[batch:batch+batch_size]:
            img = Image.open(image_path)
            image = preprocess(img).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)
        images = images.to(args.device)
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            all_image_features = []
            for inner_batch in range(0, images.size(0), batch_size):
                all_image_features.append(model.encode_image(images[inner_batch:inner_batch+batch_size]))
            all_image_features = torch.cat(all_image_features, dim=0)
            all_image_features /= all_image_features.norm(dim=-1, keepdim=True)
            text_probs = (all_image_features @ text_features.T)
    
        print("Label similarities:", text_probs.cpu().numpy().shape, "for", batch)
        image_predictions.append(text_probs.cpu().numpy())
    
    image_predictions = np.concatenate(image_predictions, axis=0)
    np.save(save_file, image_predictions)
    print(f"Saved the text_probs to {save_file}")


def generate_similarity_between_images_and_labels(cache_dir, args, labels_list):
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', cache_dir=cache_dir, device='cuda')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    if args.generate_similarities_between_images_and_labels:
        labels_tokenized = tokenizer(labels_list).to(args.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(labels_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    if args.image_root == "wikiart_images":
        ## there are 27 subdirectories in the wikiart_images directory, take all the images in all the subdirectories. Create a list of all the images. and then run the CLIP model on all the images.
        all_images = []
        for root, dirs, files in os.walk(args.image_root):
            for file in files:
                if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv"):
                    all_images.append(os.path.join(root, file))
        assert len(all_images) == 81445, f"Expected 81445 images, got {len(all_images)} images"
        if args.store_preprocessed_images:
            save_file = f"wikiart_images/preprocessed_images_{args.image_root}.pt"
            generate_preprocess(args, save_file, all_images, preprocess)
            return
        elif args.generate_similarities_between_images_and_labels:
            compute_label_similarties(args, text_features, model) #, save_file=f"wikiart_images/text_probs_{args.image_root}.npy")
    
    elif args.image_root == "mscoco_dataset":
        ## there are about 82783 images inside this directory. Create a preprocessed image numpy file for all of them and store in the mscoco_dataset directory.
        all_images = []
        for root, dirs, files in os.walk(args.image_root):
            for file in files:
                if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png"):
                    all_images.append(os.path.join(root, file))
        
        assert len(all_images) == 82783, f"Expected 82783 images, got {len(all_images)} images"
        if args.store_preprocessed_images:
            save_file = f"mscoco_dataset/preprocessed_images_{args.image_root}.pt"
            generate_preprocess(args, save_file, all_images, preprocess)
            return
    
    elif args.image_root == "all_artists_images":
        ## there are 415 artists whose laion images are downloaded in all_artists_images directory. Create a preprocessed image numpy file for each of them and store in the respective artist directory.
        import pandas as pd
        artist_list = pd.read_csv("artists_to_analyze.csv")['artist_name'].tolist()[::-1]
        all_images = {}
        for artist in artist_list:
            print("Processing artist:", artist)
            artist_images = []
            for root, dirs, files in os.walk(os.path.join(args.image_root, artist)):
                for file in files:
                    if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png"):
                        continue
                    artist_images.append(os.path.join(root, file))
            all_images[artist] = artist_images
            
            if args.store_preprocessed_images:
                save_file = f"all_artists_images/{artist}/preprocessed_images_{args.image_root}_{artist}.pt"
                generate_preprocess(args, save_file, all_images[artist], preprocess)
            elif args.generate_similarities_between_images_and_labels:
                args.this_artist = artist
                compute_label_similarties(args, text_features, model) #, save_file=f"all_artists_images/{artist}/text_probs_{args.image_root}_{artist}.npy")

    else:
        raise NotImplementedError


def generate_histogram_of_similarities(args, labels_list):
    # art_images_text_probs = np.load("wikiart_images/text_probs_wikiart_images.npy")
    # assert art_images_text_probs.shape == (81445, 15), f"Expected (81445, 15) got {art_images_text_probs.shape}"
    artist_group = 1
    art_images_text_probs = np.load(f"wikiart_images/text_probs_wikiart_website_images{artist_group}.npy")
    if artist_group == 1:
        assert art_images_text_probs.shape == (15304, 15), f"Expected (15304, 15) got {art_images_text_probs.shape}"
    elif artist_group == 2:
        assert art_images_text_probs.shape == (13659, 15), f"Expected (15304, 15) got {art_images_text_probs.shape}"
        
    non_art_images_text_probs = np.load("mscoco_dataset/text_probs_mscoco_dataset.npy")
    assert non_art_images_text_probs.shape == (82783, 15), f"Expected (82783, 15) got {non_art_images_text_probs.shape}"
    
    # True labels: 1 for art images, 0 for non-art images
    true_labels = np.concatenate( ( np.zeros(non_art_images_text_probs.shape[0]), np.ones(art_images_text_probs.shape[0]) ) )
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # Plot histograms
    fig_hist, axs_hist = plt.subplots(3, 5, figsize=(30, 10), sharex=True, sharey=True)
    fig_roc, axs_roc = plt.subplots(3, 5, figsize=(30, 10), sharex=True, sharey=True)
    fig_hist.subplots_adjust(hspace=0.5)
    fig_roc.subplots_adjust(hspace=0.5)
    label_aucs = {}
    
    # for i in range(3):
    #     for j in range(5):
    #         label_index = i * 5 + j
            
    #         # Combined probabilities and bins for histograms
    #         combined_probs = np.concatenate((art_images_text_probs[:, label_index], non_art_images_text_probs[:, label_index]))
    #         bins = np.histogram_bin_edges(combined_probs, bins=100)
            
    #         # Calculate histograms for each label with the same bins
    #         art_hist, _ = np.histogram(art_images_text_probs[:, label_index], bins=bins)
    #         non_art_hist, _ = np.histogram(non_art_images_text_probs[:, label_index], bins=bins)
            
    #         # Calculate intersection area
    #         intersection_area = np.sum(np.minimum(art_hist, non_art_hist))
            
    #         # Plot histogram
    #         axs_hist[i, j].hist(art_images_text_probs[:, label_index], bins=bins, color='b', alpha=0.5, label='Art images')
    #         axs_hist[i, j].hist(non_art_images_text_probs[:, label_index], bins=bins, color='r', alpha=0.5, label='Non-art images')
    #         axs_hist[i, j].set_title(f"Histogram of similarities for {labels_list[label_index]}\nIntersection: {intersection_area:.2f}")
    #         axs_hist[i, j].legend()
            
    #         # Calculate and plot ROC curve
    #         scores = np.concatenate((non_art_images_text_probs[:, label_index], art_images_text_probs[:, label_index]))
    #         # print(scores)
    #         fpr, tpr, thresholds = roc_curve(true_labels, scores)

    #         ## find the threshold that maximizes the F1 score
    #         f1_score = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr))
    #         threshold_for_max_f1_score = thresholds[np.argmax(f1_score)]
    #         print(f"Threshold for max F1 score: {threshold_for_max_f1_score}")
    #         ## at this threshold, print the FPR and TPR
    #         threshold_index = np.where(thresholds == threshold_for_max_f1_score)[0][0]
    #         print(f"FPR at threshold for max F1 score: {fpr[threshold_index]} for {labels_list[label_index]}")
    #         print(f"TPR at threshold for max F1 score: {tpr[threshold_index]} for {labels_list[label_index]}")

    #         roc_auc = auc(fpr, tpr)
    #         axs_roc[i, j].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.7f})')
    #         axs_roc[i, j].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #         axs_roc[i, j].set_xlim([0.0, 1.0])
    #         axs_roc[i, j].set_ylim([0.0, 1.05])
    #         axs_roc[i, j].set_xlabel('False Positive Rate')
    #         axs_roc[i, j].set_ylabel('True Positive Rate')
    #         axs_roc[i, j].set_title(f'ROC for {labels_list[label_index]}')
    #         axs_roc[i, j].legend(loc="lower right")
    #         label_aucs[labels_list[label_index]] = roc_auc
    
    for ax in axs_hist.flat:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    for ax in axs_roc.flat:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    
    print("AUCs for each label:")
    for label, auc_score in label_aucs.items():
        print(f"{label}: {auc_score:.7f}")
    
    ## print the label with the highest AUC
    max_auc_label = max(label_aucs, key=label_aucs.get)
    print(f"Label with the highest AUC: {max_auc_label}")
    
    plt.tight_layout()
    fig_hist.savefig(f"wikiart_images/histogram_of_similarities_for_art_and_non_art_images{artist_group}.png")
    fig_roc.savefig(f"wikiart_images/roc_curve_for_art_and_non_art_images{artist_group}.png")
    
    ## BASED ON THESE PLOTS - "an artwork" is the best label to use for separating art and non-art images. The AUC for this label is the highest for artists in both groups. So now we will count the number of artworks for each artist. 
    
    threshold_an_artwork_group1 = 0.1815185546875
    threshold_an_artwork_group2 = 0.1768798828125
    
    ## create a separate histogram plot for the 0th label. 
    plt.figure(figsize=(10, 7))
    label_index = 2
    threshold = threshold_an_artwork_group1 if artist_group == 1 else threshold_an_artwork_group2
    combined_probs = np.concatenate((art_images_text_probs[:, label_index], non_art_images_text_probs[:, label_index]))
    bins = np.histogram_bin_edges(combined_probs, bins=100)
    art_hist, _ = np.histogram(art_images_text_probs[:, label_index], bins=bins)
    non_art_hist, _ = np.histogram(non_art_images_text_probs[:, label_index], bins=bins)
    # intersection_area = np.sum(np.minimum(art_hist, non_art_hist))
    ## increase the fontsize of labels in legend to 20
    art_hist_data = art_images_text_probs[:, label_index]
    non_art_hist_data = non_art_images_text_probs[:, label_index]
    ## repeat the data in the art_hist_data 10 times to make the histogram the same size as the non_art_hist
    if artist_group == 1:
        art_hist_data = np.repeat(art_hist_data, 4)
    elif artist_group == 2:
        art_hist_data = np.repeat(art_hist_data, 10)
    plt.hist(art_hist_data, bins=bins, color='b', alpha=0.8, label='Art images')
    plt.hist(non_art_hist_data, bins=bins, color='r', alpha=0.8, label='Non-art images')
    # plt.title(f"Histogram of similarity to '{labels_list[label_index]}'", fontsize=20)
    ## add a vertical black line at the threshold = 0.1849365234375. Also add the point on the x-axis where the threshold is.
    plt.axvline(x=threshold, color='k', linestyle='--', label='Threshold', ymax=0.85)
    ## label the point where the threshld line touches the x-axis -- this is the point where the threshold is.
    if artist_group == 1:
        plt.text(threshold_an_artwork_group1, 4300, f"Threshold: {round(threshold, 3)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
    elif artist_group == 2:
        plt.text(threshold_an_artwork_group2, 4500, f"Threshold: {round(threshold, 3)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
    else:
        plt.text(0.18, 4500, f"Threshold: {round(threshold, 3)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
    plt.legend(fontsize=16, loc='upper left')
    plt.xlabel(f"Cosine similarity to '{labels_list[label_index]}'", fontsize=20)
    plt.ylabel("Number of images", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"wikiart_images/histogram_of_similarities_for_art_and_non_art_images_label_{label_index}_artist_group_{artist_group}.pdf")


def get_wrongly_classified_images(args, artist_group, all_images, all_mscoco_images, threshold_an_artwork_group1, threshold_an_artwork_group2):
    label_index = 2

    art_images_text_probs = np.load(f"wikiart_images/text_probs_wikiart_website_images{artist_group}.npy")
    if artist_group == 1:
        assert art_images_text_probs.shape == (15304, 15), f"Expected (15304, 15) got {art_images_text_probs.shape}"
    elif artist_group == 2:
        assert art_images_text_probs.shape == (13659, 15), f"Expected (15304, 15) got {art_images_text_probs.shape}"
        
    non_art_images_text_probs = np.load("mscoco_dataset/text_probs_mscoco_dataset.npy")
    assert non_art_images_text_probs.shape == (82783, 15), f"Expected (82783, 15) got {non_art_images_text_probs.shape}"
    
    # print the art images that are below the threshold
    if artist_group == 1:
        selected_images_art_wrong = np.where(art_images_text_probs[:, label_index] < threshold_an_artwork_group1)[0]
        selected_images_nonart_wrong = np.where(non_art_images_text_probs[:, label_index] > threshold_an_artwork_group1)[0]
    elif artist_group == 2:
        selected_images_art_wrong = np.where(art_images_text_probs[:, label_index] < threshold_an_artwork_group2)[0]
        selected_images_nonart_wrong = np.where(non_art_images_text_probs[:, label_index] > threshold_an_artwork_group2)[0]
    print(f"Number of art images below the threshold: {len(selected_images_art_wrong)}")
    print(f"Number of non-art images above the threshold: {len(selected_images_nonart_wrong)}")

    os.makedirs("wrong_classified_images", exist_ok=True)
    os.makedirs("wrong_classified_images/art_images", exist_ok=True)
    os.makedirs("wrong_classified_images/non_art_images", exist_ok=True)
    
    for i, image_index in enumerate(selected_images_art_wrong):
        image_path = all_images[image_index]
        # print(image_path)
        ## store the image in the wrong_classified_images/art_images directory
        os.system(f"cp {image_path} wrong_classified_images/art_images/")
    print("Stored the wrongly classified art images in wrong_classified_images/art_images directory.")
    
    for i, image_index in enumerate(selected_images_nonart_wrong):
        image_path = all_mscoco_images[image_index]
        # print(image_path)
        ## store the image in the wrong_classified_images/non_art_images directory
        os.system(f"cp {image_path} wrong_classified_images/non_art_images/")
    print("Stored the wrongly classified non-art images in wrong_classified_images/non_art_images directory.")


def print_outliers(args, labels_list):
    selected_labels = ["a painting", "an artwork", "a canvas art"]
    if args.image_root == "wikiart_images":
        text_probs = np.load(f"wikiart_images/text_probs_{args.image_root}.npy")
        assert text_probs.shape == (81445, 15), f"Expected (81445, 15) got {text_probs.shape}"
        ## each subdirectory inside the root is the style, and all the images inside the subdirectories are of that style, get the images for all the 27 styles inside the wikiart_images directory
        all_images = []
        all_images_particular_styles = {}
        for root, dirs, files in os.walk(args.image_root):
            for style in dirs:
                all_images_particular_styles[style] = []
                for file in os.listdir(os.path.join(root, style)):
                    if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png"):
                        all_images_particular_styles[style].append(os.path.join(root, style, file))
                all_images.extend(all_images_particular_styles[style])
        assert len(all_images) == 81445, f"Expected 81445 images, got {len(all_images)} images"
        assert len(all_images_particular_styles) == 27, f"Expected 27 styles, got {len(all_images_particular_styles)} styles"
        
    elif args.image_root == "all_artists_images":
        raise NotImplementedError
    
    # import subprocess
    # ## now get the images that have the 10 lowest similarity for the selected_labels
    # for label in selected_labels:
    #     label_index = labels_list.index(label)
    #     sorted_indices = np.argsort(text_probs[:, label_index])
    #     print(f"Label: {label}")
    #     for i in range(10):
    #         image_path = all_images[sorted_indices[i]]
    #         print(f"Image path: {image_path}, Similarity: {text_probs[sorted_indices[i], label_index]}")
    #         ## display the image
    #         print(f"Open this image: {image_path}")
    #         subprocess.run(['code', '-r', image_path])
    #         score = None
    #         while score not in ["0", "1", "2", "3", "4", "5"]:
    #             score = input(f"Enter a number between 0 and 5 for giving a score of how close the image resembles a painting: ")
            
    #         print(f"scored {image_path} as {score}.\n")
    
    ## for each style of image inside wikiart, generate the histogram of similarities for the selected_labels. Therefore we will get a histogram of size 27 X 3 (27 styles and 3 labels)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(27, 3, figsize=(30, 100), sharex=True)
    for i, style in enumerate(all_images_particular_styles):
        for j, label in enumerate(selected_labels):
            label_index = labels_list.index(label)
            text_probs_particular_style = text_probs[[all_images.index(image_path) for image_path in all_images_particular_styles[style]], label_index]
            assert len(text_probs_particular_style) == len(all_images_particular_styles[style]), f"Expected {len(all_images_particular_styles[style])} got {len(text_probs_particular_style)}"
            axs[i, j].hist(text_probs_particular_style, bins=50)
            axs[i, j].set_title(f"Histogram of similarities for {label} in {style}")
            
    for ax in axs.flat:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    plt.tight_layout()
    plt.savefig(f"wikiart_images/histogram_of_similarities_{args.image_root}_selected_labels.png")


def count_paintings_in_laion_images(args, artist, total_number_captions, threshold, labels_list, matching_label):
    label_similarities = os.path.join("all_artists_images", artist, f"text_probs_all_artists_images_{artist}.npy")
    if not os.path.exists(label_similarities):
        print(f"Label similarities not found for {artist}")
        return 0
    label_similarities = np.load(label_similarities)
    
    assert label_similarities.shape[1] == len(labels_list), f"Expected {len(labels_list)} labels, got {label_similarities.shape[1]}"
    ## for each image, get the similarity to the label "a painting" and if the similarity is greater than the threshold, then count it as a painting.
    # paintings = 0
    # paintings = np.sum(label_similarities[:, 0] > threshold)
    ## find the index of matching_label in the labels_list
    matching_label_index = labels_list.index(matching_label)
    num_matching_labels = np.sum(label_similarities[:, matching_label_index] > threshold)
    ## I downloaded a max of 100k images per celeb, so we need to scale the number of paintings by the total number of images
    # if total_number_captions > 100000:
    #     # paintings = int(paintings * total_number_captions / 100000 + 0.5)
    #     num_matching_labels = int(num_matching_labels * total_number_captions / 100000 + 0.5)
    num_matching_labels = num_matching_labels * total_number_captions // label_similarities.shape[0]
    print(f"Artist: {artist}, index: {matching_label_index}, Artworks: {num_matching_labels}, Total images: {label_similarities.shape[0]}, Percentage: {num_matching_labels/label_similarities.shape[0]*100:.2f}")
    return num_matching_labels


def get_caption_of_file(images_list):
    caption_file = "./mscoco_captions/annotations_2014/instances_train2014.json"
    import json
    with open(caption_file, "r") as f:
        all_captions = json.load(f)

    for image in images_list:
        target_file_name = image.split("/")[-1]
        # Assuming all_captions is the loaded JSON file containing the dataset annotations
        images = all_captions['images']
        annotations = all_captions['annotations']
        categories = all_captions['categories']
        
        category_map = {cat['id']: cat['name'] for cat in categories}
        
        # Find the image ID corresponding to the target file name
        image_id = next((img['id'] for img in images if img['file_name'] == target_file_name), None)

        if image_id is not None:
            # Find all annotations for the image_id and get category IDs
            category_ids = {ann['category_id'] for ann in annotations if ann['image_id'] == image_id}

            # Retrieve the category names using the category IDs
            category_names = [category_map[cat_id] for cat_id in category_ids if cat_id in category_map]

            print(f"Categories for {target_file_name}:")
            for category in category_names:
                print(category)
        else:
            print("File name not found in the dataset.")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default=None, choices=["wikiart_images", "all_artists_images", "mscoco_dataset", "wikiart_website_images"])
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--store_preprocessed_images", action='store_true')
    parser.add_argument("--artists", type=str, default=None)
    parser.add_argument("--generate_similarities_between_images_and_labels", action='store_true')
    parser.add_argument("--generate_histogram_of_similarities", action='store_true')
    parser.add_argument("--print_outliers", action='store_true')
    parser.add_argument("--compute_label_similarities_without_preprocessed_images", action='store_true')
    parser.add_argument("--count_paintings", action='store_true')
    parser.add_argument("--get_wrongly_classified_images", action='store_true')
    args = parser.parse_args()
    print(args)

    assert args.store_preprocessed_images is False, "This option is no longer supported, preprocessed images take a long time to "
    labels_list = ["a painting", "a drawing", "an artwork", "an illustration", "a sketch", "a sculpture", "a mural", "a portrait", "a landscape", "a canvas art", "an abstract art", "a modern art", "a pop art", "a surreal art", "a textile artwork"]
    # labels_list = ["a painting", "a drawing"]
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    threshold_an_artwork_group1 = 0.1815185546875
    threshold_an_artwork_group2 = 0.1768798828125

    if args.image_root == "wikiart_images":
        assert args.artists is None, "Artists should be None for wikiart_images"
        
    if args.artists is not None:
        assert args.image_root != "wikiart_images", "artists laion images are separate from wikiart images"

    if args.compute_label_similarities_without_preprocessed_images or args.get_wrongly_classified_images:
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', cache_dir=cache_dir, device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        labels_tokenized = tokenizer(labels_list).to(args.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(labels_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        if args.image_root == "wikiart_images":
            get_these_art_images = [20233, 20442, 43726, 44273, 44321]
            ## there are 27 subdirectories in the wikiart_images directory, take all the images in all the subdirectories. Create a list of all the images. and then run the CLIP model on all the images.
            all_images = []
            for root, dirs, files in os.walk(args.image_root):
                for file in files:
                    if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png") and not file.endswith(".zip") and not file.endswith(".pdf"):
                        all_images.append(os.path.join(root, file))
            assert len(all_images) == 81445, f"Expected 81445 images, got {len(all_images)} images"
            ## get the names of the images in the get_these_art_images list
            get_these_art_images = [all_images[index] for index in get_these_art_images]
            print(get_these_art_images)
            ## open these images, "add "./" before the image path"
            import subprocess
            for image_path in get_these_art_images:
                print(f"Open this image: {image_path}")
                subprocess.run(['code', '-r', os.path.join("./", image_path)])
            # compute_label_similarities_without_preprocessed_images(args, text_features, model, all_images, preprocess)
    
        elif args.image_root == "mscoco_dataset":
            get_these_non_art_indices = [16289, 50068, 57872, 76033, 77593]
            ## there are about 82783 images inside this directory. Create a preprocessed image numpy file for all of them and store in the mscoco_dataset directory.
            all_images = []
            for root, dirs, files in os.walk(args.image_root):
                for file in files:
                    if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png"):
                        all_images.append(os.path.join(root, file))
            
            assert len(all_images) == 82783, f"Expected 82783 images, got {len(all_images)} images"
            ## get the names of the images in the get_these_non_art_indices list
            get_these_non_art_indices = [all_images[index] for index in get_these_non_art_indices]
            print(get_these_non_art_indices)
            get_caption_of_file(get_these_non_art_indices)
            # import subprocess
            ## open these images, "add "./" before the image path"
            # for image_path in get_these_non_art_indices:
            #     print(f"Open this image: {image_path}")
            #     subprocess.run(['code', '-r', os.path.join("./", image_path)])
            # compute_label_similarities_without_preprocessed_images(args, text_features, model, all_images, preprocess)
        
        elif args.image_root == "all_artists_images":
            ## there are 415 artists whose laion images are downloaded in all_artists_images directory. 
            artist_list = pd.read_csv("artists_to_analyze.csv")['artist_name'].tolist()
            artist_list1 = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")['artist_name'].tolist()
            artist_list2 = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")['artist_name'].tolist()
            artist_list = artist_list1 + artist_list2
            ## remove duplicates from the artist_list
            artist_list = list(set(artist_list))
            ## read artists already done from the completed_artists.txt file
            # with open("completed_artists.txt", "r") as f:
            #     completed_artists = f.readlines()
            # completed_artists = [artist.strip() for artist in completed_artists]
            # artist_list = [artist for artist in artist_list if artist not in completed_artists]
            print(len(artist_list), "artists to process")
            all_images = {}
            ## randomzie the artist_list
            # random.shuffle(artist_list)
            for artist in artist_list:
                print("Processing artist:", artist)
                artist_images = []
                for root, dirs, files in os.walk(os.path.join(args.image_root, artist)):
                    for file in files:
                        if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv"):
                            artist_images.append(os.path.join(root, file))
                all_images[artist] = artist_images
                args.this_artist = artist
                compute_label_similarities_without_preprocessed_images(args, text_features, model, all_images[artist], preprocess)
                # with open("completed_artists.txt", "a") as f:
                #     f.write(artist + "\n")
        
        elif args.image_root == "wikiart_website_images":
            ## take all the folders inside
            from unidecode import unidecode
            artist_group = 1
            artist_list = pd.read_csv(f"/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group{artist_group}.csv")['artist_name'].tolist()
            
            replacement_list = {'apollinary goravsky': 'apollinariy-goravskiy', 'petro kholodny': 'petro-kholodny-elder', 'alexei korzukhin': 'aleksey-ivanovich-korzukhin', 'jérôme-martin langlois': 'jerome-martin-langlois'}
            all_wikiart_internet_images = []
            
            print(f"start processing artists in group {artist_group}")
            for artist_name in artist_list:
                if artist_name in replacement_list:
                    artist_name_downloaded_folder = replacement_list[artist_name]
                else:
                    artist_name_downloaded_folder = unidecode(artist_name.strip().lower().replace("'", ' ').replace(".", ' ').replace('   ', '-').replace('   ', '-').replace('  ', '-').replace(' ', '-'))    
                downloaded_images_folder = f"wikiart_images_downloaded/{artist_name_downloaded_folder}"
                if os.path.exists(downloaded_images_folder):
                    all_images = []
                    for root, dirs, files in os.walk(downloaded_images_folder):
                        for file in files:
                            if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv"):
                                all_images.append(os.path.join(root, file))
                    all_wikiart_internet_images.extend(all_images)
            
            print(f"Total number of images: {len(all_wikiart_internet_images)}")
            if args.compute_label_similarities_without_preprocessed_images:
                ## find the text similarity of these images to the labels
                compute_label_similarities_without_preprocessed_images(args, text_features, model, all_wikiart_internet_images, preprocess)
            elif args.get_wrongly_classified_images:
                ## get the wrongly classified images
                ## also get the mscoco images 
                all_mscoco_images = []
                for root, dirs, files in os.walk("mscoco_dataset"):
                    for file in files:
                        if not file.endswith(".pt") and not file.endswith(".npy") and not file.endswith(".csv") and not file.endswith(".png"):
                            all_mscoco_images.append(os.path.join(root, file))
                
                assert len(all_mscoco_images) == 82783, f"Expected 82783 images, got {len(all_images)} images"
                get_wrongly_classified_images(args, artist_group, all_wikiart_internet_images, all_mscoco_images, threshold_an_artwork_group1, threshold_an_artwork_group2)

        else:
            raise NotImplementedError
        
    elif args.generate_similarities_between_images_and_labels or args.store_preprocessed_images:
        generate_similarity_between_images_and_labels(cache_dir, args, labels_list)
        
    elif args.generate_histogram_of_similarities:
        generate_histogram_of_similarities(args, labels_list)
        
    elif args.print_outliers:
        print_outliers(args, labels_list)
    
    elif args.count_paintings:
        # threshold_painting = 0.1849365234375
        ## compute the smilarities of the images of each artist to the label: "a painting", and if the threshold is greater than 0.1849365234375, then count it as a painting.
        import pandas as pd
        # artist_list_df = pd.read_csv("artists_to_analyze.csv")
        artist_list_df1 = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
        artist_list_df2 = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
        paintings = {}
        artist_group = 1
        
        threshold = threshold_an_artwork_group1 if artist_group == 1 else threshold_an_artwork_group2
        
        if artist_group == 1:
            artist_list_df = artist_list_df1
        elif artist_group == 2:
            artist_list_df = artist_list_df2
        for artist in artist_list_df['artist_name'].tolist():
            print("Processing artist:", artist)
            total_number_captions = artist_list_df[artist_list_df['artist_name'] == artist]['counts_in_laion2b-en'].values[0]
            this_artist_painting = count_paintings_in_laion_images(args, artist, total_number_captions, threshold=threshold, labels_list=labels_list, matching_label='an artwork')
            paintings[artist] = this_artist_painting
        
        ## now add this paintings as a new column to the artists_to_analyze.csv file
        assert len(artist_list_df) == len(paintings), f"Expected {len(artist_list_df)} artists, got {len(paintings)} artists"
        # artist_list_df['count_paintings_in_laion_images'] = artist_list_df['artist_name'].map(paintings)
        artist_list_df['count_artworks_in_laion_images'] = artist_list_df['artist_name'].map(paintings)
        artist_list_df.to_csv(f"final_artists_group{artist_group}.csv", index=False)
    
    else:
        raise NotImplementedError

