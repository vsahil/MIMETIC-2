import torch, os, copy

# cache_dir = "/home/nlp/royira/vlm-efficiency/"
cache_dir = "/gscratch/h2lab/vsahil/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from diffusers.utils import make_image_grid
from diffusers.utils.pil_utils import make_image_grid


def load_model(args):
    if args.model_id in ["1", "2", "3", "4"]:
        model = f"CompVis/stable-diffusion-v1-{args.model_id}"
    elif args.model_id == "5":
        model = "runwayml/stable-diffusion-v1-5"
    elif args.model_id == "v2":
        model = "stabilityai/stable-diffusion-2-1"
    device = f"cuda:{args.device}"
    if args.model_id != "v2":
        pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None)
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, safety_checker=None, cache_dir=cache_dir)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    return pipe


def generate_one_batch_images(args, model, this_batch_size, image_batch_seed):
    generator = [torch.Generator(f"cuda:{args.device}").manual_seed(i + image_batch_seed) for i in range(this_batch_size)]
    prompts = this_batch_size * [args.full_prompt]
    dict_ = {"prompt": prompts, "generator": generator}
    if args.guidance_scale is not None:
        dict_["guidance_scale"] = args.guidance_scale
    if args.num_inference_steps is not None:
        dict_["num_inference_steps"] = args.num_inference_steps
    with torch.no_grad():
        images = model(**dict_).images
    torch.cuda.empty_cache()
    return images


def generate_batched_images(args, model=None):
    ## max batch size that it can handle is 100, so if the args.batch_size is larger than 100, then we need to generate images in batches of 100.
    num_images = 0
    max_one_batch_size = copy.deepcopy(args.batch_size)     ## Depending on the GPU, this can be very high. 
    all_images = []

    if args.set_of_people in ["celebrity", "politicians", "animals"]:
        saving_directory = f"generated_images_{args.set_of_people}_{args.model_id}/{args.celebrity}/{args.prompt}"
        os.makedirs(f"generated_images_{args.set_of_people}_{args.model_id}", exist_ok=True)
        os.makedirs(f"generated_images_{args.set_of_people}_{args.model_id}/{args.celebrity}", exist_ok=True)
    elif args.set_of_people == "cub_birds":
        saving_directory = f"generated_images_cub_birds/{args.celebrity}/{args.prompt}"
        os.makedirs("generated_images_cub_birds", exist_ok=True)
        os.makedirs(f"generated_images_cub_birds/{args.celebrity}", exist_ok=True)
    elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
        saving_directory = f"generated_images_wikiart_artists/{args.celebrity}/{args.prompt}"
        os.makedirs("generated_images_wikiart_artists", exist_ok=True)
        os.makedirs(f"generated_images_wikiart_artists/{args.celebrity}", exist_ok=True)
    else:
        raise NotImplementedError

    if args.guidance_scale is not None:
        saving_directory += f"_guidance_scale_{args.guidance_scale}"
    if args.num_inference_steps is not None:
        saving_directory += f"_num_inference_steps_{args.num_inference_steps}"

    os.makedirs(saving_directory, exist_ok=True)
    
    ## check if there are already 201 files in the directory, if yes, then skip this celebrity.
    if len(os.listdir(saving_directory)) > args.total_images:
        print(f"Skipping {args.prompt} for {args.celebrity} as it already has more than {args.total_images} images.")
        return

    image_batch_seed = 0
    while num_images < args.total_images:
        images_this_batch = min(args.total_images - num_images, max_one_batch_size)
        images = generate_one_batch_images(args, model, images_this_batch, image_batch_seed)
        all_images.extend(images)
        num_images += images_this_batch
        image_batch_seed += images_this_batch + 1
        ## do the saving here itself, instead of saving at the end. Make sure we are not overwriting the images.
        for i, image in enumerate(images):
            this_image_seq = num_images - images_this_batch + i
            image.save(f"{saving_directory}/image_seq_{this_image_seq}.png")
    assert num_images == args.total_images

    if args.total_images == 20:
        rows, cols = 4, 5
    elif args.total_images == 40:
        rows, cols = 8, 5
    elif args.total_images == 4:
        rows, cols = 2, 2
    elif args.total_images == 200:
        rows, cols = 10, 20
    elif args.total_images == 100:
        rows, cols = 10, 10
    else:
        print("Not saving the grid images")
        return model
    
    assert rows * cols == args.total_images
    grid = make_image_grid(all_images, rows, cols)
    grid.save(f"{saving_directory}/grid_images.png")

    return model
    ## we also need to detect the black images and remove them, so that they do interfere with our evaluations. --- problem handled by skipping the safe generation of images.


def generate_one_image(args):
    model = load_model(args)
    with autocast("cuda"):
        image = model(args.full_prompt)
    image.images[0].save("astronaut_rides_horse.png")


# from run_gen_images import deviceid2model
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--celebrity", type=str, default=None, help="Comma-separated list of celebrities")
    parser.add_argument("--set_of_people", type=str, default=None, choices=["celebrity", "politicians", "cub_birds", "animals", "wikiart_artists", "artists_group1", "artists_group2"])
    parser.add_argument("--prompt", type=str, default=None) #default="a photorealistic close-up photograph of ")
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--model_id", type=str, default="1", choices=["1", "2", "3", "4", "5", "v2"])
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--total_images", type=int, default=200)
    args = parser.parse_args()

    import pandas as pd 
    # complete_celebrity_list = pd.read_csv("celebrity_data/sampled_politicians.csv")['Name'].tolist()

    # Filter the list based on the input argument
    if args.celebrity is not None:
        args.celebrity = args.celebrity.split(",")
        celebrity_list = copy.deepcopy(args.celebrity)
        # if args.set_of_people == "politicians":
        #     celebrity_list_final = [celeb for celeb in complete_celebrity_list if celeb in celebrity_list]
        #     celebrity_list = celebrity_list_final
    else:
        celebrity_list = complete_celebrity_list 
    
    print(f"Generating images for all celebrities in the list: {celebrity_list} and its length is {len(celebrity_list)}")
    
    if args.set_of_people in ["celebrity", "politicians"]:
        prompts = [("a photorealistic close-up photograph of ", 1), ("High-resolution close-up image of ", 1), ("'s facial close-up", 2), ("Close-up headshot of ", 1), ("'s face portrait", 2)]
    elif args.set_of_people == "cub_birds":
        prompts = [("a photorealistic photograph of ", 1)]
    elif args.set_of_people == "animals":
        prompts = [("a photorealistic photograph of a ", 1), ("High-resolution image of a ", 1), ("An image of a ", 2), ("A detailed image of a ", 1), ("A portrait of a ", 2)]
    elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
        prompts = [("a painting in the style of ", 1), ("an artwork in the style of ", 1), ("a sketch in the style of ", 1), ("a fine art piece in the style of ", 1),  ("an illustration in the style of ", 1)]
    else:
        raise NotImplementedError
    
    args.prompt, args.prompt_type = prompts[args.prompt_id]
    

    model = None
    if model is None:
        model = load_model(args)

    for idx, celeb_here in enumerate(celebrity_list):
        args.celebrity = celeb_here
        if args.prompt_type == 1:
            args.full_prompt = args.prompt + args.celebrity
        elif args.prompt_type == 2:
            args.full_prompt = args.prompt + args.celebrity
        else: 
            raise NotImplementedError
        print(f"{args.device}, {idx}, {celeb_here} <> {args.full_prompt}")
        if args.total_images > 1:
            _ = generate_batched_images(args, model)
        else:
            image = generate_one_image(args)

'''
Todos:

generate prompt for each of them -- save the name of the generated image as the celebrity name + the prompt - done.
Generate the image using batch processing - done
determine the largest batch size possible - 20 yes | 32 yes | 40 yes | 50 no |  64 no | -- so 40 is the largest batch size possible with GPU rtx6k series.
write a script to generate the images for each of the celebrities, use different GPUs. - done
Generate 200 images per prompt per celebrity - done
save the images  - done
save the grid of images - done
'''

