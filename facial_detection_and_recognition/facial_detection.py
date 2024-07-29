# Based on advice from https://learnopencv.com/what-is-face-detection-the-ultimate-guide/, we are going to use RetinaFace for facial detection
import os, json, itertools, time, copy, random, unicodedata
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager, Process


cache_dir = "/gscratch/h2lab/vsahil/"
# cache_dir = "/home/nlp/royira/vlm-efficiency"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ["DEEPFACE_HOME"] = cache_dir
os.environ["INSIGHTFACE_HOME"] = cache_dir


celebrity_list = ['Aditya Kusupati', 'Bridgette Doremus', 'Joey Klaasen', 'Hannah Margaret Selleck', 'Mary Lee Pfeiffer', 'Miguel Bezos', 'Ben Mallah', 'Tara Lynn Wilson', 'Eric Porterfield', 'Quinton Reynolds', 'Avani Gregg', 'David Portnoy', 'Kyle Cooke', 'Sue Aikens', 
                  'Danielle Cohn', 'Marcus Freeman', 'Ryan Upchurch', 'Greg Fishel', 'Wayne Carini', 'Cole LaBrant', 'Ella Emhoff', 'Manny Khoshbin', 'Lexi Underwood', 'Ciara Wilson', 'Scott Yancey', 'Alex Choi', 'Shenae Grimes-Beech', 'Nicole Fosse', 'Isaak Presley', 
                  'Yamiche Alcindor', 'Matthew Garber', 'Rege-Jean Page', 'Nick Bolton', 'Will Reeve', 'Madison Bailey', 'Maitreyi Ramakrishnan', 'Zaya Wade', 'Sessilee Lopez', 'Albert Lin', 'Frank Sheeran', 'Fred Stoller', 'Trevor Wallace', 'Madelyn Cline', 
                  'Angie Varona', 'Yael Cohen', 'Bailey Sarian', 'Zack Bia', 'M.C. Hammer', 'Irwin Winkler', 'Danny Koker', 'Tony Lopez', 'Sasha Calle', 'Maggie Rizer', 'Jillie Mack', "Olivia O'Brien", 'Joanna Hillman', 'Belle Delphine', 'Chase Stokes', 'Kyla Weber', 
                  'Alexandra Cooper', 'Jordan Chiles', 'Bob Keeshan', "Dixie D'Amelio", 'Daisy Edgar-Jones', 'Yung Gravy', 'Jamie Bochert', 'Forrest Fenn', 'Barbie Ferreira', 'DJ Kool Herc', 'Bregje Heinen', 
                  'Dan Peña', 'Thandiwe Newton', 'Alexa Demie', 'Paul Mescal', 'Jeremy Bamber', 'Malgosia Bela', 'Lacey Evans', 'Tao Okamoto', 'Ashleigh Murray', 'Nico Tortorella', 'Annie Murphy', 'Jimmy Buffet', 'Marsai Martin', 'Sofia Hellqvist', 'Skai Jackson', 'Doja Cat']

celebrity_list0 = ['Mackenzie Foy', 'Lana Condor', 'Selita Ebanks', 'Jay Baruchel', 'Leslie Odom Jr.', 'Joey Fatone', 'Karen Elson', 'Penélope Cruz', 'Lily Donaldson', 'Lauren Daigle', 'James Garner', 'Chris Farley', 'Mark Consuelos', 'Eric Bana', 'Ray Liotta', 'Robert Kraft', 
                   'Riley Keough', 'Brody Jenner', 'Ross Lynch', 'Joel McHale', 'Melanie Martinez', 'Hunter Hayes', 'Gwendoline Christie', 'Zachary Levi', 'Troye Sivan', 'Marisa Tomei', 'Kevin James', 'Cuba Gooding Jr.', 'Sterling K. Brown', 'Rose Leslie', 'Clive Owen', 'Nick Offerman', 
                   'Ansel Elgort', 'Natalie Dormer', 'Terrence Howard', 'Sacha Baron Cohen', 'Joe Rogan', 'Olivia Culpo', 'Meg Ryan', 'Brendon Urie', 'Forest Whitaker', 'Matthew Perry', 'Sarah Paulson', 'Saoirse Ronan', 'Felicity Jones', 'Kurt Russell', 'Kacey Musgraves',
                   'Ashlee Simpson', 'Greta Thunberg', 'Carey Mulligan', 'Ashley Olsen', 'Martin Scorsese', 'Scott Disick', 'Victoria Justice', 'Mandy Moore', 'Jason Statham', 'Samuel L. Jackson', 'David Tennant', 'Jay Leno', 'Bob Ross', 'Bella Thorne', 'Steven Spielberg', 'Malcolm X', 
                   'Kendrick Lamar', 'Jessica Biel', 'Nick Jonas', 'Naomi Campbell', 'Mark Wahlberg', 'Cate Blanchett', 'Cameron Diaz', 'Dwayne Johnson', 'Floyd Mayweather', 'Oprah Winfrey', 'Ronald Reagan', 'Ben Affleck', 'Anne Hathaway', 'Stephen King', 'Johnny Depp', 'Abraham Lincoln', 'Kate Middleton', 'Donald Trump']

celebrities_with_few_images = ['Sajith Rajapaksa', 'Mia Challiner', 'Yasmin Finney', 'Gabriel LaBelle', 'Isabel Gravitt', 'Pardis Saremi', 'Dominic Sessa', 'India Amarteifio', 'Aryan Simhadri', 'Arsema Thomas', 'Sam Nivola', 'Corey Mylchreest', 'Diego Calva', 'Armen Nahapetian', 'Jaylin Webb', 
                               'Gabby Windey', 'Amaury Lorenzo', 'Kudakwashe Rutendo', 'Cwaayal Singh', 'Chintan Rachchh', 'Adwa Bader', 'Vedang Raina', 'Delaney Rowe', 'Aria Mia Loberti', 'Florence Hunt', 'Tom Blyth', 'Kris Tyson', 'Tioreore Ngatai-Melbourne', 'Cody Lightning', 'Mason Thames', 
                               'Samara Joy', 'Wisdom Kaye', 'Jani Zhao', 'Elle Graham', 'Priya Kansara', 'Boman Martinez-Reid', 'Park Ji-hu', 'Cara Jade Myers', 'Banks Repeta', 'Ali Skovbye', 'Nicola Porcella', 'Keyla Monterroso Mejia', 'Pashmina Roshan', 'Jeff Molina', 'Woody Norman', 'Leah Jeffries', 'Lukita Maxwell', 'Jordan Firstman', 'Josh Seiter', 'Ayo Edebiri']

to_be_added_celebrities = ['David Brinkley', 'Kate Chastain', 'Portia Freeman', 'Taylor Mills', 'Mary Fitzgerald', "Miles O'Brien", 'Andrew East', 'Una Stubbs', 'Nicola Coughlan', 'Matthew Bomer', 'Jim Walton', "Charli D'Amelio", 'Dan Lok', 'Nicholas Braun', 'Connor Cruise', 'Matt Stonie', 'Grant Achatz', 'Eiza González', 'Anna Cleveland', 'Melanie Iglesias', 
                               'Jacquetta Wheeler', 'Austin Abrams', 'Julia Stegner', 'Miles Heizer', 'Devon Sawa', 'Julie Chrisley', 'Barry Weiss', "Sha'Carri Richardson", 'Elliot Page', 'Emma Barton', 'Tommy Dorfman', 'Joe Lacob', 'Stephen tWitch Boss', 'Kendra Spears', 'Sam Elliot', 'Luka Sabbat', 'Hunter Schafer', 
                               'Marcus Lemonis', 'Danneel Harris', 'Ivan Moody', 'Tammy Hembrow', 'Ethan Suplee', 'Hunter McGrady', 'Mat Fraser', 'Sydney Sweeney', 'Isabel Toledo', 'Allison Stokke', 'Kim Zolciak-Biermann', 'Phoebe Dynevor', 'Jenna Ortega', 'Lew Alcindor', 'Emma Chamberlain', 'Summer Walker', 'Mariacarla Boscono', 'Justina Machado', 'Erin Foster', 
                               'Hero Fiennes-Tiffin', 'Douglas Brinkley', 'Sunisa Lee', 'Tina Knowles-Lawson', 'Peter Firth', 'Lauren Bush Lauren', 'Presley Gerber', 'Rachel Antonoff', 'Kieran Culkin', 'Annie LeBlanc', 'Michael Buffer', 'Jacquelyn Jablonski', 'Paz de la Huerta', 'Chris McCandless', 'Arthur Blank', 'Jay Ellis', 'Cacee Cobb', 'Ziyi Zhang', 'Indya Moore', 
                               'Bill Skarsgård', 'Thomas Doherty', 'Elisa Sednaoui', 'Margherita Missoni', 'Jared Followill', 'Taylor Tomasi Hill', 'Fei Fei Sun', 'Deana Carter', 'Hannah Bronfman', 'Anwar Hadid', 'Olivia Rodrigo', 'Frankie Grande', 'Diane Ladd', 'Sophia Lillis', 'Genesis Rodriguez', 'Richard Rawlings', 'Jill Wagner', 'Brady Quinn', 'Simu Liu', 'Diane Guerrero', 
                               'Matt McGorry', 'Hunter Parrish', 'Eddie Hall', 'Daria Strokous', 'Peter Hermann', 'Sasha Pivovarova', 'Liu Yifei', 'Danielle Jonas', 'Magdalena Frackowiak', 'Mark Richt', 'Crystal Renn', 'Amanda Gorman', 'Dan Crenshaw', 'Todd Chrisley', 'Eddie McGuire', 'Lori Harvey', 'Beanie Feldstein', 'David Muir', 'Martin Starr', 'Quvenzhané Wallis', 
                               'Lo Bosworth', 'Emma Corrin', 'Steve Lacy', 'Liza Koshy', 'Zak Bagans', 'Bob Morley', 'Jenna Marbles', 'Trevor Jackson', 'Jodie Turner-Smith', 'Michael Peña', 'Rhea Perlman', 'Luke Hemsworth', 'Rodrigo Santoro', 'Chris Klein', 'Lauren Bushnell', 'Oliver Hudson', 'Danny Thomas', 'Diego Boneta', 'Sara Foster', 'Michiel Huisman', 'Dan Bilzerian', 'Patricia Field', 
                               'Charles Melton', 'AJ Michalka', 'Stella Tennant', 'Lindsay Wagner', 'Justice Smith', 'Addison Rae', 'Scott Speedman', 'Christiano Ronaldo', 'Hanne Gaby Odiele', 'Michaela Coel', 'John Corbett', 'Lindsay Ellingson', 'Ryan Guzman', 'Marvin Sapp', 'John Wayne Gacy', 'Joyner Lucas', 'Rachelle Lefevre', 'Jessica Seinfeld', 'Sam Taylor Johnson', 
                               'Normani Kordei', 'Danielle Fishel', 'Robert Irvine', 'Sandra Cisneros', 'Alek Wek', 'Torrey Devitto', 'Christine Quinn', 'Jacob Sartorius', 'Chris Watts', 'Allison Holker', 'Mamie Gummer', 'Gregg Sulkin', 'Dree Hemingway', 'Alyson Stoner', 'Omar Epps', 'Jacob Elordi', 'David Dobrik', 'Cam Gigandet', 'Léa Seydoux', 'Tai Lopez', 'Alexander Skarsgård', 'Sarah Chalke', 
                               'Phoebe Bridgers', 'Lindsay Price', 'Josh Peck', 'Bernard Arnault']

set_of_prompts_human_face = [("a photorealistic close-up photograph of ", 1), ("High-resolution close-up image of ", 1), ("'s facial close-up", 2), ("Close-up headshot of ", 1), ("'s face portrait", 2)]


def get_slurm_cpus():
    # Default to 1 CPU if not running under Slurm or variables are undefined
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    num_tasks = int(os.getenv('SLURM_NTASKS', 1))

    # Calculate total CPUs allocated
    total_cpus = cpus_per_task * num_tasks
    return total_cpus


def get_slurm_gpus():
    """Returns the number of GPUs allocated by Slurm."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        # Split by comma and count the number of GPU IDs listed
        num_gpus = len(cuda_visible_devices.split(","))
    else:
        # If CUDA_VISIBLE_DEVICES is not set, assume no GPUs are allocated
        num_gpus = 0
    return num_gpus


def sharpness_of_detected_face(img_, facial_area):
    import cv2
    # Use the bounding box coordinates from RetinaFace
    x1, y1, x2, y2 = facial_area
    if not isinstance(x1, int):
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        ## convert the float to int, by rounding it to the nearest integer
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > img_.shape[1]: x2 = img_.shape[1]
    if y2 > img_.shape[0]: y2 = img_.shape[0]
    
    # Crop the face from the image
    face = img_[y1:y2, x1:x2]

    # Apply Laplacian filter to the face
    laplacian = cv2.Laplacian(face, cv2.CV_64F)

    # Calculate the variance
    variance = np.var(laplacian)

    # Now you can use the variance as a measure of sharpness
    # print("Sharpness variance:", variance)
    return variance


def find_num_faces(args, model_returned_values, print_stuff=True, img_path=None):
    import cv2
    values = model_returned_values
    if isinstance(values, dict) and args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
        num_faces = len(values.keys())
        face_area_percentages = []
        if print_stuff: print("Number of face detected: ", len(values.keys()))
        img_loaded = cv2.imread(img_path)
        
        for seq, key in enumerate(values):      ## This iterates over each detected face in the image
            if print_stuff:
                print("Face ", seq+1, " Score: ", values[key]["score"])
                print("Facial area: ", values[key]["facial_area"])
                print("Original image shape: ", values[key]["original_image_shape"])
            assert len(values[key]["original_image_shape"]) == 3 and values[key]["original_image_shape"][2] == 3        ## the last is the color channel. 
            # print(values[key]["original_image_shape"])
            original_image_area = values[key]["original_image_shape"][0] * values[key]["original_image_shape"][1]           ## changed to to original image area instead of upscaled image area
            detected_face_area = (values[key]["facial_area"][2] - values[key]["facial_area"][0]) * (values[key]["facial_area"][3] - values[key]["facial_area"][1])      ## this assumes that the output is of the format: (x1, y1, x2, y2)      ## confirm this -- yes done. See the code for extract_faces in retinaface.py
            assert detected_face_area >= 0 and detected_face_area <= original_image_area
            if print_stuff: print("Facial area as percentage of image total area: ", detected_face_area * 100.0/original_image_area)
            this_face_area_percentage = detected_face_area * 100.0/original_image_area

            # ## save the area of the detected face in "face.jpg"
            # x1, y1, x2, y2 = values[key]["facial_area"]
            # face = img_loaded[y1:y2, x1:x2]
            # cv2.imwrite("face.jpg", face)

            sharpness = sharpness_of_detected_face(img_loaded, values[key]["facial_area"])
            if print_stuff: print("Sharpness variance:", sharpness)

            ## add both area and sharpness. 
            face_area_percentages.append((this_face_area_percentage, sharpness))
        if print_stuff: print("---------------------------------------------------")
        return num_faces, face_area_percentages, original_image_area

    elif isinstance(values, list) and args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:      ## this is from insightface
        num_faces = len(values)
        face_area_percentages = []
        if print_stuff: print("Number of face detected: ", len(values))
        img_loaded = cv2.imread(img_path)
        original_image_area = img_loaded.shape[0] * img_loaded.shape[1]
        
        for seq, val in enumerate(values):  ## values is a dict with keys: 'bbox', 'landmark', 'embedding', 'gender', 'age', 'pose'. Original_image_shape is not present in this dict.
            if print_stuff:
                print("Face ", seq+1)
                print("Facial area: ", val["bbox"])
            detected_face_area = (val["bbox"][2] - val["bbox"][0]) * (val["bbox"][3] - val["bbox"][1])
            assert detected_face_area >= 0 and detected_face_area <= original_image_area, f"assertion failed in find_num_faces: {img_path}"
            if print_stuff: print("Facial area as percentage of image total area: ", detected_face_area * 100.0/original_image_area)
            assert original_image_area > 0, f"assertion failed in find_num_faces: original_image_area is <= 0 for {img_path}"
            this_face_area_percentage = detected_face_area * 100.0/original_image_area
            # sharpness = sharpness_of_detected_face(img_loaded, val["bbox"])
            # if print_stuff: print("Sharpness variance:", sharpness)
            # face_area_percentages.append((this_face_area_percentage, 0))
            face_area_percentages.append((detected_face_area, 0))       ## why do we care about the percentage of the face area rather than the actual area, let's just get the actual area.
        if print_stuff: print("---------------------------------------------------")
        return num_faces, face_area_percentages, original_image_area
        
    else:
        assert args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
        assert len(values[0][1]) == 4 and values[0][1][0] == 1
        if print_stuff:
            original_image_area = values[0][1][1] * values[0][1][2]
            if print_stuff: print("No face detected in image in image of size: ", original_image_area)
        if print_stuff: print("---------------------------------------------------")
        return 0, [], 0


def demo_example_images_face_detection():
    if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
        from retinaface import RetinaFace
    elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
        import cv2
        from insightface.app import FaceAnalysis
        insightfaceapp = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
    else:
        raise Exception('Please specify a valid face recognition model')
    
    model = None
    # img1 = "../downloaded_images/Charles Leclerc/image_seq_5_1773892.jpg"
    # img2 = "../generated_images/Oprah Winfrey/a photorealistic image of /image_seq_23.png"
    # img3 = "../downloaded_images/Donald Trump/image_seq_73_Donald-Trump.jpg"
    # img4 = "../downloaded_images/Oprah Winfrey/image_seq_326_256f63a288d02c63_474331083_10.jpg.xxxlarge.jpg"
    # img1 = "../generated_images/Bella Thorne/a photorealistic close-up photograph of /image_seq_7.png"
    img11 = "../generated_images/Martin Scorsese/a photorealistic close-up photograph of /image_seq_150.png"
    img12 = "/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/Sajith-Rajapaksa/Sajith-Rajapaksa-2.jpg"
    img13 = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Miguel Bezos/image_seq_4_slide_49.jpg"     ## no face here
    
    img1_path = "../generated_images/Oprah Winfrey/a photorealistic image of /image_seq_23.png"
    img2_path = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Oprah Winfrey/image_seq_56_oprah_035246.jpg"
    img3_path = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Donald Trump/image_seq_73_Donald-Trump.jpg"
    img4_path = '/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Thandiwe Newton/image_seq_53_Thandiwe-Newton.jpg?ve=1&tl=1_.jpeg'
    
    # import ipdb; ipdb.set_trace()
    if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
        resp1, model = RetinaFace.detect_faces(img1, model=model)
    elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
        image_here = cv2.imread(img11)
        resp1 = insightfaceapp.get(image_here)
    else:   raise NotImplementedError
    find_num_faces(args, resp1, img_path=img11)
    print(resp1, type(resp1))
    
    # ## save the area of the detected face in "face.jpg"
    # x1, y1, x2, y2 = values[key]["facial_area"]
    # face = img_loaded[y1:y2, x1:x2]
    # cv2.imwrite("face.jpg", face)
    
    # resp2, model = RetinaFace.detect_faces(img11, model=model)
    # print(resp2, type(resp2))
    # resp3, model = RetinaFace.detect_faces(img12, model=model)
    # print(resp3, type(resp3))
    # resp4, model = RetinaFace.detect_faces(img13, model=model)
    # print(resp4, type(resp4))
    # respss = [(resp1, img1), (resp2, img2), (resp3, img3), (resp4, img4)]
    # import ipdb; ipdb.set_trace()
    # for i, img_path in resp3:
    #     find_num_faces(args, i, img_path=img_path)


def plot_faces_per_image(args, no_face, one_face, two_face, more_than_two_face):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # plt.style.use('seaborn-whitegrid')
    # plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(20, 10))
    plt.xticks([])
    plt.yticks(np.arange(0, 201, 10))
    ## do not print the name of celebrities on the x-axis. Do not print anythig on the x-axis. 
    ## Make 4 line plots with x as the celebrities and y being the number of images with 0, 1, 2, and more than 2 faces.
    plt.plot(list(no_face.keys()), list(no_face.values()), label='0 faces', color='red', marker='d', linewidth=3)
    plt.plot(list(one_face.keys()), list(one_face.values()), label='1 face')
    plt.plot(list(two_face.keys()), list(two_face.values()), label='2 faces', color='green', marker='s', linewidth=3)
    plt.plot(list(more_than_two_face.keys()), list(more_than_two_face.values()), label='more than 2 faces', color='black', marker='o', linewidth=3)
    plt.xlabel('Celebrities')
    plt.ylabel('Number of images with 0, 1, 2, and more than 2 faces')

    df = pd.read_csv('/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrities_sorted.csv', sep=',', header=0)
    df = df.set_index('Name')
    df = df.loc[celebrity_list]
    df = df.reset_index()
    df = df.rename(columns={'index': 'Name'})
    assert df.shape == (len(celebrity_list), 2)
    # import ipdb; ipdb.set_trace()
    df['counts_in_laion2b-en'] = df['counts_in_laion2b-en'].astype(int)

    ## For celebrity whose have more than 20 two faces or 20 0 faces, print their as the x-tick label.
    for i, celeb in enumerate(celebrity_list):
        threshold = 23
        if no_face[celeb] >= threshold or two_face[celeb] >= threshold or more_than_two_face[celeb] >= threshold:
            y_pos = max(no_face[celeb], two_face[celeb], more_than_two_face[celeb]) - 8
            print(celeb, no_face[celeb], two_face[celeb], more_than_two_face[celeb])
            plt.text(i, y_pos, (celeb, df[df['Name']==celeb]['counts_in_laion2b-en'].values[0]), rotation=20, fontsize=15, ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'num_face_for_all_celebs_{"training" if args.training else "generated"}.png')


def __call_face_detection__(args, images_directory, celeb, output_file_statistics, output_file_each_image_info, model, print_stuff):
    ## if the celeb is already in the statistics file, then skip this celeb.
    if os.path.exists(output_file_statistics):
        statistics_df = pd.read_csv(output_file_statistics, sep="|", header=0)
        if celeb in statistics_df['celeb'].values:
            print(f"{celeb} is already in the statistics file. Skipping this celeb.")
            return
    
    import cv2
    
    if not args.parallelize_across_one_entity:
        images_and_faces = {}
        no_face = 0
        one_face = 0
        two_face = 0
        more_than_two_face = 0
        total_images = 0
        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
            from retinaface import RetinaFace
            model = None
        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
            insightfaceapp = model
        else:
            raise Exception('Please specify a valid face recognition model')

        for _, image in enumerate(os.listdir(images_directory)):
            if "grid_images" in image:      continue        ## we don't want to include the grid images
            if ".csv" in image[-4:]:      continue        ## we don't want to include the csv files
            if "open_clip_image_features" in image:      continue        ## we don't want to include .pt files
            if ".json" in image[-5:]:      continue        ## we don't want to include the json files
            total_images += 1
            printed_name = f'{images_directory}/{image}'
            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                resp, model = RetinaFace.detect_faces(printed_name, model=model)    ## the bounding box is given as (x1, y1, x2, y2)
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                image_here = cv2.imread(printed_name)
                if image_here is None:
                    print(f"Image {printed_name} is not readable. Skipping this image.")
                    continue
                resp = insightfaceapp.get(image_here)       ## the bounding box is given as (x1, y1, x2, y2)

            num_face, face_area_percentages, original_image_area = find_num_faces(args, resp, print_stuff=False, img_path=printed_name)
            images_and_faces[image] = (num_face, face_area_percentages, original_image_area)
            
            if num_face == 0:
                no_face += 1
            elif num_face == 1:
                one_face += 1
            elif num_face == 2:
                two_face += 1
            else:
                if print_stuff: print(f"{num_face} faces detected in image: ", printed_name)
                more_than_two_face += 1

    else:
        assert model is None, "model should be None when args.parallelize_across_one_entity is True"
        
        def worker_gpu_face_detection(gpu_id, data_subset, return_dict):
            no_face_this_gpu = 0
            one_face_this_gpu = 0
            two_face_this_gpu = 0
            more_than_two_face_this_gpu = 0
            total_images_this_gpu = 0
            images_and_faces_this_gpu = {}

            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                model = None
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                from insightface.app import FaceAnalysis
                insightfaceapp = FaceAnalysis(providers=[('CUDAExecutionProvider', {'device_id': gpu_id})])
                insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                model = insightfaceapp
            else:
                raise Exception('Please specify a valid face recognition model')
            
            for _, image in enumerate(data_subset):
                if "grid_images" in image:      continue
                if ".csv" in image[-4:]:      continue
                if "open_clip_image_features" in image:      continue
                if ".json" in image[-5:]:      continue
                total_images_this_gpu += 1
                printed_name = f'{images_directory}/{image}'
                if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                    resp, model = RetinaFace.detect_faces(printed_name, model=model)
                elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                    image_here = cv2.imread(printed_name)
                    resp = insightfaceapp.get(image_here)
                    
                num_face, face_area_percentages, original_image_area = find_num_faces(args, resp, print_stuff=False, img_path=printed_name)
                images_and_faces_this_gpu[image] = (num_face, face_area_percentages, original_image_area)
                
                if num_face == 0:
                    no_face_this_gpu += 1
                elif num_face == 1:
                    one_face_this_gpu += 1
                elif num_face == 2:
                    two_face_this_gpu += 1
                else:
                    more_than_two_face_this_gpu += 1
        
            # Storing results in the manager's dictionary
            return_dict[gpu_id] = (no_face_this_gpu, one_face_this_gpu, two_face_this_gpu, more_than_two_face_this_gpu, total_images_this_gpu, images_and_faces_this_gpu)

        ## here will will parallelize the face detection process across the GPUs for the images of this celeb. One process per GPU.
        def split_data(data, num_splits):
            """Splits data into sublists."""
            k, m = divmod(len(data), num_splits)
            return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits))
        
        num_gpus = get_slurm_gpus()
        assert num_gpus > 0, 'No GPUs allocated'
        data_to_process = os.listdir(images_directory)
        images_subset_total = list(split_data(data_to_process, num_gpus))

        with Manager() as manager:
            return_dict = manager.dict()
            processes = []
            print(f"processing the images for {celeb} with {num_gpus} GPUs. Total images: {len(data_to_process)}")
            for gpu_id, images_subset in enumerate(images_subset_total):
                p = Process(target=worker_gpu_face_detection, args=(gpu_id, images_subset, return_dict))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Aggregate results
            no_face = sum(res[0] for res in return_dict.values())
            one_face = sum(res[1] for res in return_dict.values())
            two_face = sum(res[2] for res in return_dict.values())
            more_than_two_face = sum(res[3] for res in return_dict.values())
            total_images = sum(res[4] for res in return_dict.values())
            images_and_faces = {}
            for res in return_dict.values():
                images_and_faces.update(res[5])
        
        
    # print(f"{celeb}'s number of images with no face: ", no_face[celeb])
    # print(f"{celeb}'s number of images with one face: ", one_face[celeb])
    # print(f"{celeb}'s number of images with two faces: ", two_face[celeb])
    # print(f"{celeb}'s number of images with more than two faces: ", more_than_two_face[celeb])

    ## now print the output for this celeb in the csv file. Make it pipe + tab separated.
    with open(output_file_statistics, "a") as f:
        f.write(f"{celeb}|{no_face}|{one_face}|{two_face}|{more_than_two_face}|{total_images}\n")

    ## Also print each image info for each celeb. Make it pipe + tab separated.
    with open(output_file_each_image_info, "a") as f:
        print(f"Looping over {celeb} with {len(images_and_faces)} images.")
        for image in images_and_faces:
            num_faces_this_image = images_and_faces[image][0]
            face_area_percentages_and_sharpness = images_and_faces[image][1]
            ## we want to store the face area percentages and sharpness in different columns. Both these columns will be a list of values.
            face_area_percentages = [i[0] for i in face_area_percentages_and_sharpness]
            sharpness = [i[1] for i in face_area_percentages_and_sharpness]
            image_area = images_and_faces[image][2]
            f.write(f"{celeb}|{image}|{num_faces_this_image}|{face_area_percentages}|{sharpness}|{image_area}\n")


def scaled_up_face_detection(args, celeb, output_file_statistics, output_file_each_image_info, model, print_stuff=False):
    ## Goal is to count the number of generated images that has 0 faces, 1 face, 2 face, and more than 2 faces for each celeb, and then plot it in a line plot
    if args.training:
        images_directory = f'/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb}'      ## we have created a soft link to the downloaded_celebrity_images directory
    
    elif args.generated:
        # images_directory = f'../generated_images/{celeb}/a photorealistic close-up photograph of '
        images_directory = f'../generated_images/{celeb}/{args.image_generation_prompt}'
    
    elif args.images_I_downloaded_from_internet:
        ## Now instead of downloading them manually, I am downloading images automatically -- cool! So now we need to run face detection on them to filter them
        images_directory = f'../high_quality_images_downloaded_from_internet/{celeb.replace(" ", "-")}'
    
    elif args.all_laion_images:
        if args.set_of_people == "celebrity":
            images_directory = f'/gscratch/h2lab/vsahil/vlm-efficiency/all_downloaded_images/{celeb}'
        elif args.set_of_people == "politicians":
            images_directory = f"/gscratch/scrubbed/vsahil/all_downloaded_images_politicians/{celeb}"
        else:
            raise Exception('Please specify either --celebrity or --politicians')

    elif args.all_laion_alias_name_images:
        ## get all aliases of this celeb
        alias_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrity_counts_with_aliases.csv", sep="|")
        alias_df = eval(alias_df[alias_df['celeb'] == celeb]['aliases_and_counts'].values[0])
        assert isinstance(alias_df, dict)
        if len(alias_df) == 0:
            print(f"{celeb} has no aliases. Skipping this celeb.")
            return
    
    elif args.all_images_internet:
        images_directory = f'/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(" ", "-")}'
    
    else:
        raise Exception('Please specify either --training or --generated')
    
    if args.all_laion_alias_name_images:
        for alias_celeb in alias_df:
            images_directory = f'/gscratch/scrubbed/vsahil/all_downloaded_images/{alias_celeb}'
            assert os.path.exists(images_directory), f"Directory {images_directory} does not exist. for {celeb} and {alias_celeb}"
            print(celeb  + "__" + alias_celeb, len(os.listdir(images_directory)), images_directory)
            __call_face_detection__(args, images_directory, celeb  + "__" + alias_celeb, output_file_statistics, output_file_each_image_info, model, print_stuff)
    else:
        print(celeb, len(os.listdir(images_directory)), images_directory)
        if not os.path.exists(images_directory):
            print(f"Directory {images_directory} does not exist. Skipping this celeb.")
            return
        __call_face_detection__(args, images_directory, celeb, output_file_statistics, output_file_each_image_info, model, print_stuff)


def select_single_person_and_large_training_images(args, celebrity_list_here):
    ## for the all images in result_faces_training_each_image_info.csv, select the images that have 1 face, and then sort the df by the largest face area percentage.
    if args.training:
        file_each_image_info = f"result_faces_training_each_image_info_{args.set_of_people}.csv"
        output_file = f"close_up_single_person_training_images_{args.set_of_people}.csv"
    elif args.generated:
        args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
        file_each_image_info = f"result_faces_generated_{args.image_generation_prompt}_each_image_info_{args.set_of_people}.csv"
        output_file = f"close_up_single_person_generated_{args.image_generation_prompt}_images_{args.set_of_people}.csv"
    elif args.images_I_downloaded_from_internet:
        file_each_image_info = f"result_faces_images_I_downloaded_from_internet_each_image_info_{args.set_of_people}.csv"
        output_file = f"close_up_single_person_images_I_downloaded_from_internet_{args.set_of_people}.csv"
    elif args.all_images_internet:
        file_each_image_info = f"result_faces_all_images_internet_each_image_info_{args.set_of_people}.csv"
        output_file = f"close_up_single_person_all_images_internet_{args.set_of_people}.csv"
    else:
        raise Exception('Please specify either --training or --generated')
    
    df = pd.read_csv(file_each_image_info, sep='|', header=0)

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("celeb|image|effective_pixels|sharpness\n")

    if args.training:
        entities_for_serpapi = []
    for celeb in celebrity_list_here:
        df_this_celeb = df[df['celeb'] == celeb]
        print(len(df_this_celeb), "total images for celeb: ", celeb)
        df_this_celeb = df_this_celeb[df_this_celeb['num_faces']==1]
        df_this_celeb = df_this_celeb.drop_duplicates(subset=['image'])
        print(len(df_this_celeb), "images with 1 face for celeb: ", celeb)
        if len(df_this_celeb) < 10 and args.training:
            entities_for_serpapi.append(celeb)
        df_this_celeb['face_area_percentages'] = df_this_celeb['face_area_percentages'].apply(lambda x: [float(i) for i in x[1:-1].split(',')][0])      ## remember that the face_area_percentages is a string of a list of floats. So we need to convert it to a list of floats.
        df_this_celeb['sharpness'] = df_this_celeb['sharpness'].apply(lambda x: [float(i) for i in x[1:-1].split(',')][0])
        df_this_celeb['image_area'] = df_this_celeb['image_area'].apply(lambda x: float(x))
        if args.all_images_internet:
            df_this_celeb['effective_learnable_pixels'] = df_this_celeb['face_area_percentages'] ## this is because we changed code for the internet images to only store the face area pixels, not as a percentage of the image area.
            df_this_celeb['effective_learnable_pixels_group'] = df_this_celeb['effective_learnable_pixels'] // 100
        else:
            df_this_celeb['effective_learnable_pixels'] = df_this_celeb['face_area_percentages'] * df_this_celeb['image_area'] / 100.0
            df_this_celeb['effective_learnable_pixels_group'] = df_this_celeb['effective_learnable_pixels'] // 100
        df_this_celeb = df_this_celeb.sort_values(by=['effective_learnable_pixels_group','sharpness'], ascending=[False, False]).drop(columns=['effective_learnable_pixels_group'])
        with open(output_file, "a") as f:
            for image_path in df_this_celeb['image'].values:
                f.write(f"{celeb}|{image_path.strip()}|{df_this_celeb[df_this_celeb['image']==image_path]['effective_learnable_pixels'].values[0]}|{df_this_celeb[df_this_celeb['image']==image_path]['sharpness'].values[0]}\n")
    
    if args.training:
        pd.DataFrame(entities_for_serpapi).to_csv(f"entities_for_serpapi_{args.set_of_people}.csv", columns=['Name'], index=False)


def select_real_photographs_from_single_person_and_large_training_images(args, return_values=False, celebrity_list_total=None):
    
    def process_files(df_questions, log_file_here, yes_vqa_answers):
        # import ipdb; ipdb.set_trace()
        ## assert that the length of log_file_here is the same as the number of rows in df_questions
        assert len(df_questions) == sum(1 for line in open(log_file_here))
        with open(log_file_here, "r") as f:
            for line in f:
                line = line.strip()
                image_path_in_answer_log = line.split("Image File:")[1].split("Question:")[0].strip()
                question_id = int(line.split("Question ID:")[1].split("Image ID:")[0].strip())
                image_id = int(line.split("Image ID:")[1].split("Image File:")[0].strip())
                answer = line.split("Answer:")[1].strip()
                # line = line.split()
                # question_id = int(line[2])
                # image_id = int(line[5])
                # answer = line[-1]
                assert answer in ['yes', 'no'], f"answer is: {answer}"
                assert question_id == image_id
                assert question_id in df_questions['uniq_id'].values

                image_path = df_questions[df_questions['uniq_id'] == question_id]['image'].values[0]
                assert image_path_in_answer_log == image_path
                image_path = image_path.split('/')
                celeb = image_path[-2]
                image = image_path[-1]
                if celeb not in yes_vqa_answers:
                    yes_vqa_answers[celeb] = []
                assert image not in yes_vqa_answers[celeb]
                if answer == 'yes':
                    yes_vqa_answers[celeb].append(image)

        if return_values:
            return yes_vqa_answers
        
        for celeb in yes_vqa_answers:
            print(celeb, ' has ', len(yes_vqa_answers[celeb]), ' images that are real photographs.')

        ## print each celeb's images to a new file
        with open("close_up_single_person_images_filtered_real_photographs.csv", "a") as f:
            for celeb in yes_vqa_answers:
                for image in yes_vqa_answers[celeb]:      ##[:100] Do a max of 100 images per celeb - no need now as we are not doing the expensive n^2 search -- we are doing it smartly. 
                    image_path = f"/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb}/{image}"
                    f.write(f"{celeb}|{image_path}\n")
        
    ## This function will read in the "close_up_single_person_images.tsv" and filter down the images that the VQA model has said yes for a real photograph. VQA model answers are stored in model_vqa_single_person_answers.log (each line looks like: Question ID: 0 Image ID:  0 Question:   is this a real photograph of a person?  Answer:  no)
    ## We will store the filtered images in a new file: "close_up_single_person_images_filtered_real_photographs.tsv"
    assert args.training
    # if len(celebrity_list_total) == 0:
    #     return
    # elif len(celebrity_list_total) == 70:
    #     ## this is the less popular celebs list
    #     question_file = [f"vqa_pic_or_paint.tsv"]
    #     log_file = ["model_vqa_single_person_answers_less_popular_celebs.log"]
    # elif len(celebrity_list_total) == 98:
    #     ## this is for the original list of 98 celebs
    #     question_file = [f"vqa_pic_or_paint0.tsv"]
    #     log_file = ["model_vqa_single_person_answers.log"]
    # elif len(celebrity_list_total) == 168:
    #     ## here the two lists were combined
    #     if celebrity_list_total[0] == 'Aditya Kusupati':
    #         question_file = [f"vqa_pic_or_paint.tsv", f"vqa_pic_or_paint0.tsv"]
    #         log_file = ["model_vqa_single_person_answers_less_popular_celebs.log", "model_vqa_single_person_answers.log"]
    #     elif celebrity_list_total[0] == 'Dan Peña':
    #         question_file = [f"vqa_pic_or_paint0.tsv", f"vqa_pic_or_paint.tsv"]
    #         log_file = ["model_vqa_single_person_answers.log", "model_vqa_single_person_answers_less_popular_celebs.log"]
    question_file = ["/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/vqa_pic_or_paint_with_image_paths_changed.tsv"]       # we reran the entire pipeline. 
    log_file = ["/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/vqa_pic_or_paint_with_image_paths_changed_log_directory/total_output.log"]

    ## First read in the VQA model answers
    yes_vqa_answers = {}
    for question_file_here, log_file_here in zip(question_file, log_file):
        df_questions = pd.read_csv(question_file_here, sep='\t', header=0)        # uniq_id	image_id	image	question	refs. Here uniq_id is the question_id
        yes_vqa_answers = process_files(df_questions, log_file_here, yes_vqa_answers)
    
    return yes_vqa_answers


def face_recognition_for_high_quality_training_images(args, celeb_here, output_file, second_output_file,  matching_threshold, do_not_write_file=False):
    '''
    This function's job is to perform face recognition task on each pair of training image that are provided to it from the filtered set of images: single person images, sorted by the product of image size and face area percentage, and then passed through VQA model. 
    We will give a maximum of 50 images per celeb to this function. This function will do ((n^2 - n)/2) image comparisons. It will return the sorted order of images, sorted in the decreasing order of the number of images they matched to. 
    The idea is that in the high quality images, the outliers (which is basically good images of other celebs having captions of another celebrity) will have low number of matches. 
    '''
    if args.training:
        df = pd.read_csv(f"close_up_single_person_training_images_{args.set_of_people}.csv", sep='|', header=0)       # celeb|image
    elif args.images_I_downloaded_from_internet:
        df = pd.read_csv(f"close_up_single_person_images_I_downloaded_from_internet_{args.set_of_people}.csv", sep='|', header=0)       # celeb|image|effective_pixels|sharpness
    else:
        raise NotImplementedError

    df_this_celeb = df[df['celeb'] == celeb_here].copy()
    if len(df_this_celeb) == 0:
        print(f"celeb {celeb_here} not found in the dataframe.")
        return 0, 0

    print(f"Total images for celeb {celeb_here}: ", len(df_this_celeb))

    ## now we have to do ((n^2 - n)/2) image comparisons. The result we be stored in csv file with the columns as celeb, image1, image2, whether they matched or not, and the total number of matches for the image1 (this will be same for all the rows in which the image1 is present).
    image_representations = []
    sequenced_images = []

    if args.training:
        with open(f"{args.face_recognition_model}_embeddings_close_up/training/{celeb_here}.json", "r") as f:
            this_celeb_training_image_embeddings = json.load(f)
    elif args.images_I_downloaded_from_internet:
        with open(f"{args.face_recognition_model}_embeddings_close_up/images_I_downloaded_from_internet/{celeb_here}.json", "r") as f:
            this_celeb_training_image_embeddings = json.load(f)

    for image_path in df_this_celeb['image'].values:
        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
            image_representations.append(this_celeb_training_image_embeddings[image_path][0]['embedding'])    
        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
            image_representations.append(this_celeb_training_image_embeddings[image_path])
        sequenced_images.append(image_path)

    image_representations = np.array(image_representations)
    assert image_representations.shape[0] == len(df_this_celeb)

    from scipy.spatial.distance import cosine
    image_representations = image_representations / np.linalg.norm(image_representations, axis=1, keepdims=True)        ## normalize the embeddings
    cosine_similarity = np.dot(image_representations, image_representations.T)  ## compute the cosine distance of each image with every other image.
    cosine_similarity = np.clip(cosine_similarity, -1, 1)   # Ensure the similarity is within proper bounds due to floating point errors.

    assert cosine_similarity.shape == (len(df_this_celeb), len(df_this_celeb))
    ## now we have the cosine distance between each image and every other image. We will now sort the images based on the number of images they matched to. Let's use the threshold of 0.4 for match and higher than that for non-match.
    if args.training:
        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
            thresholded_distances = np.where(cosine_similarity > 0.4, 1, 0)
        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
            thresholded_distances = np.where(cosine_similarity > 0.4, 1, 0)
    elif args.images_I_downloaded_from_internet:
        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
            thresholded_distances = np.where(cosine_similarity > 0.4, 1, 0)
        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
            thresholded_distances = np.where(cosine_similarity > 0.4, 1, 0)


    ## now we have the thresholded distances. We will now sum the number of matches for each image.
    num_matches = np.sum(thresholded_distances, axis=1)
    assert num_matches.shape[0] == len(df_this_celeb)
    ## now we have the number of matches for each image. We will now sort the images based on the number of matches.
    sorted_indices = np.argsort(num_matches)[::-1]
    sorted_sequenced_images = [sequenced_images[i] for i in sorted_indices]
    sorted_num_matches = [num_matches[i] for i in sorted_indices]
    ## now we have the sorted indices. We will now print the images in the sorted order.

    if not do_not_write_file:
        with open(output_file, "a") as f:
            ## get the number of images that have atleast 40% of the matches of the total number of images.
            count_high_quality_images = np.sum(np.array(sorted_num_matches) >= matching_threshold * len(df_this_celeb))      ## we did 40% so that Donald Trump also gets some image, other he had 0 images.
            print(f"Number of high quality images for celeb {celeb_here}: ", count_high_quality_images)
            if args.training:
                num_downloaded_training_images_this_celeb = len(os.listdir(f"../downloaded_images/{celeb_here}"))
            elif args.images_I_downloaded_from_internet:
                num_downloaded_training_images_this_celeb = 100
            f.write(f"{celeb_here}|{count_high_quality_images}|{len(df_this_celeb)}|{num_downloaded_training_images_this_celeb}\n")

    df_this_celeb['num_matches'] = 0
    for seq, i in enumerate(sorted_sequenced_images):
        if sorted_num_matches[seq] < matching_threshold * len(df_this_celeb):        break       ## We only want to add the number of matches for the images that have atleast 50% of the matches of the total number of images.
        df_this_celeb.loc[df_this_celeb['image']==i, 'num_matches'] = sorted_num_matches[seq]
    
    if args.training:
        df_this_celeb['image'] = df_this_celeb['image'].apply(lambda x: f"../downloaded_images/{celeb_here}/{x}")
        df_this_celeb['image'] = df_this_celeb['image'].apply(lambda x: x[:-5] + '.jpg' if x[-5:] == '.jpeg' else x)
    elif args.images_I_downloaded_from_internet:
        pass    # df_effective_pixels_and_shape_this_celeb['image'] = df_effective_pixels_and_shape_this_celeb['image'].apply(lambda x: f"../high_quality_images_downloaded_from_internet/{celeb_here}/{x}") -- do not add this
    
    ## now we want to get make groups of sorted number of matches (say into 4 quantiles). And then within each group, we sort the images by the descending order of the number of effective pixels. The outer sorting is based on number of matches, and the inner sorting is based on the number of effective pixels. 
    ## Divide the number of matches into 4 quantiles. The highest number of match is the first element of the list sorted_num_matches.
    ## we will use the pandas qcut function to divide the number of matches into 4 quantiles. -- no we will use the cut function as we want to divide the number of matches into 4 groups, and not 4 quantiles.
    if args.training:
        df_this_celeb['num_matches_group'] = pd.cut(df_this_celeb['num_matches'], bins=5, labels=False)
    elif args.images_I_downloaded_from_internet:
        df_this_celeb['num_matches_group'] = pd.cut(df_this_celeb['num_matches'], bins=4, labels=False)       ## we want more importance on effective pixels are these are usually higher quality internet images. 
    ## now we have the number of matches group. We will now sort the images based on the number of effective pixels within each group. We want descending order for both the number of matches group and the number of effective pixels.
    df_this_celeb = df_this_celeb.sort_values(by=['num_matches_group','effective_pixels'], ascending=[False, False]).drop(columns=['num_matches_group'])
    
    if not do_not_write_file:
        with open(second_output_file, "a") as f:
            for i in df_this_celeb['image'].values:
                num_matches_this_image = df_this_celeb[df_this_celeb['image']==i]['num_matches'].values[0]
                if num_matches_this_image > 0:
                    effective_pixels = df_this_celeb[df_this_celeb['image']==i]['effective_pixels'].values[0]
                    sharpness = df_this_celeb[df_this_celeb['image']==i]['sharpness'].values[0]
                    f.write(f"{celeb_here}|{i}|{num_matches_this_image}|{effective_pixels}|{sharpness}\n")

    return count_high_quality_images, len(df_this_celeb)


def demo_face_recognition(args):
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    from deepface import DeepFace
    #face verification
    img1_path = "../generated_images/Oprah Winfrey/a photorealistic image of /image_seq_23.png"
    img2_path = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Oprah Winfrey/image_seq_56_oprah_035246.jpg"
    img3_path = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Donald Trump/image_seq_73_Donald-Trump.jpg"
    img4_path = '/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Thandiwe Newton/image_seq_53_Thandiwe-Newton.jpg?ve=1&tl=1_.jpeg'
    img5_path = "/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/Charles Leclerc/image_seq_5_1773892.jpg"
    import ipdb; ipdb.set_trace()
    # result = DeepFace.verify(img1_path, img2_path, model_name = models[2])
    img1_embedding, _ = DeepFace.represent(img1_path, model_name = models[0], detector_backend='retinaface', enforce_detection=False, align=False)
    # print(result)
    print(len(img1_embedding))
    # import ipdb; ipdb.set_trace()
    img5_embedding, _ = DeepFace.represent(img5_path, model_name = models[0], detector_backend='retinaface', enforce_detection=False, align=False)
    print(len(img5_embedding), " with retinface")

    img4_embedding, _ = DeepFace.represent(img4_path, model_name = models[0], detector_backend='opencv', enforce_detection=False, align=False)      ## Earlier I was using this detector for the face which is the default detector, and is not good. Replaced that with retinaface, so we need to regenerate the embeddings for the generated images. 
    print(len(img4_embedding), " with opencv")
    
    img2_embedding, _ = DeepFace.represent(img2_path, model_name = models[0])
    img3_embedding, _ = DeepFace.represent(img3_path, model_name = models[0])
    # import ipdb; ipdb.set_trace()
    print(len(img2_embedding), len(img3_embedding), "last two images")
    

    # 
    ## get the cosine distance between the embeddings of im1 and im2, and im1 and im3
    # from scipy.spatial.distance import cosine
    # print(cosine(img1_embedding[0]['embedding'], img2_embedding[0]['embedding']))
    # print(cosine(img1_embedding[0]['embedding'], img3_embedding[0]['embedding']))
    # print(cosine(img2_embedding[0]['embedding'], img3_embedding[0]['embedding']))


def view_good_training_images(args):
    import subprocess
    from PIL import Image
    ## This function will open the top-k training images for the the selected celebs and open them in VSCode for manual inspection. I want to see if the selected images of the celebs that we want or not, and if they are good or not.
    # df_single_person_training_images = pd.read_csv("all_celebs_single_person_training_data_face_recognition_result_summarized.csv", sep='|', header=0)
    df_single_person_training_images = pd.read_csv("all_celebs_single_person_internet_images_downloaded_face_recognition_result_summarized.csv", sep='|', header=0)
    df_single_person_training_images = df_single_person_training_images.set_index('celeb')
    df_num_matches = pd.read_csv("all_celebs_single_person_internet_images_downloaded_face_recognition_result.csv", sep='|', header=0)      # celeb|count_high_quality_images|total_single_face_real_photograph_images|num_downloaded_training_images
    ## Nico Tortorella has some variability in hair colors. 
    ## Kevin James's images has some variability in terms of facial hair styles.    'Nico Tortorella', 'Chris Farley', 'Joel McHale', 'Kevin James', 'Matthew Perry', 'Sarah Paulson', 'Kurt Russell', 'Mandy Moore', 'Mark Wahlberg', 'Ben Affleck', 'Johnny Depp', 
    ## Kurt Russell has a huge variability in age, facial age, and style. 
    ## Mandy Moore also has decent variablity in hair length and image styles. 
    ## Mark Wahlberg has some variability in facial hair styles. 
    ## Ben Affleck has decent variability in facial hair styles and hair styles. 
    ## Johnny Depp has huge variability in facial hair styles and hair styles, poses, and age. 
    male_white_celebs = ['Gabriel LaBelle', 'Dominic Sessa', 'Corey Mylchreest', 'Sam Nivola', 'Tom Blyth', 'Jordan Firstman', 'Josh Seiter', 'Nicola Porcella', 'Armen Nahapetian', 'Joey Klaasen']
    male_black_celebs = ['Jaylin Webb', 'Quincy Isaiah', 'Miles Gutierrez-Riley', 'Jalyn Hall', 'Myles Frost', 'Wisdom Kaye', 'Olly Sholotan', 'Isaiah R. Hill', "Bobb'e J. Thompson", 'Myles Truitt']
    male_brown_celebs = ['Sajith Rajapaksa', 'Aryan Simhadri', 'Aditya Kusupati', 'Vihaan Samat', 'Ishwak Singh', 'Gurfateh Pirzada', 'Pavail Gulati', 'Cwaayal Singh', 'Jibraan Khan', 'Vedang Raina']
    
    female_white_celebs = ['Mia Challiner', 'Isabel Gravitt', 'Pardis Saremi', 'Elle Graham', 'Cara Jade Myers', 'Ali Skovbye', 'Gabby Windey', 'Hannah Margaret Selleck', 'Bridgette Doremus', 'Milly Alcock']
    female_black_celebs = ['Kudakwashe Rutendo', 'Ayo Edebiri', 'Kaci Walfall', 'Elisha Williams', 'Laura Kariuki', 'Akira Akbar', 'Savannah Lee Smith', 'Samara Joy', 'Arsema Thomas', 'Leah Jeffries'] #'Ariana Neal' 'Grace Duah'
    female_brown_celebs = ['Priya Kansara', 'Pashmina Roshan', 'Banita Sindhu', 'Alaia Furniturewala', 'Paloma Dhillon', 'Alizeh Agnihotri', 'Geetika Vidya Ohlyan', 'Saloni Batra', 'Sharvari Wagh', 'Arjumman Mughal']
    
    concerning_celebrity_list = male_white_celebs + male_brown_celebs + male_black_celebs + female_white_celebs + female_brown_celebs + female_black_celebs
    args.top_k_training_images = 11
    ## we want to view the top-k images for each of the celebs in ths concerning_celebrity_list 
    # for image in df_single_person_training_images.loc[celeb_here]['image'].values[:args.top_k_training_images]:
    # common_image_extensions = ['jpg', 'jpeg', 'png', 'heif', 'heic', 'webp', 'tiff', 'tif', 'bmp', 'gif']
    os.makedirs(f"good_images_each_celeb", exist_ok=True)
    for celeb_here in concerning_celebrity_list:
        # if df_num_matches[df_num_matches['celeb']==celeb_here]['count_high_quality_images'].values[0] > 25:
        #     continue        ## we only want to view images of celebs that have problems
        print(celeb_here)
        good_image_count = 0
        ## copy these images to "good_images_each_celeb" and open them there
        os.makedirs(f"good_images_each_celeb/{celeb_here.replace(' ', '-')}", exist_ok=True)
        ## do this only when celeb_here is in df_single_person_training_images
        if not celeb_here in df_single_person_training_images.index:
            print(f"{celeb_here} not found in the df_single_person_training_images dataframe.")
            continue
        for image in df_single_person_training_images.loc[celeb_here]['image'].values[:args.top_k_training_images]:
            if args.training:
                full_name_image = image
            elif args.images_I_downloaded_from_internet:
                full_name_image = f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{celeb_here.replace(' ', '-')}/{image}"
            with Image.open(full_name_image) as img:
                extension = img.format.lower()
            subprocess.run(['cp', full_name_image, f"good_images_each_celeb/{celeb_here.replace(' ', '-')}/{image}"])
            # subprocess.run(['code', '-r', f"good_images_each_celeb/{celeb_here.replace(' ', '-')}/{image}"])
            good_image_count += 1
        ## and now wait for the user to type in "next"
        # user_input = input("Type 'next' to continue to the next celeb: ")
        # while user_input != 'next':
        #     user_input = input("Type 'next' to continue to the next celeb: ")
        
        ## do not delete the images, as we want to keep them for future reference.


def save_face_cutout(args, face, image_here, celeb, image, printed_name):
    import cv2
    x1, y1, x2, y2 = face.bbox
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    ## convert the float to int, by rounding it to the nearest integer
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > image_here.shape[1]: x2 = image_here.shape[1]
    if y2 > image_here.shape[0]: y2 = image_here.shape[0]
    cutout = image_here[y1:y2, x1:x2]
    cutout_area = (y2-y1) * (x2-x1)
    extraced_face_cuts_directory = f"extracted_face_cut_outs" if args.set_of_people == "celebrity" else f"extracted_face_cut_outs_politicians"
    os.makedirs(extraced_face_cuts_directory, exist_ok=True)
    if args.all_laion_images:
        base_directory = f"{extraced_face_cuts_directory}/all_laion_images/{celeb}"
    elif args.all_laion_alias_name_images:
        base_directory = f"{extraced_face_cuts_directory}/all_laion_alias_name_images/{celeb}"
    elif args.images_I_downloaded_from_internet:
        base_directory = f"{extraced_face_cuts_directory}/images_I_downloaded_from_internet/{celeb}"
    elif args.all_images_internet:
        base_directory = f"{extraced_face_cuts_directory}/all_images_internet/{celeb}"
    else:
        raise NotImplementedError
    save_directory = f"{base_directory}/{image}"
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(save_directory, exist_ok=True)
    ## get the extension of the image
    extension = printed_name.split('.')[-1]
    save_name = f"{save_directory}/cutout_{y1}_{y2}_{x1}_{x2}_face_area_{cutout_area}.{extension}"
    try:
        cv2.imwrite(save_name, cutout)
    except:
        print(f"celeb: {celeb}, image: {image}, face: {face.bbox}, cutout: {cutout.shape}")


def face_recognition_to_compare_training_and_generated_images(args, celebrity_here=None, only_generate_embeddings=False, all_people_list=None, num_training_images_to_compare=10, insightfaceapp=None, insightface_recognition=None, use_selected_images_for_matching=False, use_filtered_laion_images_and_selected_images_for_matching=False):
    base_embeddings_directory = f"{args.face_recognition_model}_embeddings_close_up_{args.stable_diffusion_model}" if args.set_of_people == "celebrity" else f"{args.face_recognition_model}_embeddings_close_up_politicians_{args.stable_diffusion_model}" if args.set_of_people == "politicians" else f"{args.face_recognition_model}_embeddings_close_up_caption_assumption_{args.stable_diffusion_model}" if args.set_of_people == "caption_assumption" else None
    extraced_face_cuts_directory = f"extracted_face_cut_outs" if args.set_of_people == "celebrity" else f"extracted_face_cut_outs_politicians" if args.set_of_people == "politicians" else None
    
    if args.generate_embeddings_face_recognition or only_generate_embeddings:       ## we have generated embeddings for the 200 generated images for each celeb. 
        total_images = {}
        consider_only_single_person_images = True       ## for all the internet images, generated images, and previously even LAION images we only considered the single person images. now for all the LAION images we are considering all the faces. 
        if celebrity_here is None:
            celebrity_list_here = celebrity_list + celebrity_list0
        else:
            celebrity_list_here = [celebrity_here]
            if args.generated:
                args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
                if os.path.exists(f"{base_embeddings_directory}/{args.image_generation_prompt}/{celebrity_here}.json"):
                    print(f"celeb: {celebrity_here} already processed.")
                    return
            elif args.training or args.images_I_downloaded_from_internet or args.all_images_internet or args.all_laion_images or args.all_laion_alias_name_images:
                datatype_here = 'training' if args.training else 'images_I_downloaded_from_internet' if args.images_I_downloaded_from_internet else 'all_images_internet' if args.all_images_internet else 'all_laion_images' if args.all_laion_images else 'all_laion_alias_name_images'
                if os.path.exists(f"{base_embeddings_directory}/{datatype_here}/{celebrity_here}.json"):
                    print(f"celeb: {celebrity_here} already processed.")
                    return

        if args.training:       ## this was the first set of laion images I had downloaded, not used anymore.
            single_person_training_images_df = pd.read_csv(f"close_up_single_person_training_images_{args.set_of_people}.csv", sep='|', header=0)       # celeb|image|effective_pixels|sharpness
        elif args.images_I_downloaded_from_internet:
            if args.set_of_people == "celebrity":
                single_person_training_images_df = pd.read_csv(f"close_up_single_person_images_I_downloaded_from_internet_{args.set_of_people}.csv", sep='|', header=0)       # celeb|image|effective_pixels|sharpness
            consider_only_single_person_images = True
        elif args.all_images_internet:
            if args.set_of_people == "celebrity":
                single_person_training_images_df = pd.read_csv(f"close_up_single_person_all_images_internet_{args.set_of_people}.csv", sep='|', header=0)
            consider_only_single_person_images = True
        elif args.all_laion_images or args.all_laion_alias_name_images:
            consider_only_single_person_images = False      ## here we will get embeddings for all the faces, single and multiple. 
        elif args.generated:
            consider_only_single_person_images = True
        else:   raise NotImplementedError
        
        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
            from deepface import DeepFace
            loaded_model = None
        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
            import cv2
        else:
            raise NotImplementedError
        
        if args.parallelize_across_one_entity:
            assert insightfaceapp is None and insightface_recognition is None, "insightfaceapp and insightface_recognition should be None when args.parallelize_across_one_entity is True."
        
        for celeb_idx, celeb in enumerate(celebrity_list_here):
            if args.training:
                images_directory = f'/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb}'      ## we have created a soft link to the downloaded_celebrity_images directory. But we should ot encode all the images, that is very expensive. Let's only encode the single person images of a celeb. 
                df_this_celeb = single_person_training_images_df[single_person_training_images_df['celeb'] == celeb]
                if len(df_this_celeb) == 0:
                    print(f"{celeb} not found in the single_person_training_images_df dataframe.")
                    continue
                print("processing celeb: ", celeb)
                list_of_images = df_this_celeb['image'].values
            elif args.images_I_downloaded_from_internet:        ## these are the reference images collected using SerpAPI
                images_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet{'_politicians' if args.set_of_people == 'politicians' else ''}/{celeb.replace(' ', '-')}"
                if args.set_of_people == "politicians":
                    assert os.path.isdir(images_directory), f"{images_directory} not found."
                
                if not os.path.isdir(images_directory):
                    print(f"{images_directory} not found.")
                    continue
                
                if args.set_of_people == "celebrity":       ## for celebrities I did the initial step of face detection with 1 face, but that step is now removed. 
                    df_this_celeb = single_person_training_images_df[single_person_training_images_df['celeb'] == celeb]
                    if len(df_this_celeb) == 0:
                        print(f"{celeb} not found in the single_person_training_images_df dataframe.")
                        continue
                    print("processing celeb: ", celeb)
                    list_of_images = df_this_celeb['image'].values
                elif args.set_of_people == "politicians":
                    list_of_images = os.listdir(images_directory)
                    assert len(list_of_images) > 30, f"{celeb} has less than 80 images: {len(list_of_images)}"      ## I downloaded 100 using serpapi, so it should not be less than 80. 
                else:
                    raise NotImplementedError
                
            elif args.generated:
                args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
                images_directory = f'../generated_images_{args.set_of_people}_{args.stable_diffusion_model}/{celeb}/{args.image_generation_prompt}'
                list_of_images = os.listdir(images_directory)
            elif args.all_images_internet:
                images_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}"
                if not os.path.isdir(images_directory):
                    print(f"{images_directory} not found.")
                    continue
                df_this_celeb = single_person_training_images_df[single_person_training_images_df['celeb'] == celeb]
                if len(df_this_celeb) == 0:
                    print(f"{celeb} not found in the single_person_training_images_df dataframe.")
                    continue
                print("processing celeb: ", celeb)
                list_of_images = df_this_celeb['image'].values
            elif args.all_laion_images or args.all_laion_alias_name_images:
                if args.all_laion_images:
                    if args.set_of_people == "celebrity":
                        # images_directory = f'/gscratch/h2lab/vsahil/vlm-efficiency/all_downloaded_images/{celeb}'
                        images_directory = f'/gscratch/scrubbed/vsahil/all_downloaded_images/{celeb}'
                    elif args.set_of_people == "politicians":
                        images_directory = f'/gscratch/scrubbed/vsahil/all_downloaded_images_politicians/{celeb}'
                    elif args.set_of_people == "caption_assumption":
                        images_directory = f'/gscratch/scrubbed/vsahil/all_downloaded_images_for_captions/{celeb}'
                    else: raise NotImplementedError
                elif args.all_laion_alias_name_images:
                    if args.set_of_people == "celebrity":
                        images_directory = f'/gscratch/scrubbed/vsahil/all_downloaded_images/{celeb}'
                if not os.path.isdir(images_directory):
                    print(f"{images_directory} not found.")
                    raise Exception(f"{images_directory} not found.")
                    # continue
                ## take all files that do not end with .csv
                list_of_images = [image for image in os.listdir(images_directory) if ".csv" not in image[-4:]]
                # assert len(list_of_images) > 0, f"celeb: {celeb}, images_directory: {images_directory}"       -- there are 0 images celebs. 
                assert len(list_of_images) == len(os.listdir(images_directory)) - 1, f"celeb: {celeb}, images_directory: {images_directory}, len(list_of_images): {len(list_of_images)}, len(os.listdir(images_directory)): {len(os.listdir(images_directory))}"
            else:
                raise Exception('Please specify either --training or --generated')
            
            if not args.parallelize_across_one_entity:
                total_images[celeb] = 0
                image_embeddings = {}
                
                for image_id, image in enumerate(list_of_images):
                    if "grid_images" in image:      continue        ## we don't want to include the grid images
                    if ".csv" in image[-4:]:      continue        ## we don't want to include the csv files
                    if "open_clip_image_features" in image:      continue        ## we don't want to include the open_clip_image_features files
                    if ".json" in image[-5:]:      continue        ## we don't want to include the json files
                    total_images[celeb] += 1
                    printed_name = f'{images_directory}/{image}'
                    
                    if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                        detected_face_embeddings, loaded_model = DeepFace.represent(printed_name, model_name=args.face_recognition_model, enforce_detection=False, align=False, model=loaded_model, detector_backend='retinaface')        ## changing the default detector for face detection
                        if args.training:     ## and only images with 1 face are being passed for training images, hence this should hold always true for training images
                            assert len(detected_face_embeddings) == 1, f"celeb: {celeb}, image: {image}, detected_face_embeddings: {detected_face_embeddings}"
                        if len(detected_face_embeddings) == 1:
                            image_embeddings[image] = detected_face_embeddings      ## this way we can totally avoid the step of getting the number of faces in the generated images as a separate stage in the pipeline.    
                    
                    elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                        image_here = cv2.imread(printed_name)
                        insightfacefaces = insightfaceapp.get(image_here)
                        
                        if len(insightfacefaces) == 1 and consider_only_single_person_images:  ## we only want the training and generated and internet downloaded images with 1 face. Therefore we are able to skip generating the single person file for the internet downloaded images. 
                            if args.face_recognition_model == 'insightface_buffalo':
                                insightfaceembedding = insightfacefaces[0].normed_embedding
                            elif args.face_recognition_model == 'insightface_resnet100':
                                insightfaceembedding = insightface_recognition.get(image_here, insightfacefaces[0])
                            # image_embeddings[image] = insightfaceembedding.astype(float).tolist()     
                            image_embeddings[f"{image}_{insightfacefaces[0].bbox}"] = insightfaceembedding.astype(float).tolist()       ## save the dimensions of the face cut out that we can use later to compute the face area.
                            ## save the face cutout of the image
                            # if args.all_images_internet or args.images_I_downloaded_from_internet:
                            #     # print(f"Saving face cutout for celeb: {celeb}, image: {image}")
                            #     save_face_cutout(args, insightfacefaces[0], image_here, celeb, image, printed_name)
                        
                        elif len(insightfacefaces) != 1 and consider_only_single_person_images:
                            pass
                        
                        elif not consider_only_single_person_images:
                            assert args.all_laion_images or args.all_laion_alias_name_images       ## in this case we want to save the embeddings for all the faces. 
                            for face in insightfacefaces:       ## the ordering of the face here is determined by the face detection model, I do not know what it will be. 
                                if args.face_recognition_model == 'insightface_buffalo':
                                    insightfaceembedding = face.normed_embedding
                                elif args.face_recognition_model == 'insightface_resnet100':
                                    insightfaceembedding = insightface_recognition.get(image_here, face)
                                image_embeddings[f"{image}_{face.bbox}"] = insightfaceembedding.astype(float).tolist()
                                ## save the cutout of the face as cutout_{image}_{face.bbox}.jpg
                                # save_face_cutout(args, face, image_here, celeb, image, printed_name)

                    if image_id % 100 == 0 and image_id > 0:
                        print(f"Done with {image_id} images for celeb: {celeb}")
            
            else:
                ## here we will parallelize the image embedding generation across GPUs. 
                def worker_gpu_face_embedding(gpu_id, images_subset, return_dict):
                    total_images_this_gpu = 0
                    image_embeddings_this_gpu = {}
                
                    if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                        from deepface import DeepFace
                        loaded_model = None
                    elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                        import insightface
                        from insightface.app import FaceAnalysis
                        insightfaceapp = FaceAnalysis(providers=[('CUDAExecutionProvider', {'device_id': gpu_id})])
                        insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                        if args.face_recognition_model == 'insightface_resnet100':
                            insightface_recognition = insightface.model_zoo.get_model(args.insightface_model_path)     ## here we are going to use a different model for face recognition.
                            insightface_recognition.prepare(ctx_id=0)
                        
                    for image_id, image in enumerate(images_subset):
                        if "grid_images" in image:      continue        ## we don't want to include the grid images
                        if ".csv" in image[-4:]:      continue        ## we don't want to include the csv files
                        if "open_clip_image_features" in image:      continue        ## we don't want to include the open_clip_image_features files
                        if ".json" in image[-5:]:      continue        ## we don't want to include the json files
                        total_images_this_gpu += 1
                        printed_name = f'{images_directory}/{image}'
                        
                        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                            detected_face_embeddings, loaded_model = DeepFace.represent(printed_name, model_name=args.face_recognition_model, enforce_detection=False, align=False, model=loaded_model, detector_backend='retinaface')        ## changing the default detector for face detection
                            if args.training:     ## and only images with 1 face are being passed for training images, hence this should hold always true for training images
                                assert len(detected_face_embeddings) == 1, f"celeb: {celeb}, image: {image}, detected_face_embeddings: {detected_face_embeddings}"
                            if len(detected_face_embeddings) == 1:
                                image_embeddings_this_gpu[image] = detected_face_embeddings      ## this way we can totally avoid the step of getting the number of faces in the generated images as a separate stage in the pipeline.    
                        
                        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                            image_here = cv2.imread(printed_name)
                            insightfacefaces = insightfaceapp.get(image_here)
                            
                            if len(insightfacefaces) == 1 and consider_only_single_person_images:  ## we only want the training and generated and internet downloaded images with 1 face. Therefore we are able to skip generating the single person file for the internet downloaded images. 
                                if args.face_recognition_model == 'insightface_buffalo':
                                    insightfaceembedding = insightfacefaces[0].normed_embedding
                                elif args.face_recognition_model == 'insightface_resnet100':
                                    insightfaceembedding = insightface_recognition.get(image_here, insightfacefaces[0])
                                image_embeddings_this_gpu[image] = insightfaceembedding.astype(float).tolist()
                            
                            elif len(insightfacefaces) != 1 and consider_only_single_person_images:
                                pass
                            
                            elif not consider_only_single_person_images:
                                assert args.all_laion_images or args.all_laion_alias_name_images      ## in this case we want to save the embeddings for all the faces. 
                                for face in insightfacefaces:       ## the ordering of the face here is determined by the face detection model, I do not know what it will be. 
                                    if args.face_recognition_model == 'insightface_buffalo':
                                        insightfaceembedding = face.normed_embedding
                                    elif args.face_recognition_model == 'insightface_resnet100':
                                        insightfaceembedding = insightface_recognition.get(image_here, face)
                                    image_embeddings_this_gpu[f"{image}_{face.bbox}"] = insightfaceembedding.astype(float).tolist()
                                    continue        ## we do not want to save the cutouts of the faces.
                                    ## save the cutout of the face as cutout_{image}_{face.bbox}.jpg
                                    x1, y1, x2, y2 = face.bbox
                                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                                    ## convert the float to int, by rounding it to the nearest integer
                                    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                                    if x1 < 0: x1 = 0
                                    if y1 < 0: y1 = 0
                                    if x2 > image_here.shape[1]: x2 = image_here.shape[1]
                                    if y2 > image_here.shape[0]: y2 = image_here.shape[0]
                                    cutout = image_here[y1:y2, x1:x2]
                                    if args.all_laion_images:
                                        base_directory = f"{extraced_face_cuts_directory}/all_laion_images/{celeb}"
                                    elif args.all_laion_alias_name_images:
                                        base_directory = f"{extraced_face_cuts_directory}/all_laion_alias_name_images/{celeb}"
                                    save_directory = f"{base_directory}/{image}"
                                    os.makedirs(base_directory, exist_ok=True)
                                    os.makedirs(save_directory, exist_ok=True)
                                    ## get the extension of the image
                                    extension = printed_name.split('.')[-1]
                                    save_name = f"{save_directory}/cutout_{y1}_{y2}_{x1}_{x2}.{extension}"
                                    try:
                                        cv2.imwrite(save_name, cutout)
                                    except:
                                        print(f"celeb: {celeb}, image: {image}, face: {face.bbox}, cutout: {cutout.shape}")
                            
                        if image_id % 100 == 0 and image_id > 0:
                            print(f"Done with {image_id} images for celeb: {celeb} on GPU: {gpu_id}")
                        
                    return_dict[gpu_id] = [total_images_this_gpu, image_embeddings_this_gpu]
                
                
                ## here will will parallelize the face detection process across the GPUs for the images of this celeb. One process per GPU.
                def split_data(data, num_splits):
                    """Splits data into sublists."""
                    k, m = divmod(len(data), num_splits)
                    return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits))
                
                num_gpus = get_slurm_gpus()
                assert num_gpus > 0, 'No GPUs allocated'
                data_to_process = list(list_of_images)
                
                if len(data_to_process) < 10 * num_gpus:
                    num_gpus = 1        ## no need to parallelize with such low number of images. 
                
                if len(data_to_process) > 0:
                    images_subset_total = list(split_data(data_to_process, num_gpus))
                    with Manager() as manager:
                        return_dict = manager.dict()
                        processes = []
                        print(f"processing the images for {celeb} with {num_gpus} GPUs. Total images: {len(data_to_process)}")
                        for gpu_id, images_subset in enumerate(images_subset_total):
                            p = Process(target=worker_gpu_face_embedding, args=(gpu_id, images_subset, return_dict))
                            processes.append(p)
                            p.start()

                        for p in processes:
                            p.join()
                        
                        total_images[celeb] = 0
                        image_embeddings = {}
                        for gpu_id in range(num_gpus):
                            if gpu_id in return_dict:
                                total_images[celeb] += return_dict[gpu_id][0]
                                image_embeddings.update(return_dict[gpu_id][1])
                               
                else:
                    total_images[celeb] = 0
                    image_embeddings = {}
            
            os.makedirs(f"{base_embeddings_directory}", exist_ok=True)
            if args.generated:
                os.makedirs(f"{base_embeddings_directory}/{args.image_generation_prompt}", exist_ok=True)    ## save the embeddings for each celeb in the directory: /gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/facenet512_embeddings
                with open(f"{base_embeddings_directory}/{args.image_generation_prompt}/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            elif args.images_I_downloaded_from_internet:
                os.makedirs(f"{base_embeddings_directory}/images_I_downloaded_from_internet", exist_ok=True)
                with open(f"{base_embeddings_directory}/images_I_downloaded_from_internet/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            elif args.all_images_internet:
                os.makedirs(f"{base_embeddings_directory}/all_images_internet", exist_ok=True)
                with open(f"{base_embeddings_directory}/all_images_internet/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            elif args.training:
                os.makedirs(f"{base_embeddings_directory}/training", exist_ok=True)    ## save the embeddings for each celeb in the directory: /gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/facenet512_embeddings
                with open(f"{base_embeddings_directory}/training/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            elif args.all_laion_images:
                os.makedirs(f"{base_embeddings_directory}/all_laion_images", exist_ok=True)
                with open(f"{base_embeddings_directory}/all_laion_images/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            elif args.all_laion_alias_name_images:
                os.makedirs(f"{base_embeddings_directory}/all_laion_alias_name_images", exist_ok=True)
                with open(f"{base_embeddings_directory}/all_laion_alias_name_images/{celeb}.json", "w") as f:
                    json.dump(image_embeddings, f)
            
            print(f"Done with celeb: {celeb}, total images: ", total_images[celeb])

    elif args.store_similarity_matrix:
        ## This function will store similarity matrix between each pair of images (it might be training or generated or internet images.) This is mainly to be used for submodular min. Store the matrix as a numpy array.
        if args.set_of_people == "celebrity":
            celebrity_list_here = celebrity_list + celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities
        elif args.set_of_people == "politicians":
            celebrity_list_here = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/politicians_to_analyze.csv")['Name'].tolist()
            celebrity_list_here = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/more_politicians_3.csv")['Name'].tolist()
        else:
            raise NotImplementedError
        
        print(len(celebrity_list_here))
        
        if args.all_images_internet:
            base_save_directory = f"{base_embeddings_directory}/all_images_internet_similarity_matrix"
            os.makedirs(base_save_directory, exist_ok=True)
            embeddings_directory = f"{base_embeddings_directory}/all_images_internet"
            close_up_single_person_file = f"close_up_single_person_all_images_internet_{args.set_of_people}.csv"
        elif args.images_I_downloaded_from_internet:
            base_save_directory = f"{base_embeddings_directory}/images_I_downloaded_from_internet_similarity_matrix"
            os.makedirs(base_save_directory, exist_ok=True)
            embeddings_directory = f"{base_embeddings_directory}/images_I_downloaded_from_internet"
            if args.set_of_people == "celebrity":
                close_up_single_person_file = f"close_up_single_person_images_I_downloaded_from_internet_{args.set_of_people}.csv"
        else:
            raise NotImplementedError
        
        for celeb_here in celebrity_list_here:
            ## load the embeddings of the celeb
            with open(f"{embeddings_directory}/{celeb_here}.json", "r") as f:
                this_celeb_images_embeddings = json.load(f)

            ## now compute the similarity between each pair of images by computing the cosine similarity between the embeddings of the images. First we will normalize the embeddings and then compute the cosine similarity.
            image_representations = []
            sequenced_images = []
            for image in this_celeb_images_embeddings:
                image_representations.append(this_celeb_images_embeddings[image])
                sequenced_images.append(image)
            image_representations = np.array(image_representations)
            assert image_representations.shape[0] == len(this_celeb_images_embeddings)
            image_representations = image_representations / np.linalg.norm(image_representations, axis=1, keepdims=True)        ## normalize the embeddings
            cosine_similarity = np.dot(image_representations, image_representations.T)  ## compute the cosine distance of each image with every other image.
            assert cosine_similarity.shape == (len(this_celeb_images_embeddings), len(this_celeb_images_embeddings))
            ## assert that max value is 1 and min value is -1
            assert np.max(cosine_similarity) <= 1 + 1e-6 and np.min(cosine_similarity) >= -1 - 1e-6
            ## clip between -1 and 1
            cosine_similarity = np.clip(cosine_similarity, -1, 1)   # Ensure the similarity is within proper bounds due to floating point errors.
            
            if args.set_of_people == "celebrity":
                ## now also get the size of each image from the close_up_single_person file. Header: celeb|image|effective_pixels|sharpness
                df_effective_pixels_and_shape_this_celeb = pd.read_csv(close_up_single_person_file, sep='|', header=0)
                df_effective_pixels_and_shape_this_celeb = df_effective_pixels_and_shape_this_celeb[df_effective_pixels_and_shape_this_celeb['celeb']==celeb_here]
                assert len(df_effective_pixels_and_shape_this_celeb) == len(sequenced_images)
                ## for each image in sequenced_images, get their effective pixels from the file. This is used by submarine to favor larger faces in the selection process. 
                image_face_pixels = []
                for image in sequenced_images:
                    image_face_pixels.append(df_effective_pixels_and_shape_this_celeb[df_effective_pixels_and_shape_this_celeb['image']==image]['effective_pixels'].values[0])
                assert len(image_face_pixels) == len(sequenced_images)
                image_face_pixels = np.array(image_face_pixels)
            
            elif args.set_of_people == "politicians":
                ## for politicians we didn't do face detection step, but we saved the face cut out size in the name of the images itself. 
                image_face_pixels = []
                for image in sequenced_images:
                    ## each image looks like: image_seq_1.jpg_[ 73.09487  66.96366 125.60453 135.91803]. Get the 4 coordinates
                    x1, y1, x2, y2 = image.split('[')[1].split(']')[0].strip().split()
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    face_area = (y2-y1) * (x2-x1)
                    image_face_pixels.append(face_area)
                assert len(image_face_pixels) == len(sequenced_images)
                image_face_pixels = np.array(image_face_pixels)
        
            else:
                raise NotImplementedError
            
            ## now store both the cosine similarity and the image_face_pixels in two separate numpy array. 
            save_directory = f"{base_save_directory}/{celeb_here}"
            os.makedirs(save_directory, exist_ok=True)
            ## also save the image representations
            np.save(f"{save_directory}/image_representations.npy", image_representations)
            np.save(f"{save_directory}/cosine_similarity.npy", cosine_similarity)
            np.save(f"{save_directory}/image_face_pixels.npy", image_face_pixels)
            print(f"saved similarity matrix and image face pixels for celeb: {celeb_here}")

    elif args.match_embeddings_face_recognition:
        directory_string = ''       ## we leave this empty so that other folks can also run experiments from the main directory.
        if args.stable_diffusion_model != "1":
            directory_string = '/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/'
        if args.stable_diffusion_model == "v2":
            if args.set_of_people == "celebrity":
                df_laion5b = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrities_sorted_laion5b.csv", sep=',', header=0)     ## Name,counts_in_laion2b-en,first_name,first_name_counts_in_laion2b-en,counts_in_laion5b
            elif args.set_of_people == "politicians":
                df_laion5b = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_images_politicians_sorted_laion5b.csv", sep=',', header=0)     ## Name,counts_in_laion5b
            else: raise NotImplementedError
                
        assert all_people_list is not None, "Please provide the list of all people."
        ## select the images that are one face from the generated images for the selected celebs. Generate the embeddings for the selected images for the selected celebs and get the avg cosine similarity between the embeddings of the selected trainign images and the one person generated images.
        if args.set_of_people == "celebrity":
            df_single_person_training_images = pd.read_csv(f"{directory_string}all_celebs_single_person_training_data_face_recognition_result_{args.set_of_people}.csv", sep='|', header=0)       ## this is the output from the training images selection after the face recognition model
            df_single_person_images_I_downloaded_from_internet = pd.read_csv(f"{directory_string}all_celebs_single_person_images_I_downloaded_from_internet_face_recognition_result_{args.set_of_people}.csv", sep='|', header=0)       ## this is the output from the internet downloaded images selection after the face recognition model
            df_total_captions = pd.read_csv(f"{directory_string}../celebrity_data/celebrities_sorted.csv", sep=',', header=0)      # Name,counts_in_laion2b-en
        elif args.set_of_people == "politicians":
            df_total_captions = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")    ## Name,counts_in_laion2b-en
        # elif args.set_of_people == "caption_assumption":
        #     downloaded_ent = ["and", "the", "a", "an", "in", "for", "on", "at", "by", "to", "of", "it", "as"]
        #     ## convert the list to a pandas dataframe with the column name 'Name' and its counts in laion2b-en as 0 for all the entities.
        #     df_total_captions = pd.DataFrame(downloaded_ent, columns=['Name'])
        #     df_total_captions['counts_in_laion2b-en'] = 0
        else:
            raise NotImplementedError
            

        ## we also want to get the number of training data that we have downloaded for each celeb.
        num_downloaded_training_images = {}
        args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
        if use_filtered_laion_images_and_selected_images_for_matching or use_selected_images_for_matching:
            if args.same_face_threshold == "lower":
                args.threshold_for_max_f1_score = 0.45668589190104747
            elif args.same_face_threshold == "higher":
                args.threshold_for_max_f1_score = 0.5576227593909431
            else: 
                raise NotImplementedError
            
            df_find_if_laion_images_exist = pd.read_csv(f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people == 'celebrity' else 'insightface_resnet100_embeddings_close_up_politicians_1'}/num_face_counts_all_laion_images_{args.set_of_people}_threshold_{args.threshold_for_max_f1_score}.csv", sep='|', header=0)       # celeb|num_face_counts|num_downloaded_images|matching_captions|effective_num_face_counts|face_area|effective_face_area
            df_find_if_laion_images_exist = df_find_if_laion_images_exist[['celeb', 'num_face_counts', 'effective_num_face_counts', 'effective_face_area']]
            if args.consider_name_aliases:
                assert args.set_of_people == "celebrity", "Name aliases are only available for celebrities."
                df_alias_name_laion_images = pd.read_csv(f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people == 'celebrity' else 'insightface_resnet100_embeddings_close_up_politicians_1'}/num_face_counts_all_laion_alias_name_images_{args.set_of_people}_threshold_{args.threshold_for_max_f1_score}.csv", sep='|', header=0)      # celeb|alias_name|num_face_counts|num_downloaded_images|matching_captions|effective_num_face_counts|face_area|effective_face_area
                df_alias_name_laion_images = df_alias_name_laion_images[['celeb', 'alias_name', 'num_face_counts', 'effective_num_face_counts', 'effective_face_area']]

        did_not_write_header = False
        for seq_celeb, celeb_here in enumerate(all_people_list[::-1]):
            laion_filtered_images_indices, laion_filtered_images_embeddings, alias_name_laion_filtered_images_indices, alias_name_laion_filtered_images_embeddings = None, None, None, None
            selected_images_from_internet = None

            if args.set_of_people == "celebrity":
                num_downloaded_training_images[celeb_here] = len(os.listdir(f"{directory_string}../all_downloaded_images/{celeb_here}"))
            elif args.set_of_people == "politicians":
                num_downloaded_training_images[celeb_here] = len(os.listdir(f"/gscratch/scrubbed/vsahil/all_downloaded_images_politicians/{celeb_here}"))
            else:   raise NotImplementedError
            
            ## GENERATED IMAGE EMBEDDINGS SECTION -- DO NOT ADD THE DIRECTORY NAME TO THE IMAGES
            if args.set_of_people == "politicians":
                celeb_here_normalized = unicodedata.normalize('NFD', celeb_here)        ## several politician names have accents, we want to normalize them to the base form. And this is a problem because they were not generated on the same machine.
            else:
                celeb_here_normalized = celeb_here

            with open(f"{base_embeddings_directory}/{args.image_generation_prompt}/{celeb_here_normalized}.json", "r") as f:
                this_celeb_generated_images_embeddings = json.load(f)      ## now all of these images have only one face because I am only generating embeddings for (generated) images that have one face.

            generated_image_with_one_face_embeddings = []
            for image in this_celeb_generated_images_embeddings:
                if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                    assert len(this_celeb_generated_images_embeddings[image]) == 1, f'image {image} has {len(this_celeb_generated_images_embeddings[image])}'     ## Since we are operating with 1 faces, we should have only 1 embedding per image.
                    generated_image_with_one_face_embeddings.append(this_celeb_generated_images_embeddings[image][0]['embedding'])
                elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                    generated_image_with_one_face_embeddings.append(this_celeb_generated_images_embeddings[image])
                else:
                    raise Exception('Please specify a valid face recognition model.')
            generated_image_with_one_face_embeddings = np.array(generated_image_with_one_face_embeddings)

            if use_selected_images_for_matching:
                ## get the best images for everyone. 
                training_image_embeddings, selected_images_from_internet = compute_similarity_selected_images(args, call_from_face_recoginition=True, call_from_face_recognition_celeb=celeb_here)
            
            elif use_filtered_laion_images_and_selected_images_for_matching:
                using_entitys_filtered_laion_images = False
                using_entitys_filtered_alias_names_laion_images = False
                number_of_filtered_laion_images = df_find_if_laion_images_exist[df_find_if_laion_images_exist['celeb']==celeb_here]['num_face_counts'].values[0]
                ## for the alias we will have multiple entries per celeb, and we will sum the value for num_face_counts for all the alias names of the celeb.
                if args.consider_name_aliases:
                    assert args.set_of_people == "celebrity", "Name aliases are only available for celebrities."
                    number_of_filtered_alias_name_laion_images = df_alias_name_laion_images[df_alias_name_laion_images['celeb']==celeb_here]['num_face_counts'].sum()
                else:
                    number_of_filtered_alias_name_laion_images = 0
                
                if number_of_filtered_laion_images + number_of_filtered_alias_name_laion_images >= 10:
                    ## LAION IMAGE EMBEDDINGS SECTION - here for the people that have laion images that passed the filter for similarity to selected images, we will use the filtered laion images for comparison. 
                    ## For individual whose laion images did not pass the filter, we will use the selected images for comparison. If there are faces that pass the filter, that is in the columns: effective_num_face_counts in the dataframe: df_find_if_laion_images_exist. Not that all celebs will be in the dataframe what matters if effective_num_face_counts is atleast 10 or not. 
                    training_image_embeddings = []
                    if number_of_filtered_laion_images > 0:
                        laion_filtered_images_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people == 'celebrity' else 'insightface_resnet100_embeddings_close_up_politicians_1'}/all_laion_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
                        assert os.path.exists(f"{laion_filtered_images_directory}/{celeb_here}.npy"), f"{celeb_here} not found in the laion_filtered_images_directory."
                        assert os.path.exists(f"{laion_filtered_images_directory}/{celeb_here}_laion_filtered_images_indices.npy"), f"{celeb_here} not found in the laion_filtered_images_directory."
                        laion_filtered_images_embeddings = np.load(f"{laion_filtered_images_directory}/{celeb_here}.npy")
                        laion_filtered_images_indices = np.load(f"{laion_filtered_images_directory}/{celeb_here}_laion_filtered_images_indices.npy")
                        assert laion_filtered_images_embeddings.shape[0] == number_of_filtered_laion_images == laion_filtered_images_indices.shape[0], f"{celeb_here} has {laion_filtered_images_embeddings.shape[0]} laion images, and df_find_if_laion_images_exist has {number_of_filtered_laion_images} images."
                        training_image_embeddings.append(laion_filtered_images_embeddings)
                        using_entitys_filtered_laion_images = True
                    
                    if number_of_filtered_alias_name_laion_images > 0:
                        assert args.args.consider_name_aliases
                        alias_name_laion_filtered_images_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{base_embeddings_directory}/all_laion_alias_name_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
                        alias_names_of_celeb_here = df_alias_name_laion_images[df_alias_name_laion_images['celeb']==celeb_here]['alias_name'].values
                        assert len(alias_names_of_celeb_here) > 0, f"{celeb_here} has no alias names."
                        alias_name_laion_filtered_images_embeddings = []
                        alias_name_laion_filtered_images_indices = []
                        
                        for alias_name_full in alias_names_of_celeb_here:
                            ## get the num_face_counts for the alias name
                            number_of_filtered_alias_name_laion_images_celeb_here = df_alias_name_laion_images[(df_alias_name_laion_images['celeb']==celeb_here) & (df_alias_name_laion_images['alias_name']==alias_name_full)]['num_face_counts'].values[0]
                            main_celeb_name, alias_name = eval(alias_name_full)
                            assert main_celeb_name == celeb_here, f"{main_celeb_name} is not equal to {celeb_here}."
                            if number_of_filtered_alias_name_laion_images_celeb_here > 0:
                                if alias_name == 'The Beast':
                                    alias_name = '"The Beast"'
                                assert os.path.exists(f"{alias_name_laion_filtered_images_directory}/{main_celeb_name}_{alias_name}.npy"), f"{alias_name} for {celeb_here} not found in the alias_name_laion_filtered_images_directory."
                                assert os.path.exists(f"{alias_name_laion_filtered_images_directory}/{main_celeb_name}_{alias_name}_laion_filtered_images_indices.npy"), f"{alias_name} for {celeb_here} not found in the alias_name_laion_filtered_images_directory."
                                alias_name_laion_filtered_images_embeddings.append(np.load(f"{alias_name_laion_filtered_images_directory}/{main_celeb_name}_{alias_name}.npy"))
                                alias_name_laion_filtered_images_indices.append(np.load(f"{alias_name_laion_filtered_images_directory}/{main_celeb_name}_{alias_name}_laion_filtered_images_indices.npy"))
                        
                        alias_name_laion_filtered_images_embeddings = np.concatenate(alias_name_laion_filtered_images_embeddings, axis=0)
                        alias_name_laion_filtered_images_indices = np.concatenate(alias_name_laion_filtered_images_indices, axis=0)
                        assert alias_name_laion_filtered_images_embeddings.shape[0] == number_of_filtered_alias_name_laion_images == alias_name_laion_filtered_images_indices.shape[0], f"{celeb_here} has {alias_name_laion_filtered_images_embeddings.shape[0]} laion images, and df_find_if_laion_images_exist has {number_of_filtered_alias_name_laion_images} images."
                        training_image_embeddings.append(alias_name_laion_filtered_images_embeddings)
                        using_entitys_filtered_alias_names_laion_images = True
                    
                    training_image_embeddings = np.concatenate(training_image_embeddings, axis=0)

                else:
                    ## SELECTED IMAGE EMBEDDINGS SECTION - if there are not 10 laion images that passed the filter, then we will use the selected images for comparison. 
                    training_image_embeddings, selected_images_from_internet = compute_similarity_selected_images(args, call_from_face_recoginition=True, call_from_face_recognition_celeb=celeb_here)

            else:
                ## TRAINING IMAGE EMBEDDINGS SECTION
                assert args.set_of_people == "celebrity", "This section is no more used, it was used earlier for celebrity data."
                if args.select_training_images_based_on_similarity_to_generated_images or args.select_training_images_based_on_similarity_to_generated_images_on_average:
                    args.top_k_training_images = 100000000    ## we want to process all the training images for all celebs, and later select them based on the similarity to the generated images.
                
                training_image_embeddings = []
                
                # if celeb_here in df_single_person_training_images.index and celeb_here not in celebrities_with_no_good_training_images:      ## These celebs had no good training image, I just added a dummy name for completion
                if len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]) > 0:
                    assert os.path.exists(f"{directory_string}{base_embeddings_directory}/training/{celeb_here}.json"), f"{celeb_here} not found in the training embeddings."
                    
                    with open(f"{directory_string}{base_embeddings_directory}/training/{celeb_here}.json", "r") as f:
                        this_celeb_training_image_embeddings = json.load(f)

                    for image in df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]['image'].values[:args.top_k_training_images]:
                        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                            training_image_embeddings.append(this_celeb_training_image_embeddings[image.split('/')[-1]][0]['embedding'])
                        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                            if image.split('/')[-1] in this_celeb_training_image_embeddings:
                                training_image_embeddings.append(this_celeb_training_image_embeddings[image.split('/')[-1]])
                        else:
                            raise Exception('Please specify a valid face recognition model.')
                    
                    training_image_embeddings = np.array(training_image_embeddings)
                    
                    if args.select_training_images_based_on_similarity_to_generated_images or args.select_training_images_based_on_similarity_to_generated_images_on_average:
                        assert training_image_embeddings.shape[0] == min(args.top_k_training_images, len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here])), f"{celeb_here} has {training_image_embeddings.shape[0]} training images, and df_single_person_training_images has {df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here].shape}"
                    else:       ## this is the case where we find similarity to the top-k training images selected based on number of matches and number of pixels, not based on the similarity to the generated images.
                        assert training_image_embeddings.shape[0] == min(args.top_k_training_images, len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here])), f"{celeb_here} has {training_image_embeddings.shape[0]} training images, and df_single_person_training_images has {df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here].shape[0]} images."

                ## INTERNET DOWNLOADED IMAGE EMBEDDINGS SECTION
                folder_internet_images = f"{directory_string}../high_quality_images_downloaded_from_internet/"
                if celeb_here.replace(" ", "-") in os.listdir(folder_internet_images):
                    this_celeb_images_from_internet = []
                    internet_image_embeddings = []
                    with open(f"{directory_string}{base_embeddings_directory}/images_I_downloaded_from_internet/{celeb_here}.json", "r") as f:
                        this_celeb_internet_downloaded_image_embeddings = json.load(f)
                    
                    # for image in os.listdir(f"{folder_internet_images}/{celeb_here.replace(' ', '-')}/"):
                        # if image in this_celeb_internet_downloaded_image_embeddings:        ## there could be images in the internet downloaded folder that have more than 1 face, and therefore embeddings will not be generated for them. 
                    for image in df_single_person_images_I_downloaded_from_internet[df_single_person_images_I_downloaded_from_internet['celeb']==celeb_here]['image'].values[:args.top_k_training_images]:
                        if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                            internet_image_embeddings.append(this_celeb_internet_downloaded_image_embeddings[image][0]['embedding'])
                        elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                            internet_image_embeddings.append(this_celeb_internet_downloaded_image_embeddings[image])
                        else:
                            raise Exception('Please specify a valid face recognition model.')
                        this_celeb_images_from_internet.append(f"{folder_internet_images}/{celeb_here.replace(' ', '-')}/{image}")

                    if len(internet_image_embeddings) > 0:
                        internet_image_embeddings = np.array(internet_image_embeddings)
                        if isinstance(training_image_embeddings, np.ndarray):
                            training_image_embeddings = np.concatenate((training_image_embeddings, internet_image_embeddings), axis=0)
                        elif isinstance(training_image_embeddings, list) and isinstance(internet_image_embeddings, np.ndarray):
                            training_image_embeddings = internet_image_embeddings
                        else:
                            raise Exception(f"{type(training_image_embeddings)} and {type(internet_image_embeddings)} are not supported.")

                if isinstance(training_image_embeddings, list):
                    print("celeb: ", celeb_here, " has no training images. We should either have training images from LAION-2B or from the internet.")
                    continue
                
            if training_image_embeddings.shape[0] < 10:
                print(f"WE HAVE ONLY {training_image_embeddings.shape[0]} REFERENCE IMAGES FOR ", celeb_here)
                if training_image_embeddings.shape[0] < 5:
                    continue

            assert isinstance(training_image_embeddings, np.ndarray), f"{celeb_here} for {args.image_generation_prompt} has no training images. We should either have training images from LAION-2B or from the internet. "

            ## NORMALIZE THE EMBEDDINGS
            training_image_embeddings = training_image_embeddings / (np.linalg.norm(training_image_embeddings, axis=1, keepdims=True, ord=2) + 1e-16) 
            
            if generated_image_with_one_face_embeddings.shape[0] == 0:
                generated_image_with_one_face_embeddings = np.zeros_like(training_image_embeddings)
            else:
                generated_image_with_one_face_embeddings = generated_image_with_one_face_embeddings / (np.linalg.norm(generated_image_with_one_face_embeddings, axis=1, keepdims=True, ord=2) + 1e-16)

            ## GET THE COSINE SIMILARITY BETWEEN THE TRAINING IMAGES AND THE GENERATED IMAGES
            cosine_similarities_training_and_generated_images = np.dot(training_image_embeddings, generated_image_with_one_face_embeddings.T)
            assert cosine_similarities_training_and_generated_images.shape == (training_image_embeddings.shape[0], generated_image_with_one_face_embeddings.shape[0])
            assert np.all(cosine_similarities_training_and_generated_images >= -1 - 1e-8) and np.all(cosine_similarities_training_and_generated_images <= 1 + 1e-8), f"cosine_similarities_training_and_generated_images is: {cosine_similarities_training_and_generated_images}"
            
            if args.select_training_images_based_on_similarity_to_generated_images:
                args.top_k_training_images = "top_10_similar_to_each_generated_image"
            elif args.select_training_images_based_on_similarity_to_generated_images_on_average:
                args.top_k_training_images = "top_10_similar_to_average_generated_images"

            ## DEPENDING ON THE MANNER OF SELECTING WHICH TRAINING IMAGES TO USE FOR COMPARISON, GET THE COSINE SIMILARITY BETWEEN THE TRAINING IMAGES AND THE GENERATED IMAGES
            if use_selected_images_for_matching:
                ## here we will use all the selected images to compare to the generated images
                cosine_similarities_training_and_generated_images = np.sort(cosine_similarities_training_and_generated_images, axis=0)[::-1]
                assert cosine_similarities_training_and_generated_images.shape == (training_image_embeddings.shape[0], generated_image_with_one_face_embeddings.shape[0])
                ## take the max similarity of the generated images to all selected images
                cosine_similarities_training_and_generated_images = np.max(cosine_similarities_training_and_generated_images, axis=0)
                args.top_k_training_images = "all_selected_images"
                
                import cv2
                assert selected_images_from_internet is not None and len(selected_images_from_internet) == training_image_embeddings.shape[0]
                for i in range(len(selected_images_from_internet)):
                    image_read = cv2.imread(selected_images_from_internet[i])
                    face_cutout = image_read
                    base_directory = f"{directory_string}{extraced_face_cuts_directory}/selected_images_from_internet/{celeb_here}"
                    os.makedirs(base_directory, exist_ok=True)
                    save_directory = f"{base_directory}/image_{i}.jpg"
                    cv2.imwrite(save_directory, face_cutout)
            
            elif use_filtered_laion_images_and_selected_images_for_matching:
                assert using_entitys_filtered_laion_images in [True, False] and using_entitys_filtered_alias_names_laion_images in [True, False]
                import cv2
                indices_training_images_with_highest_similarity_to_generated_images = np.argsort(np.mean(cosine_similarities_training_and_generated_images, axis=1))[::-1][:num_training_images_to_compare]
                save_cut_out_faces = False
                
                if (using_entitys_filtered_laion_images or using_entitys_filtered_alias_names_laion_images) and save_cut_out_faces:
                    ## get the face cut outs of the images that are closest to the generated images.
                    assert (laion_filtered_images_indices is not None and laion_filtered_images_embeddings is not None) or (alias_name_laion_filtered_images_indices is not None and alias_name_laion_filtered_images_embeddings is not None)
                    
                    if using_entitys_filtered_laion_images:
                        with open(f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{base_embeddings_directory}/all_laion_images/{celeb_here}.json", "r") as f:
                            laion_images_embeddings_file = json.load(f)
                            print("Using LAION FACE FOR COMPARISON")
                            num_laion_filtered_images = laion_filtered_images_embeddings.shape[0]

                    if using_entitys_filtered_alias_names_laion_images:     ## both of these can be true at the same time. 
                        assert args.consider_name_aliases and args.set_of_people == "celebrity"
                        alias_names_of_celeb_here = df_alias_name_laion_images[df_alias_name_laion_images['celeb']==celeb_here]['alias_name'].values
                        # raise NotImplementedError       ## we haven't completed the implementation of this part, as we are not saving the extracted face cut outs any more. 
                        for alias_name_full in alias_names_of_celeb_here:
                            main_celeb_name, alias_name = eval(alias_name_full)
                            with open(f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{base_embeddings_directory}/all_laion_alias_name_images/{alias_name}.json", "r") as f:
                                alias_names_laion_images_embeddings_file = json.load(f)
                                print("Using ALIAS NAME LAION FACE FOR COMPARISON")
                                num_alias_name_laion_filtered_images = alias_name_laion_filtered_images_embeddings.shape[0]
                    
                    ## indices_training_images_with_highest_similarity_to_generated_images can have indices both from laion_filtered_images_indices and alias_name_laion_filtered_images_indices. Therefore, we need to get their indices independently. 
                    ## now we have the indices of laion filtered images that are closest to the training images, we need to get its indices in 
                    laion_filtered_training_images_with_highest_similarity_to_generated_images = []
                    alias_name_laion_filtered_images_training_images_with_highest_similarity_to_generated_images = []
                    for i in range(len(indices_training_images_with_highest_similarity_to_generated_images)):
                        if indices_training_images_with_highest_similarity_to_generated_images[i] < num_laion_filtered_images:
                            laion_filtered_training_images_with_highest_similarity_to_generated_images.append(laion_filtered_images_indices[indices_training_images_with_highest_similarity_to_generated_images[i]])
                        else:
                            alias_name_laion_filtered_images_training_images_with_highest_similarity_to_generated_images.append(alias_name_laion_filtered_images_indices[indices_training_images_with_highest_similarity_to_generated_images[i] - num_laion_filtered_images])

                    assert len(laion_filtered_training_images_with_highest_similarity_to_generated_images) + len(alias_name_laion_filtered_images_training_images_with_highest_similarity_to_generated_images) == min(num_training_images_to_compare, training_image_embeddings.shape[0])

                    ## indices_of_closest_images_in_the_full_set_of_laion_images = laion_filtered_images_indices[indices_training_images_with_highest_similarity_to_generated_images]
                    ## assert indices_of_closest_images_in_the_full_set_of_laion_images.shape == (min(num_training_images_to_compare, laion_filtered_images_embeddings.shape[0]),)
                    
                    ## we want to get the face cut outs of the faces in laion embeddings that in the laion_filtered_images_indices
                    closest_laion_faces_this_celeb = []
                    if using_entitys_filtered_laion_images:
                        for image_face_here in laion_images_embeddings_file:
                            if image_face_here in laion_filtered_training_images_with_highest_similarity_to_generated_images:
                                closest_laion_faces_this_celeb.append(image_face_here)
                    if using_entitys_filtered_alias_names_laion_images:
                        assert args.consider_name_aliases
                        for image_face_here in alias_names_laion_images_embeddings_file:
                            if image_face_here in alias_name_laion_filtered_images_training_images_with_highest_similarity_to_generated_images:
                                closest_laion_faces_this_celeb.append(image_face_here)
                    assert len(closest_laion_faces_this_celeb) == min(num_training_images_to_compare, laion_filtered_images_embeddings.shape[0])
                    
                    ## now we want to get the face cut outs of these images.
                    ## the size of the face is saved in the key of the image_face_boundingbox like this: "image_seq_1.jpg_[114.41725   35.908657 151.3013    82.81814 ]": embedding. We want to get these values and then compute the area of the face.
                    seq_index = 0
                    for image_face_now in closest_laion_faces_this_celeb:
                        bounding_box = image_face_now.split('_[')[1].split(']')[0]
                        x1, y1, x2, y2 = [float(x) for x in bounding_box.split()]
                        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                        if x1 < 0: x1 = 0
                        if y1 < 0: y1 = 0
                        assert x1 <= x2 and y1 <= y2
                        image_name = image_face_now.split('_[')[0]
                        if args.set_of_people == "celebrity":
                            image_read = cv2.imread(f"{directory_string}../all_downloaded_images/{celeb_here}/{image_name}")
                        elif args.set_of_people == "politicians":
                            image_read = cv2.imread(f"/gscratch/scrubbed/vsahil/all_downloaded_images_politicians/{celeb_here}/{image_name}")
                        else:
                            raise NotImplementedError
                        face_cutout = image_read[y1:y2, x1:x2]
                        face_area = (x2 - x1) * (y2 - y1)
                        base_directory = f"{directory_string}{extraced_face_cuts_directory}/laion_filtered_images/{celeb_here}"
                        os.makedirs(base_directory, exist_ok=True)
                        save_directory = f"{base_directory}/{image_name}_face_{seq_index}_face_area_{face_area}.jpg"
                        cv2.imwrite(save_directory, face_cutout)
                        seq_index += 1
                        
                elif not using_entitys_filtered_laion_images and not using_entitys_filtered_alias_names_laion_images:
                    ## here we will get the selected_images_from_internet
                    assert selected_images_from_internet is not None and len(selected_images_from_internet) == training_image_embeddings.shape[0]
                    for i in range(len(selected_images_from_internet)):
                        image_read = cv2.imread(selected_images_from_internet[i])
                        face_cutout = image_read
                        base_directory = f"{directory_string}{extraced_face_cuts_directory}/selected_images_from_internet/{celeb_here}"
                        os.makedirs(base_directory, exist_ok=True)
                        save_directory = f"{base_directory}/image_{i}.jpg"
                        cv2.imwrite(save_directory, face_cutout)
                
                assert indices_training_images_with_highest_similarity_to_generated_images.shape == (min(num_training_images_to_compare, training_image_embeddings.shape[0]),)
                cosine_similarities_training_and_generated_images = cosine_similarities_training_and_generated_images[indices_training_images_with_highest_similarity_to_generated_images]
                assert cosine_similarities_training_and_generated_images.shape == (min(num_training_images_to_compare, training_image_embeddings.shape[0]), generated_image_with_one_face_embeddings.shape[0])
                cosine_similarities_training_and_generated_images = np.mean(cosine_similarities_training_and_generated_images, axis=0)
                args.top_k_training_images = "filtered_laion_images_and_selected_images"
                    
            else:
                assert args.set_of_people == "celebrity", "This section is no more used, it was used earlier for celebrity data."
                if args.select_training_images_based_on_similarity_to_generated_images:     ## here we need to only consider the similarity of each generated images with 10 training images that have the highest similarity to them. 
                    cosine_similarities_training_and_generated_images = np.sort(cosine_similarities_training_and_generated_images, axis=0)[::-1][:num_training_images_to_compare]     ## for each generated image, take the top-10 highest cosine similarities with the training images. The axis will be 0 here as that is axis we are taking the mean later. 
                    assert cosine_similarities_training_and_generated_images.shape == (min(10, training_image_embeddings.shape[0]), generated_image_with_one_face_embeddings.shape[0])
                    cosine_similarities_training_and_generated_images = np.mean(cosine_similarities_training_and_generated_images, axis=0)
                
                elif args.select_training_images_based_on_similarity_to_generated_images_on_average:       ## here we need to only consider the similarity of each generated images with 10 training images that have the highest similarity to them.
                    indices_training_images_with_highest_similarity_to_generated_images = np.argsort(np.mean(cosine_similarities_training_and_generated_images, axis=1))[::-1][:num_training_images_to_compare]
                    ## get the names of these closest training images, and we will save them to a file. 
                    # Note that the images will be from both df_single_person_training_images and the images downloaded from Internet. -- so we need to make sure we are getting images from both of them. There could be indices in indices_training_images_with_highest_similarity_to_generated_images that are from the images downloaded from Internet, and not from df_single_person_training_images. So we need to make sure we are getting the right images.
                    if len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]) > 0:
                        indices_in_df_single_person_training_images = np.where(indices_training_images_with_highest_similarity_to_generated_images < len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]))[0]
                        indices_in_df_single_person_training_images = indices_training_images_with_highest_similarity_to_generated_images[indices_in_df_single_person_training_images]
                        training_images_with_highest_similarity_to_generated_images_name = df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]['image'].values[indices_in_df_single_person_training_images].tolist()
                        indices_in_images_from_internet = np.where(indices_training_images_with_highest_similarity_to_generated_images >= len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here]))[0]
                        indices_in_images_from_internet = indices_training_images_with_highest_similarity_to_generated_images[indices_in_images_from_internet]
                        indices_in_images_from_internet -= len(df_single_person_training_images[df_single_person_training_images['celeb']==celeb_here])        ## this is the number of images in df_single_person_training_images
                        training_images_with_highest_similarity_to_generated_images_name.extend([this_celeb_images_from_internet[i] for i in indices_in_images_from_internet])
                    else:
                        indices_in_images_from_internet = indices_training_images_with_highest_similarity_to_generated_images
                        training_images_with_highest_similarity_to_generated_images_name = [this_celeb_images_from_internet[i] for i in indices_in_images_from_internet]
                    
                    with open(f"top_10_training_images_close_to_generated_images_prompt_{args.image_generation_prompt_id}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}.csv", "a") as f:
                        for i in range(len(training_images_with_highest_similarity_to_generated_images_name)):
                            f.write(f"{celeb_here}|{training_images_with_highest_similarity_to_generated_images_name[i]}\n")
                    
                    assert indices_training_images_with_highest_similarity_to_generated_images.shape == (min(num_training_images_to_compare, training_image_embeddings.shape[0]),)
                    cosine_similarities_training_and_generated_images = cosine_similarities_training_and_generated_images[indices_training_images_with_highest_similarity_to_generated_images]
                    assert cosine_similarities_training_and_generated_images.shape == (min(num_training_images_to_compare, training_image_embeddings.shape[0]), generated_image_with_one_face_embeddings.shape[0])
                    cosine_similarities_training_and_generated_images = np.mean(cosine_similarities_training_and_generated_images, axis=0)
                
                else:       ## here we are taking the similarities with the top-20 training images, and hence taking the mean makes sense.
                    cosine_similarities_training_and_generated_images = np.mean(cosine_similarities_training_and_generated_images, axis=0)
                
            assert cosine_similarities_training_and_generated_images.shape == (generated_image_with_one_face_embeddings.shape[0], ), f"cosine_similarities_training_and_generated_images.shape: {cosine_similarities_training_and_generated_images.shape}, generated_image_with_one_face_embeddings.shape: {generated_image_with_one_face_embeddings.shape}"
            
            highest_5_cosine_similarities = np.sort(cosine_similarities_training_and_generated_images)[::-1][:5]
            highest_5_cosine_similarities_index = np.argsort(cosine_similarities_training_and_generated_images)[::-1][:5]
            print(f"Average cosine similarity for {celeb_here}", np.mean(cosine_similarities_training_and_generated_images), "and mean of highest-5 cosine similarities is ", np.mean(highest_5_cosine_similarities))
            
            effective_num_face_counts_total = 0
            effective_total_face_area_total = 0
            
            effective_num_face_counts_this_celeb_laion_images = df_find_if_laion_images_exist[df_find_if_laion_images_exist['celeb']==celeb_here]['effective_num_face_counts'].values[0]
            effective_total_matching_face_area_this_celeb_laion_images = df_find_if_laion_images_exist[df_find_if_laion_images_exist['celeb']==celeb_here]['effective_face_area'].values[0]
            effective_total_face_area_this_celeb_laion_images = effective_total_matching_face_area_this_celeb_laion_images
            
            effective_num_face_counts_total += effective_num_face_counts_this_celeb_laion_images
            effective_total_face_area_total += effective_total_face_area_this_celeb_laion_images
            
            if args.stable_diffusion_model == "v2":     ## here the training dataset was LAION 5B, so we need to scale the effective number of face counts and effective face area. 
                if args.set_of_people == "celebrity":
                    downloaded_images_directory = "/gscratch/h2lab/vsahil/vlm-efficiency/all_downloaded_images"
                elif args.set_of_people == "politicians":
                    downloaded_images_directory = "/gscratch/scrubbed/vsahil/all_downloaded_images_politicians"
                else:   raise NotImplementedError
                downloaded_images_this_celeb_directory = f"{downloaded_images_directory}/{celeb_here}"
                num_downloaded_images = len([x for x in os.listdir(downloaded_images_this_celeb_directory) if not x.endswith(".csv")])
                total_matching_captions = df_laion5b[df_laion5b['Name']==celeb_here]['counts_in_laion5b'].values[0]
                if num_downloaded_images != 0:  ## only when the downloaded images are not 0 will we change the effective number of face counts and effective face area based on caption count in LAION 5B 
                    counts_in2b = df_total_captions[df_total_captions['Name']==celeb_here]['counts_in_laion2b-en'].values[0]
                    effective_num_face_counts_total *= total_matching_captions / counts_in2b
                    effective_total_face_area_total *= total_matching_captions / counts_in2b
            else:
                total_matching_captions = df_total_captions[df_total_captions['Name']==celeb_here]['counts_in_laion2b-en'].values[0]
                

            if args.consider_name_aliases:
                assert args.set_of_people == "celebrity" and args.stable_diffusion_model != "v2"      ## we need to compute the alias names in LAION 5B for this to work.
                ## now get the effective number of face counts for the alias names of the celeb
                alias_names_of_celeb_here = df_alias_name_laion_images[df_alias_name_laion_images['celeb']==celeb_here]['alias_name'].values
                for alias_name_full in alias_names_of_celeb_here:
                    main_celeb_name, alias_name = eval(alias_name_full)
                    effective_num_face_counts_this_celeb_alias_name_laion_images = df_alias_name_laion_images[(df_alias_name_laion_images['celeb']==celeb_here) & (df_alias_name_laion_images['alias_name']==alias_name_full)]['effective_num_face_counts'].values[0]
                    effective_total_face_area_alias_name_laion_images = df_alias_name_laion_images[(df_alias_name_laion_images['celeb']==celeb_here) & (df_alias_name_laion_images['alias_name']==alias_name_full)]['effective_face_area'].values[0]
                    effective_num_face_counts_total += effective_num_face_counts_this_celeb_alias_name_laion_images
                    effective_total_face_area_total += effective_total_face_area_alias_name_laion_images
            
            write_results_to_file = True

            if write_results_to_file:
                file_name = f"average_cosine_similarity_closeup_training_and_single_face_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}.csv"
                if use_filtered_laion_images_and_selected_images_for_matching:
                    file_name =f"average_cosine_similarity_closeup_training_and_single_face_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}_{args.threshold_for_max_f1_score}.csv"

                with open(file_name, "a") as f:
                    if seq_celeb == 0 or not did_not_write_header:
                        did_not_write_header = True
                        f.write("celeb|num_single_person_training_images_plus_internet_images|num_single_face_generated_images|total_matching_captions|average_cosine_similarity|Mean_top_5_cosine_similarity|num_downloaded_training_images|effective_num_face_counts|effective_total_face_area\n")
                    f.write(f"{celeb_here}|{training_image_embeddings.shape[0]}|{generated_image_with_one_face_embeddings.shape[0]}|{total_matching_captions}|{np.mean(cosine_similarities_training_and_generated_images)}|{np.mean(highest_5_cosine_similarities)}|{num_downloaded_training_images[celeb_here]}|{effective_num_face_counts_total}|{effective_total_face_area_total}\n")

    else:
        raise Exception('Please specify either --generate_embeddings_face_recognition or --match_embeddings_face_recognition')


def helper_function_test_face_recognition_models_for_bias(args, celeb_here, different_celeb):
    import cv2
    from scipy.spatial.distance import cosine

    if args.face_recognition_model == 'Amazon_Rekognition':
        import boto3
        session = boto3.Session()
        amazon_client = session.client('rekognition')

    elif args.face_recognition_model == 'edenai':
        ## The free version of this is too slow to check. 
        ## Starting with Amazon Rekognition. import json
        # import requests
        # headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNGEyNjhiZTItYzIwMC00YjNiLWI3YTEtMDlmNTNiYWUxNGRlIiwidHlwZSI6ImFwaV90b2tlbiJ9.efUE98-epKOhPxoyXucv5rrSxZhl3ow7tZfmfuSBAwQ"}
        # url = "https://api.edenai.run/v2/image/face_compare"
        # data = {
        #     "providers": "facepp",
        #     "fallback_providers": "amazon",
        #     "show_original_response": "true"
        # }
        
        # files = {
        #     'file2': open("/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/Ayo-Edebiri/image_seq_73.jpg", 'rb'),
        #     'file1': open("/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/Ayo-Edebiri/image_seq_50.jpg", 'rb'),
        # }
        # response = requests.post(url, data=data, files=files, headers=headers)
        # result = json.loads(response.text)
        # print(result)
        # print(result['amazon']['original_response']['FaceMatches'][0]['Similarity'])
        # recognition_model_type = "expensive"        ## for the expensive model, we will not parallelize celebs within a group as that will lead to double computation which is expensive.
        pass

    elif args.face_recognition_model in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]:
        from deepface import DeepFace
        from retinaface import RetinaFace
        loaded_model = None
        retinfacemodel = None

    elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
        import insightface
        from insightface.app import FaceAnalysis
        insightfaceapp = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
        
        if args.face_recognition_model == 'insightface_resnet100':
            insightface_recognition = insightface.model_zoo.get_model(args.insightface_model_path)     ## here we are going to use a different model for face recognition.
            insightface_recognition.prepare(ctx_id=0)

    elif args.face_recognition_model in ['GhostFaceNetV1-0.5-2_A','GhostFaceNetV1-1.3-1_C', 'GhostFaceNet_W1.3_S1_ArcFace']:
        ## here we need to get the face detection of the images, and only pass the extracted faces to the model.
        from retinaface import RetinaFace
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        model_path = f"/gscratch/h2lab/vsahil/GhostNet/{args.face_recognition_model}.h5"
        ghostfacenetmodel = tf.keras.models.load_model(model_path)
        retinfacemodel = None
            
        def prepare_image_ghostfacenet(image_path, detected_faces):
            ## first load the image in original size and extract the face from it.
            original_img = img_to_array(load_img(image_path))
            ## get the bounding box of the face
            x1, y1, x2, y2 = detected_faces['face_1']['facial_area']
            ## crop the face from the original image
            img = original_img[y1:y2, x1:x2]
            ## resize the image to 112x112
            img = tf.image.resize(img, (112, 112))
            # img = load_img(image_path, target_size=(112, 112))
            # img = img_to_array(img)
            img = (img - 127.5) * 0.0078125
            img = np.expand_dims(img, axis=0)
            return img

    ## get the good images of these two celebs rights here
    good_images_of_each_celeb = {}
    good_images_of_each_celeb[celeb_here] = []
    if celeb_here != different_celeb:
        good_images_of_each_celeb[different_celeb] = []
        get_images_for = [celeb_here, different_celeb]
    else:
        get_images_for = [celeb_here]
    for this_celeb in get_images_for:
        for good_image in os.listdir(f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{this_celeb.replace(' ', '-')}/"):
            ## only add the image if its extension is .jpeg, .jpg, or .png
            if good_image.endswith('.jpeg') or good_image.endswith('.jpg') or good_image.endswith('.png'):
                good_images_of_each_celeb[this_celeb].append(f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{this_celeb.replace(' ', '-')}/{good_image}")
            if len(good_images_of_each_celeb[this_celeb]) == 10:     ## let's do only 5 good images per celeb
                break
    
    if args.metric_fmr:
        ## here we will send the inner most loop of the previious function, so that everything can be parallelized.
        similarity_scores_this_pair_of_celebs = []
        for image1 in good_images_of_each_celeb[celeb_here]:
            if args.face_recognition_model == 'Amazon_Rekognition':
                with open(image1, 'rb') as source_image:
                    source_bytes = source_image.read()
            
            elif args.face_recognition_model in ["insightface_buffalo", "insightface_resnet100"]:
                image_here1 = cv2.imread(image1)
                insightfacefaces1 = insightfaceapp.get(image_here1)
                if len(insightfacefaces1) > 0:
                    if args.face_recognition_model == 'insightface_buffalo':
                        insightfaceembedding1 = insightfacefaces1[0].normed_embedding          ## we are assuming there is atleast one face in each image.
                    elif args.face_recognition_model == 'insightface_resnet100':
                        insightfaceembedding1 = insightface_recognition.get(image_here1, insightfacefaces1[0])
                    insightfaceembedding1 = insightfaceembedding1 / (np.linalg.norm(insightfaceembedding1, ord=2) + 1e-16)
                else:
                    print(f"No face detected in image {image1} for celeb {celeb_here}")
                    continue
                
            elif args.face_recognition_model in ['GhostFaceNetV1-0.5-2_A','GhostFaceNetV1-1.3-1_C', 'GhostFaceNet_W1.3_S1_ArcFace']:
                ## first use face detection to get the face from the image, and then pass it to the model. For face detection we will use RetinaFace.
                detected_faces, retinfacemodel = RetinaFace.detect_faces(image1, model=retinfacemodel)
                img1_ready = prepare_image_ghostfacenet(image1, detected_faces)
                embedding1_ghostfacenet = ghostfacenetmodel.predict(img1_ready)
                embedding1_ghostfacenet = embedding1_ghostfacenet / (np.linalg.norm(embedding1_ghostfacenet, ord=2) + 1e-16)
                
            for image2 in good_images_of_each_celeb[different_celeb]:
                if image1 == image2:        ## this will happen in the case of args.metric_fnmr
                    assert args.metric_fnmr
                    continue
                
                if args.face_recognition_model == 'Amazon_Rekognition':
                    with open(image2, 'rb') as target_image:
                        target_bytes = target_image.read()
                    response = amazon_client.compare_faces(SimilarityThreshold=0, SourceImage={'Bytes': source_bytes}, TargetImage={'Bytes': target_bytes})        ## SimilarityThreshold of 0 is crucial, otherwise we will not get score any smaller than its value. Minimum value of SimilarityThreshold is 0. 
                    similarity_value = response['FaceMatches'][0]['Similarity']
                
                elif args.face_recognition_model == 'edenai':
                    pass
                
                elif args.face_recognition_model in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]:    ## these are the Deepface models. 
                    response, loaded_model = DeepFace.verify(image1, image2, model_name=args.face_recognition_model, distance_metric='cosine', enforce_detection=True, detector_backend='retinaface', align=False, model=loaded_model)        ## since these are all single person images, we should enforce detection.
                    similarity_value = 1 - response['distance']     ## remember that we are getting cosinedistance, and not cosine similarity.
                
                elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                    image_here2 = cv2.imread(image2)
                    insightfacefaces2 = insightfaceapp.get(image_here2)
                    if len(insightfacefaces2) > 0:
                        if args.face_recognition_model == 'insightface_buffalo':
                            insightfaceembedding2 = insightfacefaces2[0].normed_embedding          ## we are assuming there is atleast one face in each image.
                        elif args.face_recognition_model == 'insightface_resnet100':
                            insightfaceembedding2 = insightface_recognition.get(image_here2, insightfacefaces2[0])
                        insightfaceembedding2 = insightfaceembedding2 / (np.linalg.norm(insightfaceembedding2, ord=2) + 1e-16)
                        similarity_value = 1 - cosine(insightfaceembedding1, insightfaceembedding2)
                    else:
                        print(f"No face detected in image {image2} for celeb {different_celeb}")
                        continue
                
                elif args.face_recognition_model in ['GhostFaceNetV1-0.5-2_A','GhostFaceNetV1-1.3-1_C', 'GhostFaceNet_W1.3_S1_ArcFace']:
                    detected_faces, retinfacemodel = RetinaFace.detect_faces(image2, model=retinfacemodel)
                    img2_ready = prepare_image_ghostfacenet(image2, detected_faces)
                    embedding2_ghostfacenet = ghostfacenetmodel.predict(img2_ready)
                    embedding2_ghostfacenet = embedding2_ghostfacenet / (np.linalg.norm(embedding2_ghostfacenet, ord=2) + 1e-16)
                    similarity_value = 1 - cosine(embedding1_ghostfacenet.squeeze(), embedding2_ghostfacenet.squeeze())

                else:
                    raise Exception('Please specify a valid face recognition model')

                similarity_scores_this_pair_of_celebs.append(similarity_value)
            print(f"image1 {image1} done for celeb {celeb_here}")
        print("Now done with celeb", celeb_here, "and celeb", different_celeb, "the length of similarity_scores_this_pair_of_celebs is", len(similarity_scores_this_pair_of_celebs))
    
    elif args.metric_fnmr and args.face_recognition_model != 'Amazon_Rekognition':
        face_embeddings = []
        ## here we just get the faces and embeddings of the images of this celeb and the compute the similarity between all pairs of images -- will save compute
        for image in good_images_of_each_celeb[celeb_here]:
            if args.face_recognition_model in ["insightface_buffalo", "insightface_resnet100"]:
                image_here = cv2.imread(image)
                insightfacefaces = insightfaceapp.get(image_here)
                if len(insightfacefaces) > 0:
                    if args.face_recognition_model == 'insightface_buffalo':
                        insightfaceembedding = insightfacefaces[0].normed_embedding
                    elif args.face_recognition_model == 'insightface_resnet100':
                        insightfaceembedding = insightface_recognition.get(image_here, insightfacefaces[0])
                    insightfaceembedding = insightfaceembedding / (np.linalg.norm(insightfaceembedding, ord=2) + 1e-16)
                    face_embeddings.append(insightfaceembedding)
            
            elif args.face_recognition_model in ['GhostFaceNetV1-0.5-2_A','GhostFaceNetV1-1.3-1_C', 'GhostFaceNet_W1.3_S1_ArcFace']:
                detected_faces, retinfacemodel = RetinaFace.detect_faces(image, model=retinfacemodel)
                if isinstance(detected_faces, list) and detected_faces[0] == "No face detected":
                    print(f"No face detected in image {image} for celeb {celeb_here}")
                    continue
                img_ready = prepare_image_ghostfacenet(image, detected_faces)
                embedding_ghostfacenet = ghostfacenetmodel.predict(img_ready)
                embedding_ghostfacenet = embedding_ghostfacenet / (np.linalg.norm(embedding_ghostfacenet, ord=2) + 1e-16)
                face_embeddings.append(embedding_ghostfacenet)
            
            elif args.face_recognition_model in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]:
                detected_faces, retinfacemodel = RetinaFace.detect_faces(image, model=retinfacemodel)
                if isinstance(detected_faces, list) and detected_faces[0] == "No face detected":
                    print(f"No face detected in image {image} for celeb {celeb_here}")
                    continue
                original_img = cv2.imread(image)
                x1, y1, x2, y2 = detected_faces['face_1']['facial_area']
                face = original_img[y1:y2, x1:x2]
                represent, loaded_model = DeepFace.represent(face, model_name=args.face_recognition_model, model=loaded_model, enforce_detection=False, detector_backend='skip', align=False)       ## we have already detected the face
                embedding = represent[0]['embedding']
                embedding = embedding / (np.linalg.norm(embedding, ord=2) + 1e-16)
                face_embeddings.append(embedding)
        
        face_embeddings = np.array(face_embeddings) ## these embeddings are already normalized
        face_embeddings = face_embeddings.squeeze()
        print("face_embeddings.shape", face_embeddings.shape)
        ## now compute the similarity between all pairs of images. 
        cosine_similarities = np.dot(face_embeddings, face_embeddings.T)
        assert cosine_similarities.shape == (face_embeddings.shape[0], face_embeddings.shape[0])
        assert np.all(cosine_similarities >= -1 - 1e-6) and np.all(cosine_similarities <= 1 + 1e-6), f"cosine_similarities is: {cosine_similarities}"
        # Note that we would want to remove the similarity between the same image, as that is not useful.
        similarity_scores_this_pair_of_celebs = [float(score) for score in cosine_similarities[np.triu_indices(face_embeddings.shape[0], k=1)]]     ## we are only interested in the upper triangular part of the matrix, as the lower triangular part is just the transpose of the upper triangular part (so the avg will make it same anyway)., and this removes the diagonal as well. We need the float for JSON serialization.
        assert len(similarity_scores_this_pair_of_celebs) == face_embeddings.shape[0] * (face_embeddings.shape[0] - 1) // 2
        print("Now done with celeb", celeb_here, "and celeb", different_celeb, "the length of similarity_scores_this_pair_of_celebs is", len(similarity_scores_this_pair_of_celebs))
        
    elif args.metric_fnmr and args.face_recognition_model == 'Amazon_Rekognition':
        ## here we query Amazon Rekognition for the similarity between all pairs of images. Do it for each unique pair of images, as we want to save compute.
        ## get the unique pairs of images
        unique_pair_of_images = list(itertools.combinations(good_images_of_each_celeb[celeb_here], 2))
        assert len(unique_pair_of_images) == len(good_images_of_each_celeb[celeb_here]) * (len(good_images_of_each_celeb[celeb_here]) - 1) // 2
        similarity_scores_this_pair_of_celebs = []
        for image1, image2 in unique_pair_of_images:
            with open(image1, 'rb') as source_image:
                source_bytes = source_image.read()
            with open(image2, 'rb') as target_image:
                target_bytes = target_image.read()

            response = amazon_client.compare_faces(SimilarityThreshold=0, SourceImage={'Bytes': source_bytes}, TargetImage={'Bytes': target_bytes})        ## SimilarityThreshold of 0 is crucial, otherwise we will not get score any smaller than its value. Minimum value of SimilarityThreshold is 0. 
            similarity_value = response['FaceMatches'][0]['Similarity']
            similarity_scores_this_pair_of_celebs.append(similarity_value)
        print("Now done with celeb", celeb_here, "and celeb", different_celeb, "the length of similarity_scores_this_pair_of_celebs is", len(similarity_scores_this_pair_of_celebs))        

    return similarity_scores_this_pair_of_celebs


def worker_test_bias(args, celeb_here, different_celeb, similarity_scores_for_different_celebs):
    print(f"Starting celeb {celeb_here} and celeb {different_celeb} inside worker function.")
    result = helper_function_test_face_recognition_models_for_bias(args, celeb_here, different_celeb)
    print(f"celeb {celeb_here} and celeb {different_celeb} done inside worker function, the length of result is {len(result)}")
    if args.metric_fmr:
        similarity_scores_for_different_celebs[celeb_here][different_celeb] = result
        similarity_scores_for_different_celebs[different_celeb][celeb_here] = similarity_scores_for_different_celebs[celeb_here][different_celeb]       ## also copy these scores to the other side
    elif args.metric_fnmr:
        similarity_scores_for_different_celebs[celeb_here] = result


def error_callback_test_bias(e):
    print('Error:', e)

                       
def test_face_recognition_models_for_bias(args):
    male_white_celebs = ['Gabriel LaBelle', 'Dominic Sessa', 'Corey Mylchreest', 'Sam Nivola', 'Tom Blyth', 'Jordan Firstman', 'Josh Seiter', 'Nicola Porcella', 'Armen Nahapetian', 'Joey Klaasen']
    male_black_celebs = ['Jaylin Webb', 'Quincy Isaiah', 'Miles Gutierrez-Riley', 'Jalyn Hall', 'Myles Frost', 'Wisdom Kaye', 'Olly Sholotan', 'Isaiah R. Hill', "Bobb'e J. Thompson", 'Myles Truitt']
    male_brown_celebs = ['Sajith Rajapaksa', 'Aryan Simhadri', 'Aditya Kusupati', 'Vihaan Samat', 'Ishwak Singh', 'Gurfateh Pirzada', 'Pavail Gulati', 'Cwaayal Singh', 'Jibraan Khan', 'Vedang Raina']
    
    female_white_celebs = ['Gabby Windey', 'Mia Challiner', 'Isabel Gravitt', 'Pardis Saremi', 'Elle Graham', 'Cara Jade Myers', 'Ali Skovbye', 'Hannah Margaret Selleck', 'Bridgette Doremus', 'Milly Alcock']
    female_black_celebs = ['Kudakwashe Rutendo', 'Ayo Edebiri', 'Kaci Walfall', 'Elisha Williams', 'Laura Kariuki', 'Akira Akbar', 'Savannah Lee Smith', 'Samara Joy', 'Arsema Thomas', 'Leah Jeffries']
    female_brown_celebs = ['Priya Kansara', 'Pashmina Roshan', 'Banita Sindhu', 'Alaia Furniturewala', 'Paloma Dhillon', 'Alizeh Agnihotri', 'Geetika Vidya Ohlyan', 'Saloni Batra', 'Sharvari Wagh', 'Arjumman Mughal']

    parallelize_celebs = args.test_model_parallel
    os.makedirs(f"face_recognition_bias_scores/{args.face_recognition_model}", exist_ok=True)
    race_groups = ['white', 'brown', 'black']
    gender_groups = ['male', 'female']
    
    if args.plot_stacked_bar_chart_for_face_recognition_performance:
        plot_for_paper = True
        plot_matplotlib = False
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        # Step 1: Read the CSV file
        filename = f"face_recognition_bias_avg_scores_{'fmr' if args.metric_fmr else 'fnmr'}.csv"
        df = pd.read_csv(filename, sep='|')
        sns.set_context("paper", font_scale=1.5)
        # sns.set_style("whitegrid")
        # sns.set_palette(sns.color_palette("pastel"))
        sns.set_palette(sns.color_palette("deep"))
        plt.rcParams['font.family'] = 'serif'

        # Define your custom order for Groups and Models
        group_order = ['male_white_celebs', 'female_white_celebs', 'male_brown_celebs', 'female_brown_celebs', 'male_black_celebs', 'female_black_celebs']
        print_caption_count = False
        if print_caption_count:
            df_total_captions = pd.read_csv('../celebrity_data/celebrities_sorted.csv', sep=',')
            for group in group_order:
                for celeb_here in eval(group):
                    print("celeb here is", celeb_here)
                    print(f"Number of captions for {celeb_here} is", df_total_captions[df_total_captions['Name']==celeb_here]['counts_in_laion2b-en'].values[0])
        
        model_order = ['Amazon_Rekognition', 'insightface_buffalo', 'insightface_resnet100', 'GhostFaceNet_W1.3_S1_ArcFace', 'GhostFaceNetV1-0.5-2_A', 'ArcFace', 'Facenet', 'Facenet512', 'DeepFace']
        # model_order = ['Amazon_Rekognition', 'insightface_resnet100', 'GhostFaceNet_W1.3_S1_ArcFace', 'GhostFaceNetV1-0.5-2_A', 'ArcFace', 'Facenet', 'Facenet512', 'DeepFace']
        ## we want to divide the mean value of the metric for Amazon_Rekognition by 100 and the variance by 10000, as the values returned by Amazon_Rekognition are in percentage.
        metric = 'FMR' if args.metric_fmr else 'FNMR'
        df.loc[df['Face_Recognition_Model'] == 'Amazon_Rekognition', f'{metric}_Mean'] = df.loc[df['Face_Recognition_Model'] == 'Amazon_Rekognition', f'{metric}_Mean'] / 100
        df.loc[df['Face_Recognition_Model'] == 'Amazon_Rekognition', f'{metric}_Var'] = df.loc[df['Face_Recognition_Model'] == 'Amazon_Rekognition', f'{metric}_Var'] / 10000

        # Ensure that your DataFrame's categories are ordered correctly
        df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)
        df['Face_Recognition_Model'] = pd.Categorical(df['Face_Recognition_Model'], categories=model_order, ordered=True)

        # Sort the DataFrame by Group then by Model
        df_sorted = df.sort_values(['Group', 'Face_Recognition_Model'])
        df_sorted['Face_Recognition_Model'] = df_sorted['Face_Recognition_Model'].replace({'insightface_resnet100': 'Insightface', 'GhostFaceNet_W1.3_S1_ArcFace': 'GhostFaceNet_W1', 'GhostFaceNetV1-0.5-2_A': 'GhostFaceNet_V1', 'Amazon_Rekognition': 'Amazon Rekognition'})
        ## change the name of the groups. Remove the underscores from the name and capitalize the first letter of each word. Also remove the word 'celebs' from the name.
        df_sorted['Group'] = df_sorted['Group'].apply(lambda x: ' '.join(word.capitalize() for word in x.replace('_', ' ').split() if 'celebs' not in word.lower()))
        if plot_for_paper:
            # Filter out 'insightface_buffalo' and update the 'Face_Recognition_Model' categories
            df_sorted = df_sorted[df_sorted['Face_Recognition_Model'] != 'insightface_buffalo']
            df_sorted['Face_Recognition_Model'] = df_sorted['Face_Recognition_Model'].cat.remove_unused_categories()

        # Step 2: Pivot the DataFrame to prepare it for a stacked bar chart
        pivot_df = df_sorted.pivot(columns='Group', index='Face_Recognition_Model', values=f'{metric}_Mean')

        if plot_matplotlib:
            # Plot the stacked bar chart and capture the returned container objects
            containers = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7), rot=70).containers
        else:
            plt.figure(figsize=(25, 5))
            # Use Seaborn to create a bar chart. Note: This will not be stacked but can be grouped.
            ax = sns.barplot(data=df_sorted, x='Face_Recognition_Model', y=f'{metric}_Mean', hue='Group', dodge=True)

        
        if plot_matplotlib:
            if args.metric_fnmr:        ## for the FNMR, the bars are large and we can add the numbers to the plot.     
                # Iterate through each container (each corresponding to a model)
                for container in containers:
                    cumulative_heights = [0] * len(container)  # Initialize with zeros for each group
                    # For each bar (each group for a model) in the container
                    for bar, cumulative_height in zip(container, cumulative_heights):
                        # Get the height (value) of the bar
                        height = bar.get_height()
                        # Update the cumulative height for the position
                        cumulative_heights[cumulative_heights.index(cumulative_height)] += height
                        # The y position to place the text is now at the bottom of the current bar
                        y_pos = bar.get_y() + cumulative_height  # Adjusted to place text at the bottom

                        # Format the number with 2 decimal places
                        label = f"{height:.2f}"

                        # Add the text annotation inside the bar, adjusted to place at the bottom
                        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, label, ha='center', va='bottom', color='black', fontweight='bold')
        else:
            ## this is grouped bar plot and we can add numbers for both FMR and FNMR.
            for p in ax.patches:
                # Calculate the appropriate location for the text annotation
                # This places the text at the top of the bar, with a slight offset to avoid touching the bar itself.
                height = p.get_height()
                if height > 0:
                    ax.annotate(format(p.get_height(), '.2f'),  # Format the value
                                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of text
                                ha = 'center', va = 'center',  # Center alignment
                                xytext = (0, 5),  # Offset for text (makes it appear above the bar)
                                textcoords = 'offset points', color='black', fontweight='bold', fontsize=10)

        if not plot_for_paper:
            model_to_highlight = 'Facenet512'        ## do not highlight anymore
            model_index = pivot_df.index.tolist().index(model_to_highlight)  # Find the index of the group in the pivot

        rotation_angle = 0  # The angle of label rotation

        if args.metric_fmr:
            if not plot_for_paper:
                ellipse = mpatches.Ellipse((model_index, -0.55), width=1, height=0.7, angle=rotation_angle, color='red', fill=False, transform=plt.gca().transData, clip_on=False)
            plt.title('False-Match Rate For Different Face Recognition Models' + ' (Lower is better)', fontsize=22)
            plt.ylabel('False-Match Rate', fontsize=22)
        elif args.metric_fnmr:
            if not plot_for_paper:
                ellipse = mpatches.Ellipse((model_index, -0.75), width=1.4, height=0.6, angle=rotation_angle, color='red', fill=False, transform=plt.gca().transData, clip_on=False)
            plt.title('True-Match Rate For Different Face Recognition Models' + ' (Higher is better)', fontsize=22)
            plt.ylabel('True-Match Rate', fontsize=22)
        if not plot_for_paper:
            plt.gca().add_patch(ellipse)

        if not plot_for_paper:
            plt.text(model_index, -0.3, '(previous results)', color='red', ha='center', transform=plt.gca().get_xaxis_transform())

        plt.xlabel('Face Recognition Models', fontsize=24, labelpad=16)
        plt.xticks(rotation=rotation_angle, fontsize=20)
        # plt.legend(title='Demographic Groups', bbox_to_anchor=(1.05, 1), loc='upper left')        
        legend = plt.legend(title='Demographic Groups', bbox_to_anchor=(0.4, 1), loc='upper center', ncol=2, fontsize=16)
        legend.get_title().set_fontsize('16')
        plt.tight_layout()

        if plot_for_paper:  extension = 'pdf'
        else:              extension = 'png'
        plt.savefig(f"face_recognition_bias_scores/stacked_bar_chart_{'fmr' if args.metric_fmr else 'fnmr'}.{extension}", bbox_inches='tight')
        return
    
    for race in race_groups:
        for gender in gender_groups:
            ## get the celebs in each combination of race and gender
            group_name = f"male_{race}_celebs" if gender == 'male' else f"female_{race}_celebs"
            celebs_in_this_group = locals()[group_name]
            if args.metric_fmr:     ## here we want the different celebs, therefore the combination of celebs in a pair has to be different.
                all_unique_pairs_of_celebs_to_compute_similarity = list(itertools.combinations(celebs_in_this_group, 2))
                assert len(all_unique_pairs_of_celebs_to_compute_similarity) == len(celebs_in_this_group) * (len(celebs_in_this_group) - 1) // 2
            elif args.metric_fnmr:  ## here we just want to make pairs of the same celeb, so that we can compute the false non-match rate.
                all_unique_pairs_of_celebs_to_compute_similarity = [(celeb_here, celeb_here) for celeb_here in celebs_in_this_group]
                assert len(all_unique_pairs_of_celebs_to_compute_similarity) == len(celebs_in_this_group)
            else:  ## this should not happen
                raise Exception('Please specify either --metric_fmr or --metric_fnmr')

            if not parallelize_celebs:
                similarity_scores_for_different_celebs = {}
                for celeb_here, different_celeb in all_unique_pairs_of_celebs_to_compute_similarity:
                    if args.metric_fmr:
                        if celeb_here not in similarity_scores_for_different_celebs:
                            similarity_scores_for_different_celebs[celeb_here] = {}
                        assert celeb_here != different_celeb
                        if different_celeb not in similarity_scores_for_different_celebs:
                            similarity_scores_for_different_celebs[different_celeb] = {}
                        similarity_scores_for_different_celebs[celeb_here][different_celeb] = helper_function_test_face_recognition_models_for_bias(args, celeb_here, different_celeb)
                        similarity_scores_for_different_celebs[different_celeb][celeb_here] = similarity_scores_for_different_celebs[celeb_here][different_celeb]       ## also copy these scores to the other side
                    elif args.metric_fnmr:
                        assert celeb_here == different_celeb
                        similarity_scores_for_different_celebs[celeb_here] = helper_function_test_face_recognition_models_for_bias(args, celeb_here, different_celeb)
                    print(f"celeb {celeb_here} and celeb {different_celeb} done")
            
            else:   ## here we will parallelize the computation of similarity scores for each pair of celebs.
                if parallelize_celebs == True:
                    manager = Manager()
                    similarity_scores_for_different_celebs = manager.dict()
                    for celeb_here in celebs_in_this_group:
                        if args.metric_fmr:
                            similarity_scores_for_different_celebs[celeb_here] = manager.dict()
                            for different_celeb in celebs_in_this_group:
                                similarity_scores_for_different_celebs[celeb_here][different_celeb] = manager.list()
                        elif args.metric_fnmr:
                            similarity_scores_for_different_celebs[celeb_here] = manager.list()
                    
                    start_time = time.time()
                    print("Starting parallelization")
                    pool = Pool(processes=min(get_slurm_cpus(), len(all_unique_pairs_of_celebs_to_compute_similarity)))
                    for celeb_here, different_celeb in all_unique_pairs_of_celebs_to_compute_similarity:
                        if args.metric_fmr:
                            assert celeb_here != different_celeb
                        elif args.metric_fnmr:
                            assert celeb_here == different_celeb
                        pool.apply_async(worker_test_bias, args=(args, celeb_here, different_celeb, similarity_scores_for_different_celebs, ), error_callback=error_callback_test_bias)
                    pool.close()
                    pool.join()
                    print("Time taken for parallel computation", time.time() - start_time)
                    
                    def convert_to_regular_dict(proxy):
                        """
                        Recursively convert Proxy objects to their corresponding
                        regular Python data structures.
                        """
                        if isinstance(proxy, Manager().dict().__class__):
                            return {k: convert_to_regular_dict(v) for k, v in proxy.items()}
                        elif isinstance(proxy, Manager().list().__class__):
                            return [convert_to_regular_dict(v) for v in proxy]
                        else:
                            return proxy
                        
                    similarity_scores_for_different_celebs = convert_to_regular_dict(similarity_scores_for_different_celebs)

            ## save the similarity scores to a file
            args.metric_face_recog = 'FMR' if args.metric_fmr else 'FNMR'
            with open(f"face_recognition_bias_scores/{args.face_recognition_model}/similarity_scores_for_different_celebs_{group_name}_celebs_{args.face_recognition_model}_{args.metric_face_recog}.json", "a") as f:
                json.dump(similarity_scores_for_different_celebs, f)

            ## compute the avg of entire similarity scores. No this is not what we want. We need to divide the value of similarity by 100 as this is the value returned by Amazon Rekognition, which gives the similarity as a percentage. So the mean will also be divided by 100, and the variance will be divided by 100^2.
            avg = []
            if args.metric_fmr:
                for celeb_here in similarity_scores_for_different_celebs:
                    for different_celeb in similarity_scores_for_different_celebs[celeb_here]:
                        avg.extend(similarity_scores_for_different_celebs[celeb_here][different_celeb])
            elif args.metric_fnmr:
                for celeb_here in similarity_scores_for_different_celebs:
                    avg.extend(similarity_scores_for_different_celebs[celeb_here])
            print(f"Avg. {args.metric_face_recog} for {group_name} group", np.mean(avg))
            print(f"Variance in FMR for {group_name} group", np.var(avg))
            if args.metric_fmr:
                filename = "face_recognition_bias_avg_scores_fmr.csv"
            elif args.metric_fnmr:
                filename = "face_recognition_bias_avg_scores_fnmr.csv"
            with open(filename, "a") as f:
                f.write(f"{args.face_recognition_model}|{group_name}|{np.mean(avg)}|{np.var(avg)}\n")


def stats_from_similarity_data(args, df=None, x_axis_to_sort=None, metric=None, penalize_steps=1):
    if df is None:
        if args.use_selected_images_for_matching:
            args.top_k_training_images = "all_selected_images"
        elif args.use_filtered_laion_images_and_selected_images_for_matching:
            args.top_k_training_images = "filtered_laion_images_and_selected_images"
        else:
            if args.select_training_images_based_on_similarity_to_generated_images:
                args.top_k_training_images = "top_10_similar_to_each_generated_image"
            elif args.select_training_images_based_on_similarity_to_generated_images_on_average:
                args.top_k_training_images = "top_10_similar_to_average_generated_images"
            else:
                args.top_k_training_images = "all_training_images"

        if args.image_generation_prompt_id != -99:
            args.image_generation_prompt = set_of_prompts_human_face[args.image_generation_prompt_id][0]  
            df = pd.read_csv(f"average_cosine_similarity_closeup_training_and_single_face_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}.csv", sep='|', header=0)
            # df = df.sort_values(by=['total_matching_captions'], ascending=True)
            df = df.sort_values(by=['effective_num_face_counts'], ascending=True)
        else:
            df = pd.read_csv(f"combined_df_{args.set_of_people}_{args.top_k_training_images}_{args.face_recognition_model}.csv", sep='|', header=0)
            # df = df.sort_values(by=['total_matching_captions'], ascending=True)
            df = df.sort_values(by=['effective_num_face_counts'], ascending=True)
            ## change the column average_cosine_similarity mean to average_cosine_similarity and then proceed normally.
            df = df.rename(columns={'average_cosine_similarity mean': 'average_cosine_similarity'})
    # print(df[['celeb', 'total_matching_captions', 'average_cosine_similarity']])
    
    args.fit_regression_model = False
    args.use_change_detection_old = False
    args.use_change_detection = True
    
    import numpy as np
    if args.fit_regression_model:
        from scipy import stats
        import statsmodels.api as sm
        import matplotlib.pyplot as plt
        function = 'polynomial'
        
        if function == 'linear':
            ## Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['total_matching_captions'], df['average_cosine_similarity'])
            print(f"r_value is {r_value}, p_value is {p_value}, std_err is {std_err}")
        
        elif function == 'log_regression':
            ## Log regression
            ## replace the 0 values in total_matching_captions with 1, as log(0) is undefined.
            temp_df = copy.deepcopy(df)
            temp_df['total_matching_captions'] = temp_df['total_matching_captions'].replace(0, 1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(temp_df['total_matching_captions']), temp_df['average_cosine_similarity'])
            print(f"r_value is {r_value}, p_value is {p_value}, std_err is {std_err}")
        
        elif function == 'quadratic':
            ## Quadratic regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['total_matching_captions'], df['average_cosine_similarity']**2)
            print(f"r_value is {r_value}, p_value is {p_value}, std_err is {std_err}")

        elif function == 'robust':    
            ## robust regression
            X = sm.add_constant(df['total_matching_captions'])
            model = sm.RLM(df['average_cosine_similarity'], X, M=sm.robust.norms.HuberT())
            results = model.fit()
            print(results.summary())
            slope = results.params['total_matching_captions']
            intercept = results.params['const']
            robust_residuals = results.resid

        elif function == 'polynomial':
            x = df['total_matching_captions'].to_numpy()
            y = df['average_cosine_similarity'].to_numpy()
            # if args.set_of_people == "celebrity":
            #     degree = 11
            # elif args.set_of_people == "politicians":
            #     degree = 12
            degree = args.polynomial_degree
            
            use_numpy = True

            if use_numpy:
                coeffs = np.polyfit(x, y, degree)
                p = np.poly1d(coeffs)
                y_pred = p(x)
            else:
                ## Generate polynomial features
                X = np.column_stack([x**i for i in range(degree + 1)])
                X = sm.add_constant(X)  # Add a constant column to include the intercept in the model
                # Fit the model
                model = sm.OLS(y, X).fit()
                # Get p-values for the coefficients
                p_values = model.pvalues
                # You can print or return the p-values as needed
                print(p_values)
                y_pred = model.predict(X)
                    

        if function != 'polynomial':
            residuals = df['average_cosine_similarity'] - (slope * df['total_matching_captions'] + intercept)
            assert len(residuals) == len(df)
        elif function == 'robust':
            assert robust_residuals.eq(residuals).all()
        elif function == 'polynomial':
            residuals = y - y_pred
            assert len(residuals) == len(df)

        # print("Residuals are", len(residuals))
        # print("Mean of residuals is", np.mean(residuals))
        
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
        plt.title('Histogram of Residuals from Robust Regression')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.savefig(f"Histogram_of_Residuals_from_Robust_Regression_{args.image_generation_prompt_id}_{args.face_recognition_model}_{args.set_of_people}.png")
        
        # ## Plotting the Q-Q plot for residuals
        # sm.qqplot(np.array(residuals), line='45', fit=True)
        # plt.title('Q-Q Plot of Residuals')
        # plt.savefig(f"Q-Q_Plot_of_Residuals_{args.image_generation_prompt}_{args.face_recognition_model}_{args.set_of_people}.png")
        
        if use_numpy:
            ## Shapiro-Wilk test
            shapiro_test = stats.shapiro(residuals)
            print(f"Shapiro-Wilk test:\nStatistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}")

            ## Anderson-Darling test
            anderson_test = stats.anderson(residuals, dist='norm')
            print(f"Anderson-Darling test:\nStatistic: {anderson_test.statistic}, Significance Levels: {anderson_test.significance_level}, Critical Values: {anderson_test.critical_values}")

            # Kolmogorov-Smirnov test
            ks_test = stats.kstest(residuals, 'norm')
            print(f"Kolmogorov-Smirnov test:\nStatistic: {ks_test.statistic}, p-value: {ks_test.pvalue}")

        if function == 'linear':
            pred_0 = slope * 0 + intercept
        elif function == 'log_regression':
            pred_0 = slope * np.log(0) + intercept
        elif function == 'quadratic':
            pred_0 = slope * 0**2 + intercept
        elif function == 'robust':
            pred_0 = slope * 0 + intercept
        elif function == 'polynomial':
            ## get the predict of the model at x = 0
            if use_numpy:
                pred_0 = p(0)
            else:
                get_variable = [1]
                for i in range(1, degree + 1):
                    get_variable.append(0**i)
                print(f"Value of get_variable is {get_variable}")
                pred_0 = model.predict(get_variable)
            
        print(f"Value of f(0) is {pred_0}")
        ## get the errors for the points with x = 0 
        points_with_x_0 = df[df['total_matching_captions'] == 0]
        errors = points_with_x_0['average_cosine_similarity'] - pred_0
        std_deviation = np.std(errors)
        # std_deviation = np.std(residuals)       ## this is the standard deviation of the residuals
        y_star = pred_0 + 1.65 * std_deviation
        print(f"Value of y* is {y_star}", " value of f(0): ", pred_0, " std_deviation: ", std_deviation)
        x_value = 0
        if function in ['linear', 'log_regression', 'quadratic', 'robust']:
            while slope * x_value + intercept < y_star:
                x_value += 1
        elif function == 'polynomial':
            if use_numpy:
                while p(x_value) < y_star:
                    x_value += 1
                    if x_value > 20000:
                        print("The value of x_value is becoming very large, so breaking the loop.")
                        break
            else:
                while model.predict([1] + [x_value**i for i in range(1, degree + 1)]) < y_star:
                    x_value += 1
                    if x_value > 20000:
                        print("The value of x_value is becoming very large, so breaking the loop.")
                        break
        print(f"Value of X where the value of y becomes larger than y* on the fitted function is {x_value}")

    elif args.use_change_detection_old:
        from scipy.stats import mstats
        from sklearn.linear_model import TheilSenRegressor

        x = df['total_matching_captions'].to_numpy()
        y = df['average_cosine_similarity'].to_numpy()
        # Perform Mann-Kendall Trend Test to check for the presence of a trend
        trend, p_value = mstats.kendalltau(x, y)
        
        print("Trend: ", "Increasing" if trend > 0 else "Decreasing", "| p-value:", p_value)
        
        if p_value < 0.05:
            print("The trend is statistically significant.")
        else:
            print("The trend is not statistically significant.")
        
        # Use Theil-Sen estimator to find segments with significant slope changes
        model = TheilSenRegressor().fit(x.reshape(-1, 1), y)
        slope = model.coef_[0]        
        print("Estimated slope: ", slope)
        
        y_pred = model.predict(x.reshape(-1, 1))
        # Calculate residuals
        residuals = y - y_pred

        # Calculate standard deviation of residuals
        std_residuals = np.std(residuals)

        # Identify points where the absolute residual is greater than 2 times the standard deviation
        threshold = 0.15 * std_residuals
        # print(residuals, threshold)
        change_points = np.where(abs(residuals) > threshold)[0]

        print("Change points: ", change_points, type(change_points))
        ## print df for the change points
        print(df.iloc[change_points][['celeb', 'total_matching_captions', 'average_cosine_similarity']])
        
    elif args.use_change_detection:
        import ruptures as rpt
        # print(df.columns)
        if x_axis_to_sort is None:
            x = df['effective_num_face_counts'].to_numpy()
            y = df['average_cosine_similarity'].to_numpy()
        else:
            x = df[x_axis_to_sort].to_numpy()
            y = df[metric].to_numpy()

        model = "l2"  # Use L2 norm for a change in mean detection -- we should use l2 as it will minimize the error as our y values are between 0 and 1.
        algo = rpt.Pelt(model=model, min_size=1, jump=1).fit(y)
        # Find the change point
        change_points = algo.predict(pen=penalize_steps)
        change_points = np.array(change_points)
        if change_points[-1] == len(y):
            change_points = change_points[:-1]
        print("Change points: ", change_points, type(change_points))
        
        if len(change_points) > 1:
            ## only consider the first change point
            change_points = change_points[:1]
        
        ## print df for the change points
        if x_axis_to_sort is None:
            print(df.iloc[change_points][['celeb', 'total_matching_captions', 'effective_num_face_counts', 'average_cosine_similarity']])
        else:
            print(df.iloc[change_points][['celeb', 'total_matching_captions', x_axis_to_sort, metric]])
        
        ## Prepare to print the y values in regions between change points
        regions_y_values = []
        change_points = np.insert(change_points, 0, 0)
        change_points = np.append(change_points, len(y))
        for start, end in zip(change_points[:-1], change_points[1:]):
            region_y = y[start:end]
            regions_y_values.append(region_y.mean())
            print(f"Region {start}-{end-1}: {region_y.mean()}")

        print(regions_y_values, change_points)
        
        return change_points, regions_y_values
    
        # Try using Binary Segmentation
        # algo_binseg = rpt.Binseg(model="l2").fit(y)
        # result_binseg = algo_binseg.predict(n_bkps=1)  # n_bkps=1 for 1 change, adjust as necessary
        # change_points = np.array(result_binseg) - 1
        # print("Change points using Binary Segmentation: ", change_points, type(change_points))
        # ## print df for the change points
        # print(df.iloc[change_points][['celeb', 'total_matching_captions', 'average_cosine_similarity']])
        

def save_this_subplot(args, df, name, metric):
    from scipy import stats
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create a new figure and axes
    new_fig, new_ax = plt.subplots(figsize=(70, 13))  # Adjust the figure size as needed
    sns.set_theme(style="whitegrid", context="paper", palette="muted")
        
    xlabel = {
        "celebrity": "Celebrities with increasing Image Counts in the Training Data",
        "politicians": "Politicians with increasing Image Counts in the Training Data",
        "cub_birds": "Birds with increasing Image Counts in the Training Data",
        "artists_group1": "Art images and their Counts in the Training Data",
        "artists_group2": "Art images and their Counts in the Training Data",
        "celebrity_fid_distance": "Celebrities with increasing Caption Counts in the Training Data",
    }.get(args.set_of_people, "")
    
    new_ax.set_xlabel(xlabel, fontsize=50)
    
    title_label_map = {
        # 'average_cosine_similarity': ("Average Cosine Similarity between Training and Generated Images", "Average cosine similarity"),
        'average_cosine_similarity': ("Average Imitation Score of the Generated Images", "Imitation Score"),
        'average_kl_divergence': ("Average KL Divergence between Training and Generated Images", "Average KL Divergence"),
        'average_class_acc': ("Average Classification Accuracy", "Average Classification Acc"),
        'average_top5_class_acc': ("Average Classification Top-5 Accuracy", "Average Classification Top-5 Acc"),
        'average_top10_class_acc': ("Average Classification Top-10 Accuracy", "Average Classification Top-10 Acc"), 
        'Mean_top_5_cosine_similarity': ("Mean of Top 5 Cosine Similarity between Training and Generated Images", "Mean of Top 5 Cosine Similarity"),
        'clip_score': ("CLIP Score of the Generated Images", "CLIP Score"),
        'fid_distance': ("FID between the Training and Generated Images", "Fréchet Inception Distance")
    }

    title, ylabel = title_label_map.get(metric, ("", ""))
    new_ax.set_title(title, fontsize=50)
    new_ax.set_ylabel(ylabel, fontsize=50)
    sns.lineplot(x=df['celeb'], y=df[f'{metric} mean'], marker='o', label=ylabel, ax=new_ax, linewidth=3, color='blue')
    
    # Filling between lines for variance
    upper_bound = df[f'{metric} mean'] + np.sqrt(df[f'{metric} var'])
    lower_bound = df[f'{metric} mean'] - np.sqrt(df[f'{metric} var'])
    new_ax.fill_between(df['celeb'], lower_bound, upper_bound, color='gray', alpha=0.3)

    print("mean value of variance and std deviation", np.mean(df[f'{metric} var']), np.std(df[f'{metric} var']))

    # ## This is using matplotlib for regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(df['total_matching_captions'])), df[f'{metric} mean'])
            
    if args.set_of_people in ["celebrity", "politicians", "artists_group1", "artists_group2"]:
        ## we will also add the change detection point to the plot
        x_axis_to_sort = 'effective_num_face_counts' if args.set_of_people in ['celebrity', 'politicians'] else 'total_laion_paintings_this_artist' if args.set_of_people in ["artists_group1", "artists_group2"] else 'total_matching_captions'
        ## assert that df is already sorted by x_axis_to_sort, before it is sent to the function. 
        assert df[x_axis_to_sort].is_monotonic_increasing
        if args.use_only_isotonic_regression:
            ## use isotonic regression
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression()
            x = np.arange(len(df['celeb']))
            y = df[f'{metric} mean'].to_numpy()
            y_ = ir.fit_transform(x, y)
            new_ax.plot(x, y_, marker='o', label='Isotonic Regression', linewidth=3, color='green')
            print("Isotonic Regression done")
        
        else:
            change_points, regions_y_values = stats_from_similarity_data(args, df, x_axis_to_sort, f'{metric} mean', penalize_steps=100000 if args.set_of_people == 'celebrity_fid_distance' else 0.9)
            assert len(regions_y_values) + 1 == len(change_points)
            for i, change_point in enumerate(change_points):
                if change_point == 0:
                    continue
                if change_point >= len(df['celeb']):
                    break

                ## get the value on the x-axis for the change point
                change_point_x_axis_value = round(df.iloc[change_point][x_axis_to_sort])
                quantity = "faces" if args.set_of_people in ['celebrity', 'politicians'] else "images" if args.set_of_people in ["artists_group1", "artists_group2"] else "captions"
                new_ax.axvline(x=change_point, color='red', linestyle='-', linewidth=5, ymax=0.75 if args.set_of_people == 'celebrity_fid_distance' else 0.85)
                y_text_place = 0.9 * df[f'{metric} mean'].max()
                new_ax.text(change_point, y_text_place, f"Imitation Threshold: {change_point_x_axis_value} {quantity}", fontsize=50, horizontalalignment='center')
                
                ## add a horizontal line at the two region y values from previous change point to the current change point and from the current change point to the next change point.
                new_ax.axhline(y=regions_y_values[i-1], color='navy', linestyle='-.', linewidth=3, xmin=change_points[i-1] / len(df['celeb']), xmax=(change_point + 1)/ len(df['celeb']), label='Mean Similarity Before Change Point', marker='s', markersize=10)
                new_ax.axhline(y=regions_y_values[i], color='forestgreen', linestyle='--', linewidth=3, xmin=(change_point + 1)/ len(df['celeb']), xmax=change_points[i+1] / len(df['celeb']), label='Mean Similarity After Change Point', marker='^', markersize=10)
                print("added a vertical line at change point", change_point, "with y value", regions_y_values[i])

                ## outliers are the points before the change point whose metric value is greater than twice the mean valuye of the metric value in the region before the change point.
                outliers = df.iloc[:change_point][df[f'{metric} mean'] > 4 * regions_y_values[i-1]]
                print("Outliers are", outliers[['celeb', 'total_matching_captions', x_axis_to_sort, f'{metric} mean']])
            
            if args.use_isotonic_and_change_detection:
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression()
                x = np.arange(len(df['celeb']))
                y = df[f'{metric} mean'].to_numpy()
                y_ = ir.fit_transform(x, y)
                new_ax.plot(x, y_, marker='D', label='Isotonic Regression', linewidth=4, color='magenta')
                print("Isotonic Regression done")
                
    # else:
    #     regression_line = slope * np.arange(len(df['total_matching_captions'])) + intercept
    #     new_ax.plot(df['celeb'], regression_line, color='red', label='Best Fit Linear Regression Line', linewidth=3)
        
    # Use Seaborn's regplot for regression line, plotting against index for x-values
    # sns.regplot(x=np.arange(len(df['total_matching_captions'])), y=df[f'{metric} mean'], ax=new_ax, color='red', line_kws={"label":"Best Fit Linear Regression Line", "linewidth":3})
    
    # ## make a black dashed line at the value of good_image_threshold_value and make a gray translucent band around it of width good_image_threshold_value_std
    # if not args.set_of_people == 'cub_birds':
    #     index_celeb_with_no_captions = np.where(df['total_matching_captions'].values < 1)[0][-1] + 1
    #     good_image_threshold_value = df[f'{metric} mean'].values[:index_celeb_with_no_captions].mean()
    #     good_image_threshold_value_std = df[f'{metric} mean'].values[:index_celeb_with_no_captions].std()
    #     print("good_image_threshold_value and its std: ", good_image_threshold_value, good_image_threshold_value_std)
    #     new_ax.axhline(y=good_image_threshold_value, color='black', linestyle='--', label='Baseline Value', linewidth=3)
    #     new_ax.fill_between(df['celeb'], good_image_threshold_value - good_image_threshold_value_std, good_image_threshold_value + good_image_threshold_value_std, color='gray', alpha=0.3)
        
    if args.set_of_people == "celebrity_fid_distance":
        new_ax.legend(loc='upper right', fontsize=40)
    else:
        new_ax.legend(loc='upper left', fontsize=40)
    
    # Adjusting tick parameters and labels
    new_ax.tick_params(axis='y', labelsize=60)
    new_ax.tick_params(labelright=True)  # This enables the right y-axis labels (mirrored)
    new_ax.yaxis.set_ticks_position('both')  # Show ticks on both sides of the y-axis
    new_fig.subplots_adjust(right=0.8)
    
    if args.set_of_people in ['celebrity', 'politicians', "artists_group1", "artists_group2"]:
        ## get the max value of the metric and set the y-ticks accordingly
        y_max_value = df[f'{metric} mean'].max()
        ## set the y-ticks to be from 0 to y_max_value with a step of 0.1
        yticks = np.arange(0, y_max_value + 0.1, 0.1)
        # yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        new_ax.set_yticks(yticks)
    
    # Adding vertical lines for each data point
    y_min_value = df[f'{metric} mean'].min()
    for x, y in zip(df['celeb'].values, df[f'{metric} mean'].values):
        new_ax.vlines(x, ymin=y_min_value, ymax=y, colors='grey', linestyles='dashed', alpha=0.8)  # Adjust 'ymin' as needed, 'ymax' is set to the y value
    
    lower_indexes = np.arange(len(df['celeb'].values))
    new_ax.set_xticks(lower_indexes)
    new_ax.set_xticklabels(df['celeb_and_caption_count'].values[lower_indexes], rotation=90)
    new_ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    ## shade the part when effective_num_face_counts is 0 with light gray color
    if args.set_of_people in ['celebrity', 'politicians']:
        index_celeb_with_no_faces = np.where(df['effective_num_face_counts'].values < 1)[0][-1]
        new_ax.axvspan(0, index_celeb_with_no_faces, color='gray', alpha=0.3)
    elif args.set_of_people in ["artists_group1", "artists_group2"]:
        index_celeb_with_no_faces = np.where(df['total_laion_paintings_this_artist'].values < 1)[0][-1]
        new_ax.axvspan(0, index_celeb_with_no_faces, color='gray', alpha=0.3)

    total_data_points = len(df['celeb'].values)
    gap_fraction = 0.005
    gap = total_data_points * gap_fraction
    new_ax.set_xlim(-gap, len(df['celeb'].values))
    plt.tight_layout()

    saving_directory = "similarity_plots"
    if args.stable_diffusion_model == "1":
        name = "SD1.1" + "_" + name
        saving_directory = saving_directory + "_SD1.1"
    if args.stable_diffusion_model == "5":
        name = "SD1.5" + "_" + name
        saving_directory = saving_directory + "_SD1.5"
    elif args.stable_diffusion_model == "v2":
        name = "SD2.1" + "_" + name
        saving_directory = saving_directory + "_SD2.1"
    
    if args.consider_name_aliases:
        saving_directory = saving_directory + "_consider_name_aliases"
    
    os.makedirs(saving_directory, exist_ok=True)
          
    args.set_of_people = args.set_of_people + "_threshold_" + str(args.threshold_for_max_f1_score) if args.threshold_for_max_f1_score else args.set_of_people
    
    save_plot_path = f"{saving_directory}/{name}_{args.top_k_training_images}_training_images_{args.set_of_people}.pdf"
    new_fig.savefig(save_plot_path)
    print("Saved the plot at", save_plot_path)

    save_plot_path_png = f"{saving_directory}/{name}_{args.top_k_training_images}_training_images_{args.set_of_people}.png"
    new_fig.savefig(save_plot_path_png)


def plot_similarity_training_and_generated_images(args, sort_according_to_quality=False, sort_according_to_effective_pixels=False, sort_according_to_face_count=False, sort_according_to_total_face_pixels=False, separate_plot_each_race=None):
    ## Here we will read data from average_cosine_similarity_closeup_training_and_single_face_generated_images.csv and make a plot of the average cosine similarity between the training and generated images for each celeb. celeb|num_close_up_training_images|num_single_face_generated_images|total_matching_captions|average_cosine_similarity. # We will also add the number of single face_generated images in the second plot.
    global celebrity_list, set_of_prompts_human_face
    args.threshold_for_max_f1_score = None
    # base_embeddings_directory = f"{args.face_recognition_model}_embeddings_close_up_{args.stable_diffusion_model}" if args.set_of_people == "celebrity" else f"{args.face_recognition_model}_embeddings_close_up_politicians_{args.stable_diffusion_model}" if args.set_of_people == "politicians" else None
    
    if args.use_selected_images_for_matching:
        args.top_k_training_images = "all_selected_images"
    elif args.use_filtered_laion_images_and_selected_images_for_matching:
        args.top_k_training_images = "filtered_laion_images_and_selected_images"
        if args.same_face_threshold == "lower":
            args.threshold_for_max_f1_score = 0.45668589190104747
        elif args.same_face_threshold == "higher":
            args.threshold_for_max_f1_score = 0.5576227593909431
        else:
            raise NotImplementedError
    else:
        if args.select_training_images_based_on_similarity_to_generated_images:
            args.top_k_training_images = "top_10_similar_to_each_generated_image"
        elif args.select_training_images_based_on_similarity_to_generated_images_on_average:
            args.top_k_training_images = "top_10_similar_to_average_generated_images"
        else:
            args.top_k_training_images = "all_training_images"
        
    if separate_plot_each_race:     ## if this value is not None, then we will only plot the similarity for the race and gender of these celebs. We will have 6 groups in total: male and female for gender, and white, black, and brown for race. 
        group_df = pd.read_csv(f"celebrity_predicted_race_and_gender2.csv", sep="|")   # celeb|gender|race
        ## assert that the values of gender are either Man or Woman
        assert set(group_df['gender'].unique()) == {'Man', 'Woman'}
        assert set(group_df['race'].unique()) == {'asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic'}
        ## change all the white to White, black to Black, and asian, indian, middle eastern, latino hispanic to Brown
        group_df['race'] = group_df['race'].replace(['white', 'black', 'asian', 'indian', 'middle eastern', 'latino hispanic'], ['White', 'Black', 'Brown', 'Brown', 'Brown', 'Brown'])
        assert set(group_df['race'].unique()) == {'White', 'Black', 'Brown'}
        gender_here = separate_plot_each_race[0]
        race_here = separate_plot_each_race[1]
        ## get the celebs in this group of gender and race
        celebs_in_this_group = group_df[(group_df['gender'] == gender_here) & (group_df['race'] == race_here)]
        assert len(celebs_in_this_group) >= 9       ## I think the smallest group is black women of length 9. 
        celebrity_list = celebs_in_this_group['celeb'].values
        print(f"Number of celebs in the group of {gender_here} and {race_here} is {len(celebrity_list)}")
        return
    else:
        if args.set_of_people == "celebrity":
            celebrity_list = celebrity_list + celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities
            columns_to_remain_same = ['celeb', 'num_single_person_training_images_plus_internet_images', 'total_matching_captions', 'num_downloaded_training_images', 'celeb_and_caption_count', 'effective_num_face_counts']
        elif args.set_of_people == "politicians":
            politicians = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")['Name'].tolist()
            celebrity_list = politicians
            assert len(celebrity_list) == 418
            columns_to_remain_same = ['celeb', 'num_single_person_training_images_plus_internet_images', 'total_matching_captions', 'num_downloaded_training_images', 'celeb_and_caption_count', 'effective_num_face_counts']
        elif args.set_of_people == "cub_birds":
            set_of_prompts = [("a photorealistic photograph of ", 1)]
            args.image_generation_prompt = set_of_prompts[0][0]
            metric_bird = 'average_cosine_similarity'
            # metric_bird = 'average_class_acc'
            file_name = f"/gscratch/h2lab/vsahil/vlm-efficiency/birds_dataset/{metric_bird}_training_and_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_clip_{args.set_of_people}.csv"
            cub_birds = pd.read_csv(file_name, sep="|", header=0)["celeb"].tolist()
            celebrity_list = cub_birds
            columns_to_remain_same = ['celeb', 'total_matching_captions', 'celeb_and_caption_count']
        elif args.set_of_people == "celebrity_clip_score": ## here we are just using the plotting code to plot the clip scores for the celebs.
            celebrity_list = celebrity_list + celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities
            columns_to_remain_same = ['celeb', 'prompt', 'clip_score', 'total_matching_captions', 'celeb_and_caption_count']
        elif args.set_of_people == "celebrity_fid_distance":
            celebrity_list = celebrity_list + celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities
            columns_to_remain_same = ['celeb', 'total_matching_captions', 'celeb_and_caption_count', 'fid_distance']
        elif args.set_of_people == "artists_group1":
            celebrity_list = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")  ## artist_name, count_wikiart_artworks, counts_in_laion2b-en, count_range, count_paintings_in_laion_images, downloaded_images_from_wikiart_website, count_artworks_in_laion_images, count_this_artist_artworks_in_laion_images
            celebrity_list = celebrity_list['artist_name'].tolist()
            columns_to_remain_same = ['celeb', 'total_matching_captions', 'celeb_and_caption_count', 'total_laion_paintings_this_artist']
        elif args.set_of_people == "artists_group2":
            celebrity_list = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")  ## artist_name, count_wikiart_artworks, counts_in_laion2b-en, count_range, count_paintings_in_laion_images, downloaded_images_from_wikiart_website, count_artworks_in_laion_images, count_this_artist_artworks_in_laion_images
            celebrity_list = celebrity_list['artist_name'].tolist()
            columns_to_remain_same = ['celeb', 'total_matching_captions', 'celeb_and_caption_count', 'total_laion_paintings_this_artist']
        else:
            raise ValueError("The set of people is not recognized.")

    cosine_similarity_dfs = []
    
    if args.set_of_people in ["celebrity", "politicians", "celebrity_clip_score", "celebrity_fid_distance"]:
        if args.image_generation_prompt_id != -99:
            set_of_prompts = [set_of_prompts_human_face[args.image_generation_prompt_id]]
        else:
            set_of_prompts = set_of_prompts_human_face      ## in this case all the 5 prompts will be used for plotting. 
    elif args.set_of_people == "cub_birds":
        set_of_prompts_cub_birds = [("a photorealistic photograph of ", 1)]
        set_of_prompts = [set_of_prompts_cub_birds[args.image_generation_prompt_id]]
    elif args.set_of_people in ["artists_group1", "artists_group2"]:
        set_of_prompts_art_styles = [("a painting in the style of ", 1), ("an artwork in the style of ", 1), ("a sketch in the style of ", 1), ("a fine art piece in the style of ", 1),  ("an illustration in the style of ", 1)]
        if args.image_generation_prompt_id != -99:
            set_of_prompts = [set_of_prompts_art_styles[args.image_generation_prompt_id]]
        else:
            set_of_prompts = set_of_prompts_art_styles
    

    for prompt, _ in set_of_prompts:
        args.image_generation_prompt = prompt
        if args.set_of_people in ["celebrity", "politicians"]:
            if args.use_filtered_laion_images_and_selected_images_for_matching:
                added_path = ''
                if args.consider_name_aliases:
                    added_path = "results_with_alias_names/"
                df = pd.read_csv(f"{added_path}average_cosine_similarity_closeup_training_and_single_face_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}_{args.threshold_for_max_f1_score}.csv", sep='|', header=0)
            else:
                df = pd.read_csv(f"average_cosine_similarity_closeup_training_and_single_face_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_{args.face_recognition_model}_{args.set_of_people}.csv", sep='|', header=0)
        elif args.set_of_people == "cub_birds":
            file_name = f"/gscratch/h2lab/vsahil/vlm-efficiency/birds_dataset/{metric_bird}_training_and_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_training_images_using_clip_{args.set_of_people}.csv"
            df = pd.read_csv(file_name, sep='|', header=0)
        elif args.set_of_people == "celebrity_clip_score":
            df = pd.read_csv("../clip_scores_celebrity.csv", sep='|', header=0)
            df = df[df['prompt'] == prompt]
            assert df.shape[0] >= len(celebrity_list)
        elif args.set_of_people == "celebrity_fid_distance":
            df = pd.read_csv("../fid_scores_celebrity.csv", sep='|', header=0)
            df = df[df['prompt'] == prompt]
            ## assert all celebs in celebrity_list are present in df['celeb'] -- this won't be true
            # assert set(celebrity_list).issubset(set(df['celeb'].values.tolist()))
        elif args.set_of_people in ["artists_group1", "artists_group2"]:
            if args.set_of_people == "artists_group1":
                df = pd.read_csv(f"../art_styles/style_similarity_somepalli/average_cosine_similarity_wikiart_images_and_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_using_style_clip_wikiart_artist_group1.csv", sep=",", header=0)  ## artist_name,total_matching_captions,total_laion_paintings_this_artist,average_cosine_similarity
            elif args.set_of_people == "artists_group2":
                df = pd.read_csv(f"../art_styles/style_similarity_somepalli/average_cosine_similarity_wikiart_images_and_generated_images_prompt_{args.image_generation_prompt}_top_{args.top_k_training_images}_using_style_clip_wikiart_artist_group2.csv", sep=",", header=0)  ## artist_name,total_matching_captions,total_laion_paintings_this_artist,average_cosine_similarity
            else:
                raise ValueError("The set of people is not recognized.")
            ## rename artists_name to celeb in df
            df = df.rename(columns={'artist_name': 'celeb'})
            ## sort the df by total_matching_captions (ascending) and then by celeb (ascending)
            df = df.sort_values(by=['total_matching_captions', 'celeb'], ascending=[True, True])
        ## print the df with 0 total_matching_captions
        # zero_caption_df = df[df['total_matching_captions'] == 0]
        # zero_caption_df.to_csv(f"celebs_with_0_captions_{args.image_generation_prompt}_{args.face_recognition_model}.csv", sep='|', index=False)
        # if not separate_plot_each_race:
        #     assert df.shape[0] <= len(celebrity_list)
        # # create a new column in df where we have the count of captions alongside celeb name. both columns already exist in df.
        # df = df.set_index('celeb')
        # df = df.loc[celebrity_list]
        # df = df.reset_index()
   
        df['celeb_and_caption_count'] = df['celeb'] + " (" + df['total_matching_captions'].astype(str) + ")"
        if len(cosine_similarity_dfs) > 0:
            ## here we want to assert that some columns will remain the exact same across these image prompts, after sorting by celeb
            for column in columns_to_remain_same:
                ## sort the last df in cosine_similarity_dfs by celeb and this df by celeb and assert that the columns are the same. The order is decided by the order in celebrity_list
                last_df = cosine_similarity_dfs[-1]                
                ## now compare the columns
                try:
                    assert last_df[column].values.tolist() == df[column].values.tolist(), f"column: {column} is not the same for prompt: {prompt}"
                except AssertionError as e:
                    import ipdb; ipdb.set_trace()
        cosine_similarity_dfs.append(df)
    combined_df = pd.concat(cosine_similarity_dfs)

    common_columns_df = combined_df[columns_to_remain_same].drop_duplicates()
    # Calculate the average and variance for each entity
    if args.set_of_people in ["celebrity", "politicians"]:
        agg_df = combined_df.groupby('celeb').agg({'average_cosine_similarity': ['mean', 'var'], 'Mean_top_5_cosine_similarity': ['mean', 'var'], 'num_single_face_generated_images': ['mean', 'var']}).reset_index()
        agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    elif args.set_of_people == "cub_birds":
        agg_df = combined_df.groupby('celeb').agg({f'{metric_bird}': ['mean', 'var']}).reset_index()
        agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    elif args.set_of_people == "celebrity_clip_score":
        agg_df = combined_df.groupby('celeb').agg({'clip_score': ['mean', 'var']}).reset_index()
        agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    elif args.set_of_people == "celebrity_fid_distance":
        agg_df = combined_df.groupby('celeb').agg({'fid_distance': ['mean', 'var']}).reset_index()
        agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    elif args.set_of_people in ["artists_group1", "artists_group2"]:
        agg_df = combined_df.groupby('celeb').agg({'average_cosine_similarity': ['mean', 'var']}).reset_index()
        agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]

    # Merge the aggregated data with the common columns
    df = pd.merge(common_columns_df, agg_df, on='celeb', how='left')
    
    assert len(df) == len(common_columns_df) == len(agg_df)
    assert len(df.columns) == len(common_columns_df.columns) + len(agg_df.columns) - 1

    if args.set_of_people in ["celebrity", "politicians"]:
        df = df[df['celeb'] != 'Aryan Simhadri']        ## too much noise in this person's image
        df = df[df['celeb']  != 'Félix Tshisekedi']     ## problem due to accents. 
        df = df[df['celeb']  != 'Ahmet Davutoğlu']
        # count_file = pd.read_csv(f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{base_embeddings_directory}/num_face_counts_celebrity.csv", sep='|', header=0)  ## celeb|num_face_counts|num_downloaded_images|matching_captions|effective_num_face_counts|face_area|effective_face_area
        ## add a new column in df "num_face_counts" and fill the values from count_file for the same celeb
        # df = df.set_index('celeb')
        # import ipdb; ipdb.set_trace()
        # df['effective_num_face_counts'] = count_file.set_index('celeb')['effective_num_face_counts']
        # df['total_face_pixels'] = count_file.set_index('celeb')['effective_face_area']
        # df['total_face_pixels'] = df['total_face_pixels'].astype(float)
        # df = df.reset_index()
        ## edit the column "celeb_and_caption_count" to include the effective_num_face_counts as well
        # df['celeb_and_caption_count'] = df['celeb'] + " (" + df['total_matching_captions'].astype(str) + ", " + df['effective_num_face_counts'].astype(str) + ")"
        df['celeb_and_caption_count'] = df['celeb'] + " (" + df['effective_num_face_counts'].astype(str) + ")"
    elif args.set_of_people in ["artists_group1", "artists_group2"]:
        # df['celeb_and_caption_count'] = df['celeb'] + " (" + df['total_matching_captions'].astype(str) + ", " + df['total_laion_paintings_this_artist'].astype(str) + ")"
        df['celeb_and_caption_count'] = df['celeb'] + " (" + df['total_laion_paintings_this_artist'].astype(str) + ")"


    def measure_noise_in_the_similarity_neasurement_between_generated_and_training_images(df, x_axis_column, y_axis_column='average_cosine_similarity mean'):
        ## here we will make three regression lines, one for rows with total_matching_captions == 0, the second with total_matching_captions > 0 and <= 500, and the third with total_matching_captions > 500. We will use the average_cosine_similarity as the dependent variable and the total_matching_captions as the independent variable. 
        # import ipdb; ipdb.set_trace()
        df = df.sort_values(by=[x_axis_column], ascending=True)
        index_celeb_with_no_captions = np.where(df[x_axis_column].values < 1)[0][-1] + 1
        index_celeb_with_captions_and_less_than_500 = np.where(df[x_axis_column].values <= 500)[0][-1] + 1
        ## now make the three regression lines, and measure the RMSE in the predictions for each of these lines. DO not forget to import the necessary libraries.
        x = df[x_axis_column].values.reshape(-1, 1)
        y = df['average_cosine_similarity mean'].values.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        # import ipdb; ipdb.set_trace()
        reg1 = LinearRegression().fit(x[:index_celeb_with_no_captions], y[:index_celeb_with_no_captions])
        reg2 = LinearRegression().fit(x[index_celeb_with_no_captions:index_celeb_with_captions_and_less_than_500], y[index_celeb_with_no_captions:index_celeb_with_captions_and_less_than_500])
        reg3 = LinearRegression().fit(x[index_celeb_with_captions_and_less_than_500:], y[index_celeb_with_captions_and_less_than_500:])
        full_reg = LinearRegression().fit(x, y)
        y_pred1 = reg1.predict(x[:index_celeb_with_no_captions])
        y_pred2 = reg2.predict(x[index_celeb_with_no_captions:index_celeb_with_captions_and_less_than_500])
        y_pred3 = reg3.predict(x[index_celeb_with_captions_and_less_than_500:])
        y_pred_full = full_reg.predict(x)
        rmse1 = mean_squared_error(y[:index_celeb_with_no_captions], y_pred1, squared=False)
        rmse2 = mean_squared_error(y[index_celeb_with_no_captions:index_celeb_with_captions_and_less_than_500], y_pred2, squared=False)
        rmse3 = mean_squared_error(y[index_celeb_with_captions_and_less_than_500:], y_pred3, squared=False)
        rmse_full = mean_squared_error(y, y_pred_full, squared=False)
        ## round them to 3 decimal places
        rmse1, rmse2, rmse3, rmse_full = round(rmse1, 3), round(rmse2, 3), round(rmse3, 3), round(rmse_full, 3)
        print(f"RMSE for the three regression lines are when using {args.face_recognition_model} and sorted by {x_axis_column}: ", rmse1, rmse2, rmse3, " and the RMSE for the full regression line is ", rmse_full)
        

    if sort_according_to_quality:
        ## In this case we will sort the celebrities not just depending on the number of captions they have, but rather than the number of quality images they have as a percentage of the downloaded images.
        df_quality = pd.read_csv("all_celebs_single_person_training_data_face_recognition_result.csv", sep='|', header=0) ## This has the columns: celeb|count_high_quality_images|total_single_face_real_photograph_images|num_downloaded_training_images
        ## we want to sort the df based on the number of high quality images as a percentage of the total number of downloaded images.
        df_quality['percentage_high_quality_images'] = (df_quality['count_high_quality_images'] / df_quality['num_downloaded_training_images']) * df['total_matching_captions']
        df_quality = df_quality.sort_values(by=['percentage_high_quality_images'], ascending=True)
        ## now get the order of celebs from this sorted df
        sorted_celebs = df_quality['celeb'].values
        # print(len(sorted_celebs), len(celebrity_list))
        # print([i for i in sorted_celebs if i not in celebrity_list])
        assert len(sorted_celebs) == len(celebrity_list)
        celebrity_list = sorted_celebs
        ## now we sort the df based on the sorted_celebs
        df = df.set_index('celeb')
        df = df.loc[celebrity_list]
        df = df.reset_index()

    elif sort_according_to_effective_pixels:
        df_effective_pixels_and_shape = pd.read_csv(f"all_celebs_single_person_training_data_face_recognition_result_summarized.csv", sep='|', header=0)       # celeb|image|effective_pixels|sharpness
        df_effective_pixels_and_shape = df_effective_pixels_and_shape.set_index('celeb')
        ## aggregate the effective pixels for each celeb -- sum them up
        df_effective_pixels_and_shape = df_effective_pixels_and_shape.groupby('celeb').agg({'effective_pixels': 'sum', 'sharpness': 'mean'})
        df_effective_pixels_and_shape = df_effective_pixels_and_shape.loc[df['celeb'].values]
        df_effective_pixels_and_shape = df_effective_pixels_and_shape.reset_index()
        assert len(df_effective_pixels_and_shape) == len(df) and df_effective_pixels_and_shape['celeb'].values.tolist() == df['celeb'].values.tolist()
        df_effective_pixels_and_shape['num_downloaded_training_images'] = df['num_downloaded_training_images'].values
        df_effective_pixels_and_shape['avg_effective_pixels'] = (df_effective_pixels_and_shape['effective_pixels'] / df_effective_pixels_and_shape['num_downloaded_training_images']) * df['total_matching_captions']
        df_effective_pixels_and_shape = df_effective_pixels_and_shape.sort_values(by=['avg_effective_pixels'], ascending=True)
        ## now get the order of celebs from this sorted df
        sorted_celebs = df_effective_pixels_and_shape['celeb'].values
        # print(sorted_celebs)
        assert len(sorted_celebs) == len(celebrity_list)
        celebrity_list = sorted_celebs
        ## now we sort the df based on the sorted_celebs
        df = df.set_index('celeb')
        df = df.loc[df_effective_pixels_and_shape['celeb'].values]
        df = df.reset_index()

    elif sort_according_to_face_count:
        if args.set_of_people in ["celebrity", "politicians"]:
            df = df.sort_values(by=['effective_num_face_counts'], ascending=True)
            ## exclude the celebs with 0 effective_num_face_counts
            # print("length of df before removing celebs with 0 effective_num_face_counts", len(df))
            # df = df[df['effective_num_face_counts'] > 0]
            # print("length of df after removing celebs with 0 effective_num_face_counts", len(df))
        elif args.set_of_people in ["artists_group1", "artists_group2"]:
            df = df.sort_values(by=['total_laion_paintings_this_artist'], ascending=True)
    
    elif sort_according_to_total_face_pixels:
        df = df.sort_values(by=['total_face_pixels'], ascending=True)
      
    else:
        ## in this case we sort according to the number of captions
        df = df.sort_values(by=['total_matching_captions'], ascending=True)

    # measure_noise_in_the_similarity_neasurement_between_generated_and_training_images(copy.deepcopy(df), x_axis_column='total_matching_captions')
    # if sort_according_to_face_count and args.set_of_people in ["celebrity", "politicians"]:
    #     measure_noise_in_the_similarity_neasurement_between_generated_and_training_images(copy.deepcopy(df), x_axis_column='effective_num_face_counts')
    # if sort_according_to_total_face_pixels:
    #     measure_noise_in_the_similarity_neasurement_between_generated_and_training_images(copy.deepcopy(df), x_axis_column='total_face_pixels')

    if args.set_of_people in ["celebrity", "politicians"]:
        name_this_subplot = f'separate_plot_average_cosine_similarity_using_{args.face_recognition_model}'
        if separate_plot_each_race:
            name_this_subplot += f"_{separate_plot_each_race[0]}_{separate_plot_each_race[1]}"
        metric = 'average_cosine_similarity'
    elif args.set_of_people == "cub_birds":
        name_this_subplot = f'separate_plot_{metric_bird}_using_clip_cub_birds'
        metric = metric_bird
    elif args.set_of_people == "celebrity_clip_score":
        name_this_subplot = f'clip_score_{set_of_prompts[0][0]}'
        metric = 'clip_score'
    elif args.set_of_people == "celebrity_fid_distance":
        name_this_subplot = f'fid_distance_{set_of_prompts[0][0]}'
        metric = 'fid_distance'
    elif args.set_of_people in ["artists_group1", "artists_group2"]:
        name_this_subplot = 'separate_plot_average_cosine_similarity_using_style_clip_wikiart_artists'
        metric = 'average_cosine_similarity'
    else:
        raise NotImplementedError

    ## store the combined df in a csv file
    # df.to_csv(f"combined_df_{args.set_of_people}_{args.top_k_training_images}_{args.face_recognition_model}.csv", sep='|', index=False)
    save_this_subplot(args, df, name=name_this_subplot, metric=metric)
    

def tokenize_image_captions(args):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ## we will open the image_captions.csv file for each celeb, load all the captions, tokenize them, and then count the number of tokens in each caption and then average them over the entire file. 
    global celebrity_list
    average_token_counts = {}
    caption_length_file = "average_caption_token_counts.csv"
    with open(caption_length_file, "w") as f:
        f.write(f'celeb|average_caption_token_counts|num_downloaded_images\n')
    
    for celeb in celebrity_list:
        image_caption_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb}/image_captions.csv"
        df = pd.read_csv(image_caption_file, sep='|', header=0)
        token_counts = df['caption'].apply(lambda caption: len([token for token in nlp(caption)]))
        # Calculate the average number of tokens per caption for the celebrity
        average_token_counts[celeb] = token_counts.mean()
        print(f"Average number of tokens per caption for {celeb} is: ", average_token_counts[celeb])
        with open(caption_length_file, "a") as f:
            f.write(f"{celeb}|{average_token_counts[celeb]}|{len(df)}\n")

    print(average_token_counts)


def clustering_training_image_embeddings(args, plot_images_downloaded_from_internet=False):
    '''
    We will cluster the training image embeddings (obtained from a SigLIP model, not the face detection model since we want to capture the the variance in the full image), and then plot the number of images in each cluster and the number of clusters for each celeb. 
    '''
    global celebrity_list, celebrity_list0
    celebrity_list = celebrity_list + celebrity_list0
    if plot_images_downloaded_from_internet:
        raise NotImplementedError
    ## each celeb's training image embeddings are stored in ../downloaded_images/{celeb}/open_clip_image_features_{celeb_here}.pt and ../high_quality_images_downloaded_from_internet/{celeb_here}/open_clip_image_features_{celeb_here}.pt
    ## we will load the embeddings for each celeb and cluster them using HDBSCAN (so that we do not have to tune the hyperparameters). The minimum number of datapoints required to form a cluster will be 5 if the total number of images is less than 50. Else it will be 10% of the total number of images. 
    ## we will also plot the number of images in each cluster and the number of clusters for each celeb.
    import hdbscan, torch
    clusters_and_cluster_sizes = {}
    # import ipdb; ipdb.set_trace()
    for celeb_here in celebrity_list:
        # print(f"Clustering training image embeddings for {celeb_here}")
        if os.path.exists(f"/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb_here}/open_clip_image_features_{celeb_here}.pt"):
            training_image_embeddings = torch.load(f"/gscratch/h2lab/vsahil/vlm-efficiency/downloaded_images/{celeb_here}/open_clip_image_features_{celeb_here}.pt")
        # if os.path.exists(f"../high_quality_images_downloaded_from_internet/{celeb_here}/open_clip_image_features_{celeb_here}.pt"):        ## This should be in a separate plot as these are for the high quality images.
        #     training_image_embeddings = torch.load(f"../high_quality_images_downloaded_from_internet/{celeb_here}/open_clip_image_features_{celeb_here}.pt")
        else:
            print(f"Training image embeddings not found for {celeb_here}")
            continue
        ## note that training_image_embeddings is a dictionary with keys as image names and values as the embeddings, and not a tensor.
        ## we will convert this to a tensor
        # training_image_embeddings = torch.stack(list(training_image_embeddings.values()))
        training_image_embeddings = torch.cat(list(training_image_embeddings.values()))
        print(f"Shape of training_image_embeddings for {celeb_here} is: ", training_image_embeddings.shape)
        ## now we will cluster the embeddings
        if training_image_embeddings.shape[0] < 50:
            min_cluster_size = 5
        elif training_image_embeddings.shape[0] < 200:
            # min_cluster_size = int(0.1 * training_image_embeddings.shape[0])
            min_cluster_size = 10
        else:
            min_cluster_size = 20
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(training_image_embeddings.cpu().numpy())
        print(f"Number of clusters for {celeb_here} is: ", len(np.unique(cluster_labels)))
        print(f"Number of images in each cluster for {celeb_here} is: ", np.unique(cluster_labels, return_counts=True))
        ## convert these arrays in np.unique to list so that it can be saved in json
        clusters_and_cluster_sizes[celeb_here] = (np.unique(cluster_labels).tolist(), np.unique(cluster_labels, return_counts=True)[1].tolist())
    ## save the clusters_and_cluster_sizes dict so that we can plot the number of images in each cluster and the number of clusters for each celeb. With json I got an error: TypeError: Object of type ndarray is not JSON serializable
    with open("clusters_of_training_images.json", "a") as f:
    # clusters_and_cluster_sizes = {k: v for k, v in clusters_and_cluster_sizes.items() if v is not None}
        json.dump(clusters_and_cluster_sizes, f)


def get_celebrity_gender_and_race(args, celeb_here):
    ## Here we will use deepface to get the race and gender of the celebrities from the good training images, and then select the value with the highest probability averaged over these images.
    global celebrity_list, celebrity_list0, celebrities_with_few_images
    df_good_training_images = pd.read_csv("all_celebs_single_person_training_data_face_recognition_result_summarized.csv", sep='|', header=0)
    df_good_training_images = df_good_training_images.set_index('celeb')      ## these are the good images for this celebrity, take top-10 from here
    
    df_good_internet_images =pd.read_csv("all_celebs_single_person_internet_images_downloaded_face_recognition_result_summarized.csv", sep='|', header=0)
    df_good_internet_images = df_good_internet_images.set_index('celeb')      ## these are the good internet images for this celebrity, take top-10 from here
    
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    from deepface import DeepFace
    # img1_path = "../generated_images/Oprah Winfrey/a photorealistic image of /image_seq_23.png"
    # result = DeepFace.analyze(img1_path, actions=["gender", "race"], enforce_detection=True, detector_backend='retinaface', align=False, silent=True)
    # print(result)
    if celeb_here is None:
        celebrity_list_here = celebrities_with_few_images + celebrity_list + celebrity_list0
    else:
        celebrity_list_here = [celeb_here]
    
    final_predicted_gender_and_race = {}
    for celeb in celebrity_list_here:
        list_of_images_for_this_celeb = []
        if celeb in df_good_training_images.index:
            if len(df_good_training_images.loc[celeb]) == 1:
                list_of_images_for_this_celeb.append(df_good_training_images.loc[celeb]['image'])
            else:
                list_of_images_for_this_celeb.extend(df_good_training_images.loc[celeb].head(10)['image'].values)
        if celeb in df_good_internet_images.index:
            if len(df_good_internet_images.loc[celeb]) == 1:
                raise NotImplementedError       ## we have always downloaded more than atleast 7 images per person. 
            else:
                ## here we need to add "../high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}/" to the image path
                list_of_images_for_this_celeb.extend(df_good_internet_images.loc[celeb].head(10)['image'].apply(lambda x: f"../high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}/" + x).values)
                
        assert len(list_of_images_for_this_celeb) > 5 and len(list_of_images_for_this_celeb) <= 20
        ## now that we have the images for this person, let's get the prediction for the gender and race for this person's images.
        ## during this process we also need to assert that there is just one face, and that should be the case because we are using retinaface for detection.
        gender_prediction_this_celeb = {}
        race_prediction_this_celeb = {}
        for img_here in list_of_images_for_this_celeb:
            result = DeepFace.analyze(img_here, actions=["gender", "race"], enforce_detection=True, detector_backend='retinaface', align=False, silent=True)
            assert len(result) == 1, f"More than one face detected in {img_here} for {celeb}"   ## since we selected the good images for each celeb, we should get only one face in each image
            ## result[0] looks like {'gender': {'Woman': 99.69680905342102, 'Man': 0.303193717263639}, 'dominant_gender': 'Woman', 'region': {'x': 130, 'y': 105, 'w': 169, 'h': 234}, 'race': {'asian': 0.002311367367878535, 'indian': 0.015543579832112213, 'black': 99.97528195675827, 'white': 3.8135648311694517e-06, 'middle eastern': 1.8940441622079048e-06, 'latino hispanic': 0.006848831892339063}, 'dominant_race': 'black'}
            ## we want to add all the values of the keys in both gender and race predictions
            for gender in result[0]['gender']:
                if gender in gender_prediction_this_celeb:
                    gender_prediction_this_celeb[gender] += result[0]['gender'][gender]
                else:
                    gender_prediction_this_celeb[gender] = result[0]['gender'][gender]
            
            for race in result[0]['race']:
                if race in race_prediction_this_celeb:
                    race_prediction_this_celeb[race] += result[0]['race'][race]
                else:
                    race_prediction_this_celeb[race] = result[0]['race'][race]
                    
        ##  now we have the total gender and race predictions for all the good training images, and we will take the one with the highest probability
        gender_with_highest_prob = max(gender_prediction_this_celeb, key=gender_prediction_this_celeb.get)
        race_with_highest_prob = max(race_prediction_this_celeb, key=race_prediction_this_celeb.get)
        # final_predicted_gender_and_race[celeb] = (gender_with_highest_prob, race_with_highest_prob)
        print(f"Predicted gender and race for {celeb} is {(gender_with_highest_prob, race_with_highest_prob)}")
        if len(celebrity_list_here) == 1:    
            return (gender_with_highest_prob, race_with_highest_prob)

    ## store the final_predicted, instead of json, write it to a csv file with 3 columns: celeb|gender|race
    # with open("celebrity_predicted_race_and_gender.csv", "w") as f:
    #     # f.write(f'celeb|gender|race\n')
    #     for celeb in final_predicted_gender_and_race:
    #         f.write(f"{celeb}|{final_predicted_gender_and_race[celeb][0]}{final_predicted_gender_and_race[celeb][1]}\n")
    

def prepare_montage_selected_images(args, celeb_here, selected_indices):
    assert args.all_images_internet
    base_embeddings_directory = f"{args.face_recognition_model}_embeddings_close_up" if args.set_of_people == "celebrity" else f"{args.face_recognition_model}_embeddings_close_up_politicians"
    ## given a few images from the all images internet for a celebrity, prepare a montage of these images.
    # close_up_single_person_file = pd.read_csv("close_up_single_person_all_images_internet_celebrity.csv", sep='|', header=0)  ## header: celeb|image|effective_pixels|sharpness
    # this_celeb_images = close_up_single_person_file[close_up_single_person_file['celeb'] == celeb_here]
    
    embeddings_directory = f"{base_embeddings_directory}/all_images_internet"
    with open(f"{embeddings_directory}/{celeb_here}.json", "r") as f:
        this_celeb_images_embeddings = json.load(f)

    ## now compute the similarity between each pair of images by computing the cosine similarity between the embeddings of the images. First we will normalize the embeddings and then compute the cosine similarity.
    sequenced_images = []
    for image in this_celeb_images_embeddings:
        sequenced_images.append(image)
    
    visualizing_all = False
    if selected_indices == "all":
        visualizing_all = True
        selected_indices = range(len(sequenced_images))
    assert max(selected_indices) < len(sequenced_images)

    ## get the names of the images based on the selected indices
    # selected_images = this_celeb_images.iloc[selected_indices]['image'].values
    selected_images = [sequenced_images[i] for i in selected_indices]
    print(selected_images)

    image_path = f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb_here.replace(' ', '-')}/"
    pathed_selected_images = [image_path + img for img in selected_images]
    
    ## now create a montage of these images. maximum 5 images in a row.
    import cv2
    import numpy as np
    images = [cv2.imread(img) for img in pathed_selected_images]
    ## resize the images to the same size
    images = [cv2.resize(img, (224, 224)) for img in images]
    ## now create a montage of these images
    columns = min(7, len(images))
    rows = int(np.ceil(len(images) / columns))
    montage = np.zeros((rows * 224, columns * 224, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        x = (i % columns) * 224
        y = (i // columns) * 224
        montage[y:y+224, x:x+224] = img
        ## add text to the image top left corner
        image_name = selected_images[i].split('/')[-1]
        cv2.putText(montage, image_name, (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    base_save_directory = f"{base_embeddings_directory}/all_images_internet_similarity_matrix/{celeb_here}"
    assert os.path.exists(base_save_directory)
    os.makedirs(f"{base_save_directory}/montage_selected_images", exist_ok=True)
    if not visualizing_all:
        selected_indices = "_".join([str(i) for i in selected_indices])
        cv2.imwrite(f"{base_save_directory}/montage_selected_images/{celeb_here}_{selected_indices}.png", montage)
    else:
        cv2.imwrite(f"{base_save_directory}/montage_selected_images/all_{celeb_here}.png", montage)


def compute_similarity_selected_images(args, call_from_face_recoginition=False, call_from_face_recognition_celeb=None):
    assert args.stable_diffusion_model is not None
    base_embeddings_directory = f"{args.face_recognition_model}_embeddings_close_up_{args.stable_diffusion_model}" if args.set_of_people == "celebrity" else f"{args.face_recognition_model}_embeddings_close_up_politicians_{args.stable_diffusion_model}" if args.set_of_people == "politicians" else f"{args.face_recognition_model}_embeddings_close_up_caption_assumption_{args.stable_diffusion_model}" if args.set_of_people == "caption_assumption" else None
    
    def get_face_counts_using_selected_images_and_laion_images(args, celeb, selected_images_embedding, min_threshold):
        ## here we have the embeddings from the selected images of this celeb, and we want to load the embeddings of their laion images, and compute the cosine similarity between all pairs of faces
        ## treat each face in each image as independent. The cosine similarity will be computed between each face in each image in LAION and each embedding in the selected images.
        ## we will normalize the embeddings and then compute the cosine similarity.
        assert args.all_laion_images or args.all_laion_alias_name_images
                    
        if args.all_laion_images:
            assert isinstance(celeb, str)
            if args.set_of_people == "celebrity":
                embeddings_directory = f"{base_embeddings_directory}/all_laion_images"
                laion_filtered_images_directory = f"{base_embeddings_directory}/all_laion_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
            elif args.set_of_people == "politicians":
                embeddings_directory = f"{base_embeddings_directory}/all_laion_images"
                laion_filtered_images_directory = f"{base_embeddings_directory}/all_laion_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
            else:   raise NotImplementedError
        
        elif args.all_laion_alias_name_images:
            assert args.set_of_people == "celebrity"
            embeddings_directory = f"{base_embeddings_directory}/all_laion_alias_name_images"
            laion_filtered_images_directory = f"{base_embeddings_directory}/all_laion_alias_name_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
            assert isinstance(celeb, tuple)
            main_celeb = celeb[0]
            celeb = celeb[1]
        
        else:      ## this should not happen
            raise NotImplementedError
        
        print(f"opening the embeddings file for {celeb}")
        with open(f"{embeddings_directory}/{celeb}.json", "r") as f:
            laion_images_embeddings_file = json.load(f)
        
        if len(laion_images_embeddings_file) == 0:      ## this happens to celebs with no matches in the LAION dataset. 
            # return np.zeros(selected_images_embedding.shape[0])
            return 0, 0
        
        ## this gives us the keys as the image_face_boundingbox and its embedding as the value
        laion_image_embeddings = [laion_images_embeddings_file[image_face] for image_face in laion_images_embeddings_file]
        ## now we also want to get the number of pixels in each face that matches with the selected images. 
        ## the size of the face is saved in the key of the image_face_boundingbox like this: "image_seq_1.jpg_[114.41725   35.908657 151.3013    82.81814 ]": embedding. We want to get these values and then compute the area of the face.
        face_areas = []
        for image_face in laion_images_embeddings_file:     ## image_face is the key
            bounding_box = image_face.split('_[')[1].split(']')[0]
            x1, y1, x2, y2 = [float(x) for x in bounding_box.split()]
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            area_of_face = (x2 - x1) * (y2 - y1)
            assert x1 <= x2 and y1 <= y2
            face_areas.append(area_of_face)
        
        face_areas = np.array(face_areas)
        assert len(face_areas) == len(laion_image_embeddings)
        laion_image_embeddings = np.array(laion_image_embeddings)
        laion_image_embeddings = laion_image_embeddings / np.linalg.norm(laion_image_embeddings, axis=1)[:, np.newaxis]
        selected_images_embedding = selected_images_embedding / np.linalg.norm(selected_images_embedding, axis=1)[:, np.newaxis]
        cosine_similarity_matrix = np.dot(laion_image_embeddings, selected_images_embedding.T)
        assert cosine_similarity_matrix.shape[0] == len(laion_image_embeddings) and cosine_similarity_matrix.shape[1] == len(selected_images_embedding)
        # assert np.allclose(np.diag(cosine_similarity_matrix), 1.0)        ## this will not hold as we are not doing self similarity
        assert np.all(cosine_similarity_matrix >= -1.0 - 1e-6) and np.all(cosine_similarity_matrix <= 1.0 + 1e-6)
        ## now we have the cosine similarity of each laion image with each selected image, we want to take the max similarity of a laion image to any selected image.
        max_similarity = np.max(cosine_similarity_matrix, axis=1)
        assert max_similarity.shape[0] == len(laion_image_embeddings)
        ## count the number of images with similarity greater than min_threshold
        num_images_with_similarity_greater_than_threshold = np.sum(max_similarity > min_threshold)
        if num_images_with_similarity_greater_than_threshold == 0:
            return 0, 0

        ## also store the embeddings of the faces that have greater than min_threshold similarity to a file so that we can use it to compare to the generated images.
        os.makedirs(laion_filtered_images_directory, exist_ok=True)
        ## note that max_similarity is a numpy array corresponding to the laion_image_embeddings, it does not have the keys of the laion_images_embeddings_file. We will use the laion_images_embeddings_file to get the embeddings of the faces that have similarity greater than min_threshold. 
        laion_filtered_images_embeddings = np.array([laion_images_embeddings_file[image_face] for image_face, similarity in zip(laion_images_embeddings_file, max_similarity) if similarity > min_threshold])
        ## save the names of the images that have similarity greater than min_threshold ## get rid of the bounding box part of the image_face
        laion_filtered_images_names = [image_face.split('_[')[0] for image_face, similarity in zip(laion_images_embeddings_file, max_similarity) if similarity > min_threshold]
        ## we also want to save the indices of the images that have similarity greater than min_threshold, so that we can access those faces later.
        laion_filtered_images_indices = np.array([image_face for image_face, similarity in zip(laion_images_embeddings_file, max_similarity) if similarity > min_threshold])
        laion_filtered_images_embeddings = laion_filtered_images_embeddings / np.linalg.norm(laion_filtered_images_embeddings, axis=1)[:, np.newaxis]
        assert len(laion_filtered_images_embeddings) == num_images_with_similarity_greater_than_threshold == len(laion_filtered_images_indices)
        ## assert that the cosine similarity of the laion filtered images with the selected images is greater than min_threshold for atleast one selected image
        assert np.all(np.max(np.dot(laion_filtered_images_embeddings, selected_images_embedding.T), axis=1) > min_threshold)
        if args.all_laion_images:
            ## now save the laion_filtered_images_embeddings to a file, this is a numpy array
            np.save(f"{laion_filtered_images_directory}/{celeb}.npy", laion_filtered_images_embeddings)
            ## also save the laion_filtered_images_indices to a file, this is a numpy array
            np.save(f"{laion_filtered_images_directory}/{celeb}_laion_filtered_images_indices.npy", laion_filtered_images_indices)
            ## save the names of the images that have similarity greater than min_threshold, 
            # with open(f"{laion_filtered_images_directory}/{celeb}_laion_filtered_images_names.txt", "w") as f:
            #     for name in laion_filtered_images_names:
            #         f.write(f"{name}\n")
        elif args.all_laion_alias_name_images:
            ## now save the laion_filtered_images_embeddings to a file, this is a numpy array
            np.save(f"{laion_filtered_images_directory}/{main_celeb}_{celeb}.npy", laion_filtered_images_embeddings)
            ## also save the laion_filtered_images_indices to a file, this is a numpy array
            np.save(f"{laion_filtered_images_directory}/{main_celeb}_{celeb}_laion_filtered_images_indices.npy", laion_filtered_images_indices)
            ## save the names of the images that have similarity greater than min_threshold,
            with open(f"{laion_filtered_images_directory}/{main_celeb}_{celeb}_laion_filtered_images_names.txt", "w") as f:
                for name in laion_filtered_images_names:
                    f.write(f"{name}\n")
        else:      ## this should not happen
            raise NotImplementedError
        # return max_similarity
        assert num_images_with_similarity_greater_than_threshold <= len(laion_image_embeddings)
        ## now get the face areas of the faces that have similarity greater than min_threshold
        face_areas = face_areas[max_similarity > min_threshold]
        assert len(face_areas) == num_images_with_similarity_greater_than_threshold
        
        def get_face_counts_split_by_number_of_people_in_images(laion_filtered_images_names):
            ## Here we want to get the image counts of the faces split by the image type: single person images, 2 person images, and more than 2 person images. The number of times an image occurs in the image embeddings file is the number of people in that image.
            image_counts = {}
            for image_face in laion_images_embeddings_file:
                image_name = image_face.split('_[')[0]
                if image_name in image_counts:
                    image_counts[image_name] += 1
                else:
                    image_counts[image_name] = 1
            
            ## now we want to count how many of the single image faces have similarity greater than min_threshold, how many of the 2 person images have similarity greater than min_threshold, and how many of the more than 2 person images have similarity greater than min_threshold.
            single_person_image_matching_faces = 0
            two_person_image_matching_faces = 0
            more_than_two_person_image_matching_faces = 0
            for image in laion_filtered_images_names:
                if image_counts[image] == 1:
                    single_person_image_matching_faces += 1
                elif image_counts[image] == 2:
                    two_person_image_matching_faces += 1
                else:
                    more_than_two_person_image_matching_faces += 1
                    
            print("Number of single person images with similarity greater than min_threshold: ", single_person_image_matching_faces)
            print("Number of two person images with similarity greater than min_threshold: ", two_person_image_matching_faces)
            print("Number of more than two person images with similarity greater than min_threshold: ", more_than_two_person_image_matching_faces)
            
        # get_face_counts_split_by_number_of_people_in_images(laion_filtered_images_names)
            
        return num_images_with_similarity_greater_than_threshold, np.sum(face_areas)
    
    
    def get_best_images_and_compute_similarity(args, celeb, selected_indices, bad_celebs_exam=False, take_from_high_quality_internet=False, print_file=None, get_selected_image_embeddings=False):
        
        if args.set_of_people == "celebrity" or args.set_of_people == "caption_assumption":
            if take_from_high_quality_internet: assert bad_celebs_exam
            
            if bad_celebs_exam and take_from_high_quality_internet:
                list_of_all_images = os.listdir(f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}")
                list_of_all_images = [x for x in list_of_all_images if x.startswith(celeb.replace(' ', '-'))]
                ## here the sequence of the images to pick is the celeb-{i}.ext -- it is the i here in selected_indices -- we don't know the extension of the images.
                selected_images = [f"{celeb.replace(' ', '-')}-{i}" for i in selected_indices]
                ## now use the starting part of the selected images to get the extension of the images.
                selected_images_with_extension = []
                for image in selected_images:
                    for img in list_of_all_images:
                        if img.startswith(image):
                            selected_images_with_extension.append(img)
                            break
                assert len(selected_images) == len(selected_images_with_extension)
                selected_images = selected_images_with_extension
                
                embeddings_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people == 'celebrity' else 'insightface_resnet100_embeddings_close_up_politicians_1'}/images_I_downloaded_from_internet"
                with open(f"{embeddings_directory}/{celeb}.json", "r") as f:
                    this_celeb_images_embeddings = json.load(f)
                
                selected_images_embeddings = [this_celeb_images_embeddings[image] for image in selected_images]
                final_selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}/" + img for img in selected_images]
                    
            elif bad_celebs_exam:
                list_of_all_images = os.listdir(f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}")
                list_of_all_images = [x for x in list_of_all_images if x.endswith(".jpg")]
                ## here the sequence of the images to pick is the image_seq_{i}.jpg -- it is the i here in selected_indices
                selected_images = [f"image_seq_{i}.jpg" for i in selected_indices]
                # assert that all images in selected_images are in list_of_all_images
                assert all([img in list_of_all_images for img in selected_images])
                embeddings_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people in ['celebrity', 'caption_assumption'] else 'insightface_resnet100_embeddings_close_up_politicians_1'}/all_images_internet"
                with open(f"{embeddings_directory}/{celeb}.json", "r") as f:
                    this_celeb_images_embeddings = json.load(f)
                
                selected_images_embeddings = [this_celeb_images_embeddings[image] for image in selected_images]
                final_selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}/" + img for img in selected_images]
            
            else:
                ## Here are the images are being taken from all_image_internet
                embeddings_directory = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/{'insightface_resnet100_embeddings_close_up_1' if args.set_of_people in ['celebrity', 'caption_assumption'] else 'insightface_resnet100_embeddings_close_up_politicians_1'}/all_images_internet"
                with open(f"{embeddings_directory}/{celeb}.json", "r") as f:
                    this_celeb_images_embeddings = json.load(f)

                ## now compute the similarity between each pair of images by computing the cosine similarity between the embeddings of the images. First we will normalize the embeddings and then compute the cosine similarity.
                sequenced_images = []
                for image in this_celeb_images_embeddings:
                    sequenced_images.append(image)
                assert max(selected_indices) < len(sequenced_images)
                selected_images = [sequenced_images[i] for i in selected_indices]
                
                selected_images_embeddings = [this_celeb_images_embeddings[image] for image in selected_images]
                final_selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}/" + img for img in selected_images]

        elif args.set_of_people == "politicians":
            embeddings_directory = "/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/insightface_resnet100_embeddings_close_up_politicians_1/images_I_downloaded_from_internet"
            with open(f"{embeddings_directory}/{celeb}.json", "r") as f:
                this_celeb_images_embeddings = json.load(f)

            sequenced_images = []
            for image in this_celeb_images_embeddings:
                sequenced_images.append(image)
            
            assert max(selected_indices) < len(sequenced_images)
            selected_images = [sequenced_images[i] for i in selected_indices]       
            
            selected_images_embeddings = [this_celeb_images_embeddings[img] for img in selected_images]
            selected_images_embeddings = np.array(selected_images_embeddings)
            ## all the selected images are of the form: 'image_seq_{i}.jpg_[{x1} {y1} {x2} {y2}]' -- so get the file names by just taking the image_seq_{i}.jpg. 
            selected_images = [img.split('_[')[0] for img in selected_images]
            final_selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet_politicians/{celeb.replace(' ', '-')}/" + img for img in selected_images]
            ## assert all the selected images exist
            assert all([os.path.exists(img) for img in final_selected_images]), f"final selected images were: {final_selected_images}"
            
        else:
            raise NotImplementedError

        ## get the names of the images based on the selected indices
        # print(selected_images)
        assert len(selected_images) == len(selected_indices) == len(selected_images_embeddings)
        
        selected_images_embeddings = np.array(selected_images_embeddings)
        if get_selected_image_embeddings:
            return selected_images_embeddings, final_selected_images
        
        ## given the embeddings of the selected images, compute the cosine similarity between each pair of images.
        selected_images_embeddings = selected_images_embeddings / np.linalg.norm(selected_images_embeddings, axis=1)[:, np.newaxis]
        cosine_similarity_matrix = np.dot(selected_images_embeddings, selected_images_embeddings.T)
        assert cosine_similarity_matrix.shape[0] == cosine_similarity_matrix.shape[1] == len(selected_images)
        assert np.allclose(np.diag(cosine_similarity_matrix), 1.0)
        assert np.all(cosine_similarity_matrix >= -1.0 - 1e-6) and np.all(cosine_similarity_matrix <= 1.0 + 1e-6)
        avg_cosine_similarity = np.mean(cosine_similarity_matrix)
        print(f"Average cosine similarity for {celeb} is: ", avg_cosine_similarity)

        # with open(print_file, "a") as f:
        #     f.write(f"{celeb}|{avg_cosine_similarity}\n")

        if args.set_of_people == "celebrity":
            ## print all the selected image for each celebrity in a file. 
            selected_images_file = f"{base_embeddings_directory}/selected_images_for_all_celebs.csv"
            ## add the full path file to the selected_images_file
            if bad_celebs_exam and take_from_high_quality_internet:
                selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}/" + img for img in selected_images]
            elif bad_celebs_exam:
                selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}/" + img for img in selected_images]
            else:
                selected_images = [f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}/" + img for img in selected_images]
                
            ## assert that each of the selected images exists
            assert all([os.path.exists(img) for img in selected_images])
        
        return None, None


    def face_counts_using_selected_images_in_random_laion_images(args, celeb, selecting_images_embeddings, min_threshold):
        assert args.set_of_people == "caption_assumption"
        embeddings_directory = f"{base_embeddings_directory}/all_laion_images"
        # laion_filtered_images_directory = f"{base_embeddings_directory}/all_laion_images_embeddings_filtered_by_selected_images_{args.threshold_for_max_f1_score}"
        ## read all the json files inside the embeddings directory and get the embeddings of the faces that have similarity greater than min_threshold
        ## assert that there are 13 files inside the embeddings_directory that end with .json
        assert len([file for file in os.listdir(embeddings_directory) if file.endswith(".json")]) == 13
        laion_image_embeddings = []
        laion_image_file_names = []
        for file in os.listdir(embeddings_directory):
            if file.endswith(".json"):
                with open(f"{embeddings_directory}/{file}", "r") as f:
                    laion_images_embeddings_file = json.load(f)
                laion_image_embeddings_this_file = [laion_images_embeddings_file[image_face] for image_face in laion_images_embeddings_file]
                ## get the full path of the images
                laion_images_file_names_this_file = [f"/gscratch/scrubbed/vsahil/all_downloaded_images_for_captions/{file.split('.')[0]}/" + image_face.split('_[')[0] for image_face in laion_images_embeddings_file]
                laion_image_embeddings.extend(laion_image_embeddings_this_file)
                laion_image_file_names.extend(laion_images_file_names_this_file)
        
        # with open(f"{embeddings_directory}/as.json", "r") as f:
        #     laion_images_embeddings_file = json.load(f)
        
        # ## get the laion images embeddings for the image starting with image_seq_3502.jpg
        # laion_image_embeddings = [laion_images_embeddings_file[image_face] for image_face in laion_images_embeddings_file if image_face.startswith("image_seq_3502.jpg")]
        # laion_image_embeddings = np.array(laion_image_embeddings)
        
        ## now we have the laion_image embeddings, match it with the selecting_images_embeddings, if the similarity to any of the selecting_images_embeddings is greater than min_threshold, then count it as a face.
        laion_image_embeddings = np.array(laion_image_embeddings)
        print("total faces in random laion images", laion_image_embeddings.shape[0])
        laion_image_embeddings = laion_image_embeddings / np.linalg.norm(laion_image_embeddings, axis=1)[:, np.newaxis]
        selecting_images_embeddings = selecting_images_embeddings / np.linalg.norm(selecting_images_embeddings, axis=1)[:, np.newaxis]
        cosine_similarity_matrix = np.dot(laion_image_embeddings, selecting_images_embeddings.T)
        assert cosine_similarity_matrix.shape[0] == len(laion_image_embeddings) and cosine_similarity_matrix.shape[1] == len(selecting_images_embeddings)
        assert np.all(cosine_similarity_matrix >= -1.0 - 1e-6) and np.all(cosine_similarity_matrix <= 1.0 + 1e-6)
        ## now we have the cosine similarity of each laion image with each selected image, we want to take the max similarity of a laion image to any selected image.
        max_similarity = np.max(cosine_similarity_matrix, axis=1)
        assert max_similarity.shape[0] == len(laion_image_embeddings)
        ## count the number of images with similarity greater than min_threshold
        num_images_with_similarity_greater_than_threshold = np.sum(max_similarity > min_threshold)
        ## get the file names of the images that have similarity greater than min_threshold
        laion_filtered_images_names = [laion_image_file_names[i] for i in range(len(laion_image_file_names)) if max_similarity[i] > min_threshold]
        print(f"Number of face of {celeb} in random 100K LAION images is {num_images_with_similarity_greater_than_threshold}")
        print(f"the files that have similarity greater than min_threshold are: {laion_filtered_images_names}")
        
            
    # assert args.set_of_people == "celebrity"
    ## This function takes the embeddings of the images of a celeb that have been selected by the submodular min function followed by manual vetting and are ensured to belong to that celeb. Take the embedding of all those images and compute the cosine similarity between each pair of images.
    bad_celeb_list = ['Taylor Tomasi Hill', 'Frank Sheeran', 'Nick Offerman', 'Aditya Kusupati', 'Miguel Bezos', 'Ben Mallah', 'Tara Lynn Wilson', 'Quinton Reynolds', 'Manny Khoshbin', 'Alex Choi', 'Jeremy Bamber', 'Taylor Mills', 'Jim Walton', 'Mat Fraser', 'Peter Firth', 'Chris McCandless', 'Olivia Rodrigo', 'Patricia Field', 'Sajith Rajapaksa', 'Kudakwashe Rutendo', 'Chintan Rachchh', 'Tioreore Ngatai-Melbourne']
    no_images_people = ['Ciara Wilson', 'Jillie Mack', 'Kyla Weber', 'Cacee Cobb', 'Sam Taylor Johnson', 'Danielle Jonas']
    total_failed = bad_celeb_list + no_images_people
    take_from_high_quality_internet = ['Taylor Tomasi Hill', 'Manny Khoshbin', 'Jillie Mack', 'Kyla Weber', 'Aditya Kusupati', 'Miguel Bezos', 'Ben Mallah', 'Tara Lynn Wilson', 'Quinton Reynolds', 'Sajith Rajapaksa', 'Taylor Mills', 'Alex Choi', 'Cacee Cobb', 'Sam Taylor Johnson', 'Danielle Jonas', 'Ciara Wilson']
    for take_from_high_quality in take_from_high_quality_internet:
        assert take_from_high_quality in total_failed

    if call_from_face_recoginition:
        if args.set_of_people == "celebrity":
            ## here we just want to get the selected image embeddings of each selected image for each celeb
            full_celeb_list = celebrities_with_few_images + to_be_added_celebrities + celebrity_list + celebrity_list0
        elif args.set_of_people == "politicians":
            politicians_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")
            assert len(politicians_df) == 418
            full_celeb_list = politicians_df['Name'].to_list()
        else:   raise NotImplementedError
        
        assert call_from_face_recognition_celeb and call_from_face_recognition_celeb in full_celeb_list
        bad_celebs_exam = False
        take_from_high_quality = False
        celeb = call_from_face_recognition_celeb

        if args.set_of_people == "celebrity":
            if celeb in total_failed:
                bad_celebs_exam = True
                if celeb in take_from_high_quality_internet:
                    take_from_high_quality = True

            if bad_celebs_exam and celeb in take_from_high_quality_internet:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{celeb.replace(' ', '-')}/submarine_best_subset_{celeb}.json"
            elif bad_celebs_exam:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{celeb.replace(' ', '-')}/submarine_best_subset_{celeb}.json"
            else:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/insightface_resnet100_embeddings_close_up_1/all_images_internet_similarity_matrix/{celeb}/submarine_best_subset_{celeb}.json"
                
        elif args.set_of_people == "politicians":
            best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/insightface_resnet100_embeddings_close_up_politicians_1/images_I_downloaded_from_internet_similarity_matrix/{celeb}/submarine_best_subset_{celeb}.json"
        
        else: 
            raise NotImplementedError
            
        assert os.path.exists(best_subset_file)
        with open(best_subset_file, "r") as f:
            data = json.load(f)
            selected_indices = data["set"]
        
        selected_image_embeddings, final_selected_images = get_best_images_and_compute_similarity(args, celeb, selected_indices, bad_celebs_exam, take_from_high_quality_internet=take_from_high_quality, get_selected_image_embeddings=True)
        return selected_image_embeddings, final_selected_images
    
    if args.set_of_people == "caption_assumption":
        full_celeb_list = ["Floyd Mayweather","Oprah Winfrey","Ronald Reagan","Ben Affleck","Anne Hathaway","Stephen King","Johnny Depp","Abraham Lincoln","Kate Middleton","Donald Trump"]
        # full_celeb_list = ["Johnny Depp"]
    
    elif args.all_laion_images and not args.set_of_people == "caption_assumption":
        if args.set_of_people == "celebrity":
            full_celeb_list = celebrities_with_few_images + to_be_added_celebrities + celebrity_list + celebrity_list0
            # full_celeb_list = ["Portia Freeman", "Cacee Cobb", "Charli D'Amelio", "Kendra Spears", "Elliot Page", "Ziyi Zhang", "Olivia Rodrigo", "Addison Rae", "Fei Fei Sun", "Annie Murphy", "Sofia Hellqvist", "Emma Corrin", "Sam Taylor Johnson", "Sha'Carri Richardson"]
            
        elif args.set_of_people == "politicians":
            politicians_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")   ## Name,counts_in_laion2b-en
            assert len(politicians_df) == 418
            full_celeb_list = politicians_df['Name'].to_list()
        
        else:   ## this should not happen
            raise NotImplementedError

    elif args.all_laion_alias_name_images and not args.set_of_people == "caption_assumption":
        assert args.set_of_people == "celebrity"
        ## here the full celeb list are all the aliases of the celebs in the laion dataset.
        alias_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrity_counts_with_aliases.csv", sep="|")     ##celeb|full_name_count|aliases_and_counts|alias_total_count|Outlier
        ## get all the alias names in a list dictionary, where the key is the alias name and the value is the main celeb name. The main celeb name is "celeb" and each celeb has a dict under aliases_and_counts, where each of their alias names with their counts are given. For this purpose, the counts are not needed.
        dict_alias_key_celeb_value = {}
        for celeb in alias_df['celeb']:     ## this is a problem because mutltiple people can share an alias and we want to account for both of them. 
            for alias in eval(alias_df[alias_df['celeb'] == celeb]['aliases_and_counts'].values[0]):
                dict_alias_key_celeb_value[(celeb, alias)] = celeb
        assert len(dict_alias_key_celeb_value) == 465, f"{len(dict_alias_key_celeb_value)} != 465"
        full_celeb_list = list(dict_alias_key_celeb_value.keys())
        # full_celeb_list = [('John Wayne Gacy', 'Gacy'), ('Scott Speedman', 'Speedman'), ('Phoebe Bridgers', 'Bridgers'), ('Phoebe Dynevor', 'Dynevor'), ('Sofia Hellqvist', 'Hellqvist'), ('Jacob Sartorius', 'Sartorius'), 
        #                    ('Julia Stegner', 'Stegner'), ('Kim Zolciak-Biermann', 'Biermann'), ('Diego Boneta', 'Boneta'), ('Devon Sawa', 'Sawa'), ('Meg Ryan', 'Ryan'), ('Danny Thomas', 'Thomas'), 
        #                    ('Ashlee Simpson', 'Simpson'), ('Rose Leslie', 'Leslie'), ('Kurt Russell', 'Russell'), ('Lukita Maxwell', 'Maxwell'), ('Lori Harvey', 'Harvey'), ('Michael Buffer', 'Buffer'), 
        #                    ('Lindsay Price', 'Price'), ('Lacey Evans', 'Evans')]
        full_celeb_list = [('Danneel Harris', 'Danneel'), ('Normani Kordei', 'Normani'), ('Penélope Cruz', 'Penélope'), ('Avani Gregg', 'Avani'), ('Cacee Cobb', 'Cacee'), ('Simu Liu', 'Simu'), ('Michiel Huisman', 'Michiel'), ('Léa Seydoux', 'Léa'), ('Skai Jackson', 'Skai'), ('Selita Ebanks', 'Selita'), ('Forest Whitaker', 'Forest'), ('Samuel L. Jackson', 'Samuel'), ('David Tennant', 'David'), ('Joe Rogan', 'Joe'), ('Nick Offerman', 'Nick'), ('Nicole Fosse', 'Nicole'), ('Steve Lacy', 'Steve'), ('John Corbett', 'John'), ('Cody Lightning', 'Cody'), ('Olivia Culpo', "Olivia")]
    
    else:
        raise NotImplementedError
    
    file_open_mode = "a"
    
    print_file = f"selected_images_{args.set_of_people}_avg_cosine_similarity.csv"
    if not os.path.exists(print_file) and not args.set_of_people == "caption_assumption":
        with open(print_file, "w") as f:
            f.write("celeb|avg_cosine_similarity\n")

    compute_avg_value = False
    get_selected_image_embeddings = True
    compute_similarity_for_different_people = False     ## we use this as the negative examples and the similarity among the same people as the positive examples, and use it to establish the threshold for determining if someone is the same person or not. 

    if compute_avg_value:
        cosine_similarities_same_celebs = pd.read_csv(f"selected_images_{args.set_of_people}_avg_cosine_similarity.csv", sep='|', header=0)
        assert len(cosine_similarities_same_celebs) == len(full_celeb_list), f"{len(cosine_similarities_same_celebs)} != {len(full_celeb_list)}"
        avg_cosine_similarities_same_celebs = cosine_similarities_same_celebs['avg_cosine_similarity'].values
        avg_cosine_similarities_same_celebs = [float(x) for x in avg_cosine_similarities_same_celebs]
        print("Average cosine similarity for all celebs is: ", np.mean(avg_cosine_similarities_same_celebs), np.std(avg_cosine_similarities_same_celebs), np.min(avg_cosine_similarities_same_celebs), np.max(avg_cosine_similarities_same_celebs), np.mean(avg_cosine_similarities_same_celebs) - 1 * np.std(avg_cosine_similarities_same_celebs), np.mean(avg_cosine_similarities_same_celebs) + 1 * np.std(avg_cosine_similarities_same_celebs))
        # print(cosine_similarities_same_celebs[cosine_similarities_same_celebs['avg_cosine_similarity'] < 0.6].sort_values(by='avg_cosine_similarity')) #['celeb'].tolist())
        # print(cosine_similarities_same_celebs[cosine_similarities_same_celebs['avg_cosine_similarity'] >= 0.85].sort_values(by='avg_cosine_similarity')) #['celeb'].tolist())
        
        cosine_similarities_different_celebs = pd.read_csv(f"selected_images_{args.set_of_people}_avg_cosine_similarity_different_people.csv", sep='|', header=0) ## celeb1|celeb2|avg_cosine_similarity
        avg_cosine_similarities_different_celebs = cosine_similarities_different_celebs['avg_cosine_similarity'].values
        avg_cosine_similarities_different_celebs = [float(x) for x in avg_cosine_similarities_different_celebs]
        print("Average cosine similarity for different celebs is: ", np.mean(avg_cosine_similarities_different_celebs), np.std(avg_cosine_similarities_different_celebs), np.min(avg_cosine_similarities_different_celebs), np.max(avg_cosine_similarities_different_celebs), np.mean(avg_cosine_similarities_different_celebs) - 1 * np.std(avg_cosine_similarities_different_celebs), np.mean(avg_cosine_similarities_different_celebs) + 1 * np.std(avg_cosine_similarities_different_celebs))
        
        avg_cosine_similarities_same_celebs = avg_cosine_similarities_same_celebs * (len(avg_cosine_similarities_different_celebs) // len(avg_cosine_similarities_same_celebs))     ## we just repeat the list of the avg_cosine_similarities_same_celebs to match the length of avg_cosine_similarities_different_celebs, for better histogram plotting.
        # assert that all elements are less than 1 + 1e-6
        assert np.all(np.array(avg_cosine_similarities_same_celebs) <= 1.0 + 1e-6) and np.all(np.array(avg_cosine_similarities_same_celebs) >= -1.0 - 1e-6)
        
        from sklearn.metrics import roc_curve
        scores = np.concatenate((avg_cosine_similarities_different_celebs, avg_cosine_similarities_same_celebs))
        true_labels = np.concatenate( ( np.zeros(len(avg_cosine_similarities_different_celebs)), np.ones(len(avg_cosine_similarities_same_celebs)) ) )
        fpr, tpr, thresholds = roc_curve(true_labels, scores)
        
        ## ## find the threshold that maximizes the F1 score
        f1_score = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr))
        # for threshold_index in range(400, 500):
        #     print(f"Threshold: {thresholds[threshold_index]}, FPR: {fpr[threshold_index]}, TPR: {tpr[threshold_index]}, F1 score: {f1_score[threshold_index]}")
        if args.set_of_people == "celebrity" or args.set_of_people == "politicians":
            ## choose the threshold to be midpoint of the lowest value of avg_cosine_similarities_same_celebs and the highest value of avg_cosine_similarities_different_celebs
            threshold_for_max_f1_score = (np.min(avg_cosine_similarities_same_celebs) + np.max(avg_cosine_similarities_different_celebs)) / 2       ## 0.45668589190104747
            print(f"Threshold for max F1 score: {threshold_for_max_f1_score}")
            ## at this threshold, print the FPR and TPR, note that this threshold might not be in the thresholds array, so we need to find the closest threshold to this value.
            threshold_index = np.argmin(np.abs(thresholds - threshold_for_max_f1_score))
            print(f"Threshold: {thresholds[threshold_index]}, FPR: {fpr[threshold_index]}, TPR: {tpr[threshold_index]}, F1 score: {f1_score[threshold_index]}")
        else:    
            threshold_for_max_f1_score = thresholds[np.argmax(f1_score)]        ## 0.5576227593909431 higher for celebs
            print(f"Threshold for max F1 score: {threshold_for_max_f1_score}")
            ## at this threshold, print the FPR and TPR
            threshold_index = np.where(thresholds == threshold_for_max_f1_score)[0][0]
            print(f"Threshold: {thresholds[threshold_index]}, FPR: {fpr[threshold_index]}, TPR: {tpr[threshold_index]}, F1 score: {f1_score[threshold_index]}")
            
        ## plot the histogram of the cosine similarities
        import matplotlib.pyplot as plt
        bins = 20
        plt.figure(figsize=(10, 7))
        plt.hist(avg_cosine_similarities_same_celebs, bins=bins, alpha=0.8, label='Similarity between same people', color='blue')
        plt.hist(avg_cosine_similarities_different_celebs, bins=bins, alpha=0.8, label='Similarity between different people', color='red')
        plt.xlabel("Average Cosine Similarity", fontsize=20)
        plt.ylabel("Number of Celebrities", fontsize=20)
        ## add the minimum and maximum values of the cosine similarity as texts on the top left of the plot
        # plt.text(0.55, 22.5, f"Min similarity: {round(np.min(avg_cosine_similarities_same_celebs), 2)}", fontsize=12)
        # plt.text(0.55, 21, f"Max similarity: {round(np.max(avg_cosine_similarities_same_celebs), 2)}", fontsize=12)
        
        ## add a vertical black line at the threshold = 0.1849365234375. Also add the point on the x-axis where the threshold is.
        plt.axvline(x=threshold_for_max_f1_score, color='k', linestyle='--', label='Threshold', ymax=0.63)
        ## label the point where the threshld line touches the x-axis -- this is the point where the threshold is.
        plt.text(threshold_for_max_f1_score-0.03, 12000, f"Threshold: {round(threshold_for_max_f1_score, 2)}", fontsize=22, verticalalignment='bottom', horizontalalignment='center')
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"histogram_avg_cosine_similarity_selected_images_for_same_people_and_different_people_{args.set_of_people}.pdf")
        print("plotting of the histogram of the cosine similarities done")
        return

    image_embeddings_each_celeb_for_selected_images = {}
    num_face_counts = {}
    face_areas_celebs = {}
    
    if args.set_of_people in ["celebrity", "politicians", "caption_assumption"]:        ## let's use the same threshold for politician as well. 
        if args.same_face_threshold == "lower":
            args.threshold_for_max_f1_score = 0.45668589190104747
        elif args.same_face_threshold == "higher":
            args.threshold_for_max_f1_score = 0.5576227593909431
        else:
            raise NotImplementedError


    for celeb_processing in full_celeb_list:
        bad_celebs_exam = False
        take_from_high_quality = False
        if args.all_laion_images:
            main_celeb = celeb_processing
        elif args.all_laion_alias_name_images:
            main_celeb = celeb_processing[0]
        
        if args.set_of_people == "celebrity" or args.set_of_people == "caption_assumption":
            if main_celeb in total_failed:
                bad_celebs_exam = True
                if main_celeb in take_from_high_quality_internet:
                    take_from_high_quality = True

            if bad_celebs_exam and main_celeb in take_from_high_quality_internet:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet/{main_celeb.replace(' ', '-')}/submarine_best_subset_{main_celeb}.json"
            elif bad_celebs_exam:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/all_internet_images/{main_celeb.replace(' ', '-')}/submarine_best_subset_{main_celeb}.json"
            else:
                best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/insightface_resnet100_embeddings_close_up_1/all_images_internet_similarity_matrix/{main_celeb}/submarine_best_subset_{main_celeb}.json"
            
        elif args.set_of_people == "politicians":
            best_subset_file = f"/gscratch/h2lab/vsahil/vlm-efficiency/facial_detection_and_recognition/insightface_resnet100_embeddings_close_up_politicians_1/images_I_downloaded_from_internet_similarity_matrix/{main_celeb}/submarine_best_subset_{main_celeb}.json"
        
        else: 
            raise NotImplementedError
            
        if not os.path.exists(best_subset_file):
            print(f"celeb {main_celeb} does not have a best subset file")
            assert False

        with open(best_subset_file, "r") as f:
            data = json.load(f)
            selected_indices = data["set"]
        
        if len(selected_indices) < 10:
            if args.set_of_people == "celebrity":
                if not bad_celebs_exam:
                    assert len(selected_indices) == 0       ## this is used because with submarine we explicitly set the minimum size of the subset to be 10.
            print(f"celeb {main_celeb} has {len(selected_indices)} images, which is less than 10")
        
        selected_image_embeddings, final_selected_images = get_best_images_and_compute_similarity(args, main_celeb, selected_indices, bad_celebs_exam, take_from_high_quality_internet=take_from_high_quality, 
                                        print_file=print_file, get_selected_image_embeddings=get_selected_image_embeddings)

        # print("selected_image_embeddings shape: ", selected_image_embeddings.shape)
        # import ipdb; ipdb.set_trace()
        
        if args.set_of_people == "caption_assumption":
            face_counts_using_selected_images_in_random_laion_images(args, celeb_processing, selected_image_embeddings, min_threshold=args.threshold_for_max_f1_score)
            continue

        if get_selected_image_embeddings and not compute_similarity_for_different_people:
            assert args.all_laion_images or args.all_laion_alias_name_images or args.set_of_people == "caption_assumption"      ## we would want to find the images in these two sets that match a person's face, or in the random set for the experiment of correctness of assumption. 
            num_images_with_similarity_greater_than_threshold, face_area_this_celeb = get_face_counts_using_selected_images_and_laion_images(args, celeb_processing, selected_image_embeddings, min_threshold=args.threshold_for_max_f1_score)
            print(f"number of face counts for {celeb_processing} is: ", num_images_with_similarity_greater_than_threshold)
            num_face_counts[celeb_processing] = num_images_with_similarity_greater_than_threshold
            face_areas_celebs[celeb_processing] = face_area_this_celeb

        if get_selected_image_embeddings:
            normalized_selected_image_embeddings = selected_image_embeddings / np.linalg.norm(selected_image_embeddings, axis=1)[:, np.newaxis]
            image_embeddings_each_celeb_for_selected_images[celeb_processing] = normalized_selected_image_embeddings

    if args.set_of_people == "caption_assumption":
        return

    
    if compute_similarity_for_different_people:
        avg_cosine_similarities_different_people = {}
        ## compute the cosine similarity between the embeddings of the images of different people. 
        for i in range(len(full_celeb_list)):
            for j in range(i+1, len(full_celeb_list)):
                celeb1 = full_celeb_list[i]
                celeb2 = full_celeb_list[j]
                assert celeb1 != celeb2
                assert (celeb1, celeb2) not in avg_cosine_similarities_different_people and (celeb2, celeb1) not in avg_cosine_similarities_different_people
                embeddings1 = image_embeddings_each_celeb_for_selected_images[celeb1]
                embeddings2 = image_embeddings_each_celeb_for_selected_images[celeb2]
                cosine_similarity_matrix = np.dot(embeddings1, embeddings2.T)
                assert cosine_similarity_matrix.shape[0] == len(embeddings1) and cosine_similarity_matrix.shape[1] == len(embeddings2)
                assert np.all(cosine_similarity_matrix >= -1.0 - 1e-6) and np.all(cosine_similarity_matrix <= 1.0 + 1e-6)
                avg_cosine_similarity = np.mean(cosine_similarity_matrix)
                avg_cosine_similarities_different_people[(celeb1, celeb2)] = avg_cosine_similarity
            print("Done with ", celeb1)
        print("Done with all celebs")
        ## now put this in file that we will use later to add to the histogram of the cosine similarities
        with open(f"selected_images_{args.set_of_people}_avg_cosine_similarity_different_people.csv", "w") as f:
            f.write("celeb1|celeb2|avg_cosine_similarity\n")
            for celeb1, celeb2 in avg_cosine_similarities_different_people:
                f.write(f"{celeb1}|{celeb2}|{avg_cosine_similarities_different_people[(celeb1, celeb2)]}\n")
        return

    exit()
    if get_selected_image_embeddings and not compute_similarity_for_different_people:
        if args.all_laion_images:
            if args.set_of_people == "celebrity":
                ## we also want to print the caption counts for each celeb
                celeb_caption_count = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrities_sorted.csv")        ## columns: Name,counts_in_laion2b-en,first_name,first_name_counts_in_laion2b-en. Read the name and counts_in_laion2b-en columns
                ## we also need to multply the num face counts by total_matching_captions / num of downloaded images of that celeb. 
                downloaded_images_directory = "/gscratch/h2lab/vsahil/vlm-efficiency/all_downloaded_images"
            elif args.set_of_people == "politicians":
                celeb_caption_count = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")
                downloaded_images_directory = "/gscratch/scrubbed/vsahil/all_downloaded_images_politicians"
            else: raise NotImplementedError
        
        elif args.all_laion_alias_name_images:
            assert args.set_of_people == "celebrity"
            alias_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrity_counts_with_aliases.csv", sep="|")
            all_aliases_name_and_count = alias_df['aliases_and_counts'].tolist()       ## this is a column where everything is a dictionary (as a string)
            all_aliases_name_and_count = [eval(alias) for alias in all_aliases_name_and_count]
            ## this is a list of dicts, each dict has the alias name as the key and the count of the alias name as the value
            ## merge all the dictionaries into one dictionary
            all_aliases_name_and_count = {k: v for alias_dict in all_aliases_name_and_count for k, v in alias_dict.items()}
            downloaded_images_directory = "/gscratch/scrubbed/vsahil/all_downloaded_images/"
        
        else:
            raise NotImplementedError
        
        if args.all_laion_images:
            num_face_counts_file = f"{base_embeddings_directory}/num_face_counts_all_laion_images_{args.set_of_people}_threshold_{args.threshold_for_max_f1_score}.csv"
        elif args.all_laion_alias_name_images:
            num_face_counts_file = f"{base_embeddings_directory}/num_face_counts_all_laion_alias_name_images_{args.set_of_people}_threshold_{args.threshold_for_max_f1_score}.csv"
        else:
            raise NotImplementedError
        
        with open(num_face_counts_file, file_open_mode) as f:
            if args.all_laion_images:
                f.write("celeb|num_face_counts|num_downloaded_images|matching_captions|effective_num_face_counts|face_area|effective_face_area\n")
            elif args.all_laion_alias_name_images:
                f.write("celeb|alias_name|num_face_counts|num_downloaded_images|matching_captions|effective_num_face_counts|face_area|effective_face_area\n")
            
            for celeb_processing in num_face_counts:
                if args.all_laion_images:
                    matching_captions = celeb_caption_count[celeb_caption_count['Name'] == celeb_processing]['counts_in_laion2b-en'].values[0]
                    downloaded_images_this_celeb_directory = f"{downloaded_images_directory}/{celeb_processing}"
                elif args.all_laion_alias_name_images:
                    # matching_captions = all_aliases_name_and_count[celeb_processing]
                    ## in this case celeb_processing is a tuple of (celeb, alias_name)
                    matching_captions = all_aliases_name_and_count[celeb_processing[1]]
                    downloaded_images_this_celeb_directory = f"{downloaded_images_directory}/{celeb_processing[1]}"
                
                num_downloaded_images = len([x for x in os.listdir(downloaded_images_this_celeb_directory) if not x.endswith(".csv")])
                try:
                    assert num_downloaded_images == len(os.listdir(f"{downloaded_images_this_celeb_directory}")) - 1         ## there is 1 .csv file in the directory
                except:
                    print(f"EXCEPTION1 celeb: {celeb_processing}, num_downloaded_images: {num_downloaded_images}, length of images directory: {len(os.listdir(f'{downloaded_images_this_celeb_directory}'))}")
                if num_downloaded_images == 0:
                    effective_num_face_counts = num_face_counts[celeb_processing]
                    effective_face_area_this_celeb = face_areas_celebs[celeb_processing]
                else:
                    effective_num_face_counts = num_face_counts[celeb_processing] * matching_captions / num_downloaded_images
                    effective_num_face_counts = int(effective_num_face_counts + 0.5)
                    effective_face_area_this_celeb = face_areas_celebs[celeb_processing] * matching_captions / num_downloaded_images
                    effective_face_area_this_celeb = int(effective_face_area_this_celeb + 0.5)
                effective_face_area_this_celeb /= 1000000.0
                if args.all_laion_images:
                    f.write(f"{celeb_processing}|{num_face_counts[celeb_processing]}|{num_downloaded_images}|{matching_captions}|{effective_num_face_counts}|{face_areas_celebs[celeb_processing]}|{effective_face_area_this_celeb}\n")
                elif args.all_laion_alias_name_images:
                    f.write(f"{dict_alias_key_celeb_value[celeb_processing]}|{celeb_processing}|{num_face_counts[celeb_processing]}|{num_downloaded_images}|{matching_captions}|{effective_num_face_counts}|{face_areas_celebs[celeb_processing]}|{effective_face_area_this_celeb}\n")
                try:
                    assert num_downloaded_images <= matching_captions
                except:
                    print(f"EXCEPTION2 celeb: {celeb_processing}, num_downloaded_images: {num_downloaded_images}, matching_captions: {matching_captions}")
        return 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_of_people", type=str, default=None, choices=["celebrity", "politicians", "cub_birds", "celebrity_clip_score", "celebrity_fid_distance", "artists_group1", "artists_group2", "caption_assumption"], required=True, help="Set of people to work with. celebrity_clip_score and celebrity_fid_distance are only for plotting the scores for paper. There is no other processing for it in this file. ")
    parser.add_argument("--celebrity", type=str, default=None, help="Name of the celebrity to detect faces for the training set")
    parser.add_argument('--training', action='store_true', help='generate questions for training images')
    parser.add_argument('--generated', action='store_true', help='generate questions for generated images')
    parser.add_argument('--image_generation_prompt_id', type=int, default=None, help='prompt id for image generation. We are using this to generate deep face embeddings for them. -99 means we want to plot the average cosine similarities for all the prompts in one plot. ', choices=[0, 1, 2, 3, 4, -99])
    parser.add_argument('--images_I_downloaded_from_internet', action='store_true', help='these are the images I downloaded from internet for celebs with less training images')
    parser.add_argument('--all_laion_images', action='store_true', help='these are the images from laion2b-en for a particular celeb')
    parser.add_argument('--all_laion_alias_name_images', action='store_true', help='these are the images of the alias names of the celebs')
    parser.add_argument('--random_laion_images', action='store_true', help='use to test the correctness of the assumption of caption choice only. ')
    parser.add_argument("--consider_name_aliases", action='store_true', help='consider the alias names of the celebs', default=False)
    
    parser.add_argument('--all_images_internet', action='store_true', help='these are the images from internet for a particular celeb. It is just a different directory. ')
    parser.add_argument('--only_plot_face_detection', action='store_true', help='only plot the results')

    parser.add_argument('--face_detection', action='store_true', help='for the task of facial recognition')
    parser.add_argument('--select_single_person_and_large_training_images', action='store_true', help='select high quality training images')
    parser.add_argument('--select_real_photographs_from_single_person_and_large_training_images', action='store_true', help='select real photographs from high quality training images')
    parser.add_argument('--face_recognition_for_high_quality_training_images', action='store_true', help='perform face recognition on high quality training images')
    parser.add_argument('--run_face_detection_parallel', action='store_true', help='run face detection in parallel')
    
    parser.add_argument("--parallelize_across_entities", action='store_true', help='parallelize the face detection or face embedding generation across entities')
    parser.add_argument("--parallelize_across_one_entity", action='store_true', help='parallelize the face detection or face embedding generation across one entity')
    parser.add_argument("--sequentialize_across_entities", action='store_true', help='sequentialize the face detection or face embedding generation across entities')

    parser.add_argument('--demo_only_face_detection', action='store_true', help='only run the demo example images')
    parser.add_argument('--demo_face_recognition', action='store_true', help='only run the demo example images')
    parser.add_argument('--generate_embeddings_face_recognition', action='store_true', help='generate embeddings for all images for training or generated images')
    parser.add_argument('--generate_face_embeddings_parallel', action='store_true', help='generate embeddings for all images for training or generated images in parallel using CPUs')
    parser.add_argument('--match_embeddings_face_recognition', action='store_true', help='match generated images to selected training images')
    parser.add_argument('--store_similarity_matrix', action='store_true', help='This is used for performing submodular minimization objective. This stores the similarity between each pair of images (for now images from internet)')
    parser.add_argument('--face_recognition_to_compare_training_and_generated_images', action='store_true', help='for the task of facial recognition')
    parser.add_argument('--top_k_training_images', default=20, type=int, help='top k training images to compare with generated images')
    parser.add_argument('--select_training_images_based_on_similarity_to_generated_images', action='store_true', help='select training images most similar to the generated images, instead of the top-k training images. This might be useful for the scenarios where the model generates images similar to only a subset of the training images. ')
    parser.add_argument('--select_training_images_based_on_similarity_to_generated_images_on_average', action='store_true', help='select training images most similar to the generated images, instead of the top-k training images. This might be useful for the scenarios where the model generates images similar to only a subset of the training images. The difference with the previous option is that we first select the training images based on similarity to all generated images. ')
    parser.add_argument('--use_selected_images_for_matching', action='store_true', help='use the selected images from the internet for matching to the generated images instead of the top-k training images')
    parser.add_argument('--use_filtered_laion_images_and_selected_images_for_matching', action='store_true', help='use the filtered laion images and selected images from the internet for matching to the generated images instead of the top-k training images')
    parser.add_argument('--plot_similarity_training_and_generated_images', action='store_true', help='plot the similarity between training and generated images')
    parser.add_argument('--use_only_isotonic_regression', action='store_true', help='use isotonic regression for the plot')
    parser.add_argument('--use_isotonic_and_change_detection', action='store_true', help='use isotonic regression for the plot and also use change detection to find the change points')
    parser.add_argument('--separate_plot_each_race', action='store_true', help='if this is true, we will generate separate plots for similarity between training and generated images')

    parser.add_argument('--tokenize_image_captions', action='store_true', help='tokenize image captions')
    parser.add_argument('--clustering_training_image_embeddings', action='store_true', help='clustering training image embeddings, which will be added to the plot. ')
    parser.add_argument('--view_good_training_images', action='store_true', help='view the top-20 good training images for each celeb')
    parser.add_argument('--stats_from_similarity_data', action='store_true', help='generate stats from similarity data')
    parser.add_argument('--polynomial_degree', type=int, default=1, help='polynomial degree for the polynomial fit')
    parser.add_argument('--face_recognition_to_compare_training_images_of_different_groups', action='store_true', help='we will use this function to compare the training images of different groups of people -- like white/brown/black males/females. This is to understand the biases of the face recognition model. ')
    parser.add_argument('--test_face_recognition_models_for_bias', action='store_true', help='we will use this test other face recognition models for bias. ')
    parser.add_argument('--face_recognition_model', type=str, default=None, help='face recognition model to use', choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "Amazon_Rekognition", 'insightface_buffalo', 'insightface_resnet100', 'GhostFaceNetV1-0.5-2_A','GhostFaceNetV1-1.3-1_C', 'GhostFaceNet_W1.3_S1_ArcFace'])
    parser.add_argument('--test_model_parallel', action='store_true', help='test the face recognition models in parallel for the celebs')
    parser.add_argument('--metric_fmr', action='store_true', help='If true, we measure the FMR of the different face recognition models')
    parser.add_argument('--metric_fnmr', action='store_true', help='If true, we measure the FNMR of the different face recognition models')
    parser.add_argument('--plot_stacked_bar_chart_for_face_recognition_performance', action='store_true', help='plot the stacked bar chart for the face recognition performance. If this is true, we are not going to generate any metrics, we will just plot the results. ')
    parser.add_argument('--get_celebrity_gender_and_race', action='store_true', help='Get the gender and race classification for the celebs from their training images, and we will use that to produce separate plots for the celebrities')
    parser.add_argument('--prepare_montage_selected_images', action='store_true', help='This is used for visualizing the images that submodular min selects. ')
    parser.add_argument('--compute_similarity_selected_images', action='store_true', help='This is used for computing the similarity between the images selected using submodular min for each celeb. ')
    parser.add_argument('--same_face_threshold', type=str, default='lower', choices=['lower', 'higher'], help='If lower, we use the mid point of the classifier around 0.46 and if higher we use the minimum point of the classifier, around 0.56')
    
    parser.add_argument('--num_cpus', type=int, default=None, help='If you do not use SLURM, choose the number of CPUs you can use')
    parser.add_argument('--insightface_model_path', type=str, default='/mmfs1/home/vsahil/.insightface/models/resnet-100.onnx', help='Path to the insightface model')
    parser.add_argument("--ablation_experiment", type=str, help="If true, that means we are running an ablation experiment and the paths for the reference images need to be changed. ", default=None, choices=["SD1.5", "SD2.1"])
    
    parser.add_argument("--stable_diffusion_model", type=str, choices=["1", "5", "v2"], default=None, help="If we are using the stable diffusion model, we need to specify the version of the model. ")
    parser.add_argument("--project_root_directory", type=str, default=None) # "/gscratch/h2lab/vsahil/vlm-efficiency"
    args = parser.parse_args()
    

    assert sum([args.generate_face_embeddings_parallel, args.match_embeddings_face_recognition, args.store_similarity_matrix]) <= 1, 'Please specify only one of --generate_face_embeddings_parallel, --match_embeddings_face_recognition, --store_similarity_matrix'
    
    if args.generated:
        assert args.image_generation_prompt_id is not None, 'Please specify --image_generation_prompt_id'

    if args.demo_only_face_detection:
        assert args.face_recognition_model is not None, 'The face recognition model we use will affect the face detection'
        demo_example_images_face_detection()
        exit()

    if args.demo_face_recognition:
        assert args.face_recognition_model is not None, 'Please specify --face_recognition_model'
        demo_face_recognition(args)
        exit()

    if args.tokenize_image_captions:
        tokenize_image_captions(args)
        exit()

    if args.clustering_training_image_embeddings:
        clustering_training_image_embeddings(args, plot_images_downloaded_from_internet=False)
        exit()

    if args.view_good_training_images:
        view_good_training_images(args)
        exit()

    if args.stats_from_similarity_data:
        stats_from_similarity_data(args)
        exit()

    if args.compute_similarity_selected_images:
        compute_similarity_selected_images(args)
        exit()

    if not args.face_recognition_to_compare_training_and_generated_images and not args.plot_similarity_training_and_generated_images:
        # assert args.training or args.generated or args.images_I_downloaded_from_internet, 'Please specify either --training or --generated'
        assert sum([args.training, args.generated, args.images_I_downloaded_from_internet, args.all_laion_images, args.all_images_internet, args.all_laion_alias_name_images]) == 1, 'Please specify either --training or --generated or --images_I_downloaded_from_internet'


    if args.get_celebrity_gender_and_race:
        def worker(celebrity_here, return_dict):
            # Your processing here
            # Instead of writing to a file, store the result in the return_dict
            result = get_celebrity_gender_and_race(args, celeb_here=celebrity_here)
            return_dict[celebrity_here] = result

        if __name__ == "__main__":
            manager = Manager()
            return_dict = manager.dict()

            celebrity_list_remaining = celebrities_with_few_images + celebrity_list + celebrity_list0
            pool = Pool(processes=min(get_slurm_cpus(), len(celebrity_list_remaining)))
            for celebrity in celebrity_list_remaining:
                pool.apply_async(worker, args=(celebrity, return_dict))
            pool.close()
            pool.join()

            # Now, write to the file in a single process
            with open("celebrity_predicted_race_and_gender.csv", "w") as f:
                f.write(f'celeb|gender|race\n')
                for celeb, result in return_dict.items():
                    f.write(f"{celeb}|{result[0]}{result[1]}\n")


    elif args.run_face_detection_parallel:
        assert args.face_recognition_model is not None, 'The face recognition model we use will affect the face detection' 
        from functools import partial

        def worker_cpu(celebrity_here, output_file_statistics, output_file_each_image_info, model):
            scaled_up_face_detection(args, celeb_here=celebrity_here, output_file_statistics=output_file_statistics, output_file_each_image_info=output_file_each_image_info, model=model)
        
        def worker_gpu(gpu_id, data_subset, output_file_statistics, output_file_each_image_info):
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                model = None
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                from insightface.app import FaceAnalysis
                insightfaceapp = FaceAnalysis(providers=[('CUDAExecutionProvider', {'device_id': gpu_id})])
                insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                model = insightfaceapp
            else:
                raise Exception('Please specify a valid face recognition model')
            for celebrity_here in data_subset:
                scaled_up_face_detection(args, celeb=celebrity_here, output_file_statistics=output_file_statistics, output_file_each_image_info=output_file_each_image_info, model=model)
        
        parallelize_across_cpus = False
        assert args.parallelize_across_entities or args.parallelize_across_one_entity
        ## check the cuda version and if it is not 11.6, then kill the process with a warning message.
        if args.parallelize_across_one_entity or args.parallelize_across_entities:     ## check for the correct CUDA version if using GPU
            ## get the cuda version
            cuda_version = os.popen('nvcc --version').read().split('\n')[3].split(',')[1].split(' ')[2]
            if cuda_version != '11.6':
                print(f"Please use cuda version 11.6 for parallelizing across GPUs. Your cuda version is {cuda_version}")
                exit()
            else:
                print(f"Your cuda version is {cuda_version}. You can proceed with parallelizing across GPUs.")
        
        if __name__ == "__main__":
            politicians = pd.read_csv("../celebrity_data/sampled_politicians.csv")['Name'].tolist()

            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                model = None
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                if args.parallelize_across_entities or args.parallelize_across_one_entity:
                    ## here we will need to load the model separately in each gpu
                    model = None
                else:
                    from insightface.app import FaceAnalysis
                    insightfaceapp = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                    model = insightfaceapp
            else:
                raise Exception('Please specify a valid face recognition model')
            
            if args.training:
                output_file_statistics = f"result_faces_training_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_training_each_image_info_{args.set_of_people}.csv"
            elif args.generated:
                args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
                output_file_statistics = f"result_faces_generated_{args.image_generation_prompt}_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_generated_{args.image_generation_prompt}_each_image_info_{args.set_of_people}.csv"
            elif args.images_I_downloaded_from_internet:
                if args.set_of_people == "politicians":
                    politicians = pd.read_csv("entities_for_serpapi.csv")['Name'].tolist()
                output_file_statistics = f"result_faces_images_I_downloaded_from_internet_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_images_I_downloaded_from_internet_each_image_info_{args.set_of_people}.csv"
            elif args.all_images_internet:
                output_file_statistics = f"result_faces_all_images_internet_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_all_images_internet_each_image_info_{args.set_of_people}.csv"
            elif args.all_laion_images:
                output_file_statistics = f"result_faces_all_laion_images_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_all_laion_images_each_image_info_{args.set_of_people}.csv"
            elif args.all_laion_alias_name_images:
                output_file_statistics = f"result_faces_all_laion_alias_name_images_statistics_{args.set_of_people}.csv"
                output_file_each_image_info = f"result_faces_all_laion_alias_name_images_each_image_info_{args.set_of_people}.csv"
            else:
                raise Exception('Please specify either --training or --generated')

            # add header to the each image file: celeb|image|num_faces|face_area_percentages|sharpness|image_area
            # add header to the statistisc file: celeb|0 faces|1 face|2 faces|more than 2 faces|total images
            # add the header only if these files do not exist
            if not os.path.exists(output_file_statistics):
                with open(output_file_statistics, "w") as f:
                    f.write('celeb|0 faces|1 face|2 faces|more than 2 faces|total images\n')
            if not os.path.exists(output_file_each_image_info):
                with open(output_file_each_image_info, "w") as f:
                    f.write('celeb|image|num_faces|face_area_percentages|sharpness|image_area\n')
            
            if args.set_of_people == "celebrity" and not args.all_laion_alias_name_images:
                celebrity_list_remaining = celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities + celebrity_list
                random.shuffle(celebrity_list_remaining)
                print(celebrity_list_remaining, len(celebrity_list_remaining))

            elif args.set_of_people == "politicians":
                politicians_df = pd.read_csv(f"{args.project_root_directory}/celebrity_data/good_politicians_to_analyze.csv")   ## Name,counts_in_laion2b-en
                assert len(politicians_df) == 418
                if args.parallelize_across_entities:
                    celebrity_list_remaining = politicians_df[politicians_df['counts_in_laion2b-en'] < 1000]['Name'].tolist()
                elif args.parallelize_across_one_entity:
                    ## sort in increasing order of the number of images
                    politicians_df = politicians_df.sort_values(by='counts_in_laion2b-en', ascending=True)
                    celebrity_list_remaining = politicians_df[politicians_df['counts_in_laion2b-en'] >= 1000]['Name'].tolist()
                else:
                    pass
                random.shuffle(celebrity_list_remaining)
                print(celebrity_list_remaining, len(celebrity_list_remaining))

            if parallelize_across_cpus is True:
                cpus_here = args.num_cpus if args.num_cpus else get_slurm_cpus()
                print(f"Num politicians: {len(politicians)} | {cpus_here}")
                entities_scanned = set(pd.read_csv(output_file_statistics, sep="|").iloc[:,0].tolist())
                politicians = list(set(politicians).difference(entities_scanned))
                print(f"len entities: {entities_scanned}")
                print(f"Remaining politicians to work on: {len(politicians)}")

                pool = Pool(processes=min(cpus_here, len(politicians)))
                worker_with_files = partial(worker_cpu, output_file_statistics=output_file_statistics, output_file_each_image_info=output_file_each_image_info, model=model)
                if args.set_of_people == "celebrity":
                    pool.map(worker_with_files, celebrity_list_remaining)
                elif args.set_of_people == "politicians":
                    pool.map(worker_with_files, politicians)
                pool.close()
                pool.join()

            elif args.parallelize_across_entities is True:
                def split_data(data, num_splits):
                    """Splits data into sublists."""
                    k, m = divmod(len(data), num_splits)
                    return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits))
                
                num_gpus = get_slurm_gpus()
                assert num_gpus > 0, 'No GPUs allocated'
                data_to_process = celebrity_list_remaining if args.set_of_people == "celebrity" else politicians
                data_subsets = list(split_data(data_to_process, num_gpus))
                
                processes = []
                for gpu_id, data_subset in enumerate(data_subsets):
                    p = Process(target=worker_gpu, args=(gpu_id, data_subset, output_file_statistics, output_file_each_image_info))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
            
            else:
                ## this is not parallelized
                for celeb_here in celebrity_list_remaining:
                    scaled_up_face_detection(args, celeb=celeb_here, output_file_statistics=output_file_statistics, output_file_each_image_info=output_file_each_image_info, model=model)


    elif args.select_single_person_and_large_training_images:
        assert not args.face_detection and not args.face_recognition_to_compare_training_and_generated_images and not args.face_recognition_for_high_quality_training_images
        if args.set_of_people == "celebrity":
            celebrity_list_remaining = celebrities_with_few_images + celebrity_list + celebrity_list0 + to_be_added_celebrities
        elif args.set_of_people == "politicians":
            celebrity_list_remaining = pd.read_csv("../celebrity_data/sampled_politicians.csv")['Name'].tolist()
        select_single_person_and_large_training_images(args, celebrity_list_remaining)
        exit()


    elif args.select_real_photographs_from_single_person_and_large_training_images:
        assert not args.face_detection and not args.face_recognition_to_compare_training_and_generated_images
        select_real_photographs_from_single_person_and_large_training_images(args)
        exit()


    elif args.face_recognition_for_high_quality_training_images:
        assert not args.face_detection and not args.face_recognition_to_compare_training_and_generated_images
        assert args.face_recognition_model is not None, 'Please specify --face_recognition_model'
        
        if args.training:
            output_file = f"all_celebs_single_person_training_data_face_recognition_result_summarized_{args.set_of_people}.csv"
            second_output_file = f"all_celebs_single_person_training_data_face_recognition_result_{args.set_of_people}.csv"
        elif args.images_I_downloaded_from_internet:
            output_file = f"all_celebs_single_person_images_I_downloaded_from_internet_face_recognition_result_summarized_{args.set_of_people}.csv"
            second_output_file = f"all_celebs_single_person_images_I_downloaded_from_internet_face_recognition_result_{args.set_of_people}.csv"
        else:   
            raise NotImplementedError
        
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write('celeb|count_high_quality_images|total_single_person_images|num_downloaded_training_images_this_celeb\n')
        if not os.path.exists(second_output_file):
            with open(second_output_file, "w") as f:
                f.write('celeb|image|num_matches_this_image|effective_pixels|sharpness\n')

        do_not_write_file = False
        high_quality_images = {}
        total_images = {}
        # if args.set_of_people
        if args.set_of_people == "celebrity":
            celebrity_list = celebrities_with_few_images + celebrity_list + celebrity_list0 + to_be_added_celebrities
        elif args.set_of_people == "politicians":
            politicians = pd.read_csv("../celebrity_data/sampled_politicians.csv")['Name'].tolist()
            celebrity_list = politicians


        for celeb_ in celebrity_list:
            count_high_quality_images, count_total_images = face_recognition_for_high_quality_training_images(args, celeb_here=celeb_, output_file=output_file, second_output_file=second_output_file,  matching_threshold=0.4, do_not_write_file=do_not_write_file)
            high_quality_images[celeb_] = count_high_quality_images
            total_images[celeb_] = count_total_images
        
        celebs_to_download_internet_images = []
        for celeb_ in celebrity_list:
            if total_images[celeb_] == 0:
                print(f"Total images for {celeb_} is 0")
                celebs_to_download_internet_images.append(celeb_)
                continue
            if high_quality_images[celeb_] < 10:
                print(f"High quality images for {celeb_} is {high_quality_images[celeb_]}")
                celebs_to_download_internet_images.append(celeb_)
        
        if args.training:
            print(celebs_to_download_internet_images, len(celebs_to_download_internet_images))
            already_downloaded_celebs = [i.replace(".json", "") for i in os.listdir("insightface_resnet100_embeddings_close_up/images_I_downloaded_from_internet/") if i.endswith('.json')]
            missing_celebs = [celeb for celeb in celebs_to_download_internet_images if celeb not in already_downloaded_celebs]
            print("need to download: ", missing_celebs)


    elif args.only_plot_face_detection:
        assert not args.face_recognition and not args.face_detection
        with open(f"result_faces_{'training' if args.training else 'generated'}.json", "r") as f:
            output = json.load(f)
        plot_faces_per_image(args, output["no_face"], output["one_face"], output["two_face"], output["more_than_two_face"])


    elif args.face_recognition_to_compare_training_and_generated_images:
        assert not args.face_detection
        if args.match_embeddings_face_recognition:
            if args.set_of_people == "celebrity":
                all_people_list = celebrities_with_few_images + celebrity_list + celebrity_list0 + to_be_added_celebrities
            elif args.set_of_people == "politicians":
                politicians_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")   ## Name,counts_in_laion2b-en
                assert len(politicians_df) == 418
                politicians = politicians_df['Name'].tolist()
                all_people_list = politicians
            else:
                raise NotImplementedError
            print(len(all_people_list), f"for {args.set_of_people}")
            face_recognition_to_compare_training_and_generated_images(args, all_people_list=all_people_list, num_training_images_to_compare=10, use_selected_images_for_matching=args.use_selected_images_for_matching, use_filtered_laion_images_and_selected_images_for_matching=args.use_filtered_laion_images_and_selected_images_for_matching)
        else:
            face_recognition_to_compare_training_and_generated_images(args)


    elif args.generate_face_embeddings_parallel:
        def worker_cpu(celebrity_here):
            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                ## HEre the Deepface model will be loaded inside the function, not optimized as not using rn. 
                insightfaceapp = None
                insightface_recognition = None
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                import insightface
                from insightface.app import FaceAnalysis
                insightfaceapp = FaceAnalysis(providers=[('CPUExecutionProvider')])
                insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                if args.face_recognition_model == 'insightface_resnet100':
                    insightface_recognition = insightface.model_zoo.get_model(args.insightface_model_path)     ## here we are going to use a different model for face recognition.
                    insightface_recognition.prepare(ctx_id=0)
            face_recognition_to_compare_training_and_generated_images(args, celebrity_here=celebrity_here, only_generate_embeddings=True, insightfaceapp=insightfaceapp, insightface_recognition=insightface_recognition)

        def worker_gpu(gpu_id, data_subset):
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                ## HEre the Deepface model will be loaded inside the function, not optimized as not using rn. 
                insightfaceapp = None
                insightface_recognition = None
            elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                import insightface
                from insightface.app import FaceAnalysis
                insightfaceapp = FaceAnalysis(providers=[('CUDAExecutionProvider', {'device_id': gpu_id})])
                insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                if args.face_recognition_model == 'insightface_resnet100':
                    insightface_recognition = insightface.model_zoo.get_model(args.insightface_model_path)     ## here we are going to use a different model for face recognition.
                    insightface_recognition.prepare(ctx_id=0)
                
            for celebrity_here in data_subset:
                face_recognition_to_compare_training_and_generated_images(args, celebrity_here=celebrity_here, only_generate_embeddings=True, insightfaceapp=insightfaceapp, insightface_recognition=insightface_recognition)

        def celebs_already_done(celebrity_list):
            remove_list = []
            orig_celebrity_list = celebrity_list.copy()
            base_embeddings_directory = f"{args.face_recognition_model}_embeddings_close_up_{args.stable_diffusion_model}" if args.set_of_people == "celebrity" else f"{args.face_recognition_model}_embeddings_close_up_politicians_{args.stable_diffusion_model}"
            for celebrity_here in celebrity_list:
                if args.generated:
                    args.image_generation_prompt, args.image_generation_prompt_type = set_of_prompts_human_face[args.image_generation_prompt_id]
                    if os.path.exists(f"{base_embeddings_directory}/{args.image_generation_prompt}/{celebrity_here}.json"):
                        remove_list.append(celebrity_here)
                elif args.training or args.images_I_downloaded_from_internet or args.all_images_internet or args.all_laion_images or args.all_laion_alias_name_images:
                    datatype_here = 'training' if args.training else 'images_I_downloaded_from_internet' if args.images_I_downloaded_from_internet else 'all_images_internet' if args.all_images_internet else 'all_laion_images' if args.all_laion_images else 'all_laion_alias_name_images'
                    if os.path.exists(f"{base_embeddings_directory}/{datatype_here}/{celebrity_here}.json"):
                        remove_list.append(celebrity_here)
                else:
                    raise NotImplementedError
            for celeb in remove_list:
                celebrity_list.remove(celeb)
            assert len(celebrity_list) == len(orig_celebrity_list) - len(remove_list)
            return celebrity_list

        if __name__ == "__main__":
            parallelize_across_cpus = False
            ## get the cuda version
            cuda_version = os.popen('nvcc --version').read().split('\n')[3].split(',')[1].split(' ')[2]
            if cuda_version != '11.6':
                print(f"Please use cuda version 11.6 for parallelizing across GPUs. Your cuda version is {cuda_version}")
                exit()
            else:
                print(f"Your cuda version is {cuda_version}. You can proceed with parallelizing across GPUs.")
            ## when args.parallelize_across_entities is true, we process celebs with less than 1000 images. 
            ## when args.parallelize_across_one_entity is true, we process celebs with more than 1000 images. This is max optimized performance. 
            assert args.parallelize_across_entities or args.parallelize_across_one_entity or args.sequentialize_across_entities, 'Please specify either --parallelize_across_entities or --parallelize_across_one_entity'
            assert args.stable_diffusion_model is not None, 'Please specify --stable_diffusion_model'
            
            if args.set_of_people == "celebrity" and not args.all_laion_alias_name_images:
                celebrity_list_remaining = celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities + celebrity_list
                # celebrity_list_remaining = ['Bridgers', 'Dynevor', 'Hellqvist', 'Speedman', 'Sartorius', 'Stegner', 'Biermann', 'Boneta', 'Sawa', 'Gacy', 'Ryan', 'Thomas', 'Simpson', 'Leslie', 'Russell', 'Maxwell', 'Harvey', 'Buffer', 'Price', 'Evans'][::-1]
                celebrity_list_remaining = celebs_already_done(celebrity_list_remaining)
                random.shuffle(celebrity_list_remaining)
                # celebrity_list_remaining = ['Cacee Cobb', 'Sofia Hellqvist']      ## outliers of the type 2. 
                print(celebrity_list_remaining, len(celebrity_list_remaining))
                
            elif args.set_of_people == "celebrity" and args.all_laion_alias_name_images:
                alias_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrity_counts_with_aliases.csv", sep="|")     ##celeb|full_name_count|aliases_and_counts|alias_total_count|Outlier
                all_aliases = alias_df['aliases_and_counts'].tolist()       ## this is a column where everything is a dictionary (as a string)
                all_aliases = [eval(alias) for alias in all_aliases]        ## convert the string to a dictionary
                assert [type(alias) for alias in all_aliases] == [dict]*len(all_aliases)
                if args.parallelize_across_entities:
                    ## here get the aliaes that have less than 1000 images. Note that all_aliases is a list of dict where the key is the alias and the value is its count.
                    all_aliases = [alias for alias_dict in all_aliases for alias in alias_dict.keys() if alias_dict[alias] < 900]
                elif args.parallelize_across_one_entity:
                    ## here get the aliaes that have more than 1000 images. Note that all_aliases is a list of dict where the key is the alias and the value is its count.
                    all_aliases = [alias for alias_dict in all_aliases for alias in alias_dict.keys() if alias_dict[alias] >= 900]
                else:
                    all_aliases = [alias for alias_dict in all_aliases for alias in alias_dict.keys()]
                print(len(all_aliases))
                # all_aliases = ['Bridgers', 'Dynevor', 'Hellqvist', 'Speedman', 'Sartorius', 'Stegner', 'Biermann', 'Boneta', 'Sawa', 'Gacy', 'Ryan', 'Thomas', 'Simpson', 'Leslie', 'Russell', 'Maxwell', 'Harvey', 'Buffer', 'Price', 'Evans'][::-1]
                all_aliases = ['Danneel', 'Normani', 'Penélope', 'Avani', 'Cacee', 'Simu', 'Michiel', 'Léa', 'Skai', 'Selita', 'Forest', 'Samuel', 'David', 'Joe', 'Nick', 'Nicole', 'Steve', 'John', 'Cody', "Olivia"]
                all_aliases = celebs_already_done(all_aliases)
                print(len(all_aliases), "to process now")
                print(all_aliases)
                celebrity_list_remaining = all_aliases
            
            elif args.set_of_people == "politicians":
                politicians_df = pd.read_csv(f"{args.project_root_directory}/celebrity_data/good_politicians_to_analyze.csv")   ## Name,counts_in_laion2b-en
                assert len(politicians_df) == 418
                if args.parallelize_across_entities:
                    celebrity_list_remaining = politicians_df[politicians_df['counts_in_laion2b-en'] < 1000]['Name'].tolist()
                elif args.parallelize_across_one_entity:
                    ## sort in increasing order of the number of images
                    politicians_df = politicians_df.sort_values(by='counts_in_laion2b-en', ascending=True)
                    celebrity_list_remaining = politicians_df[politicians_df['counts_in_laion2b-en'] >= 1000]['Name'].tolist()
                else:
                    pass
                celebrity_list_remaining = celebs_already_done(celebrity_list_remaining)
                # random.shuffle(celebrity_list_remaining)
                print(celebrity_list_remaining, len(celebrity_list_remaining))
            
            elif args.set_of_people == "caption_assumption":
                celebrity_list_remaining = ["and", "the", "a", "an", "in", "for", "on", "at", "by", "to", "of", "it", "as"]
            
            if len(celebrity_list_remaining) == 0:
                print("All celebs have been processed")
                exit()

            if parallelize_across_cpus:
                cpus_here = args.num_cpus if args.num_cpus else get_slurm_cpus() // 2
                print(f"Num people to process: {len(celebrity_list_remaining)} | {cpus_here}")
                pool = Pool(processes = min(cpus_here, len(celebrity_list_remaining)))
                pool.map(worker_cpu, celebrity_list_remaining)  # maps the worker function to each celebrity
                pool.close()  # no more tasks will be submitted to the pool
                pool.join()  # wait for the worker processes to exit

            elif args.parallelize_across_entities:
                def split_data(data, num_splits):
                    # """Splits data into sublists."""
                    # k, m = divmod(len(data), num_splits)
                    # return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits))
                    """Splits data into sublists in a round-robin fashion. So that the first elements in the overall list is processed first. """
                    return [data[i::num_splits] for i in range(num_splits)]
                
                num_gpus = get_slurm_gpus()
                assert num_gpus > 0, 'No GPUs allocated'
                print(f"Num people to process: {len(celebrity_list_remaining)} | {num_gpus} GPUs")
                data_subsets = list(split_data(celebrity_list_remaining, num_gpus))
                print(f"Num people to process: {len(celebrity_list_remaining)} | {num_gpus} GPUs | {len(data_subsets)} subsets | {len(data_subsets[0])} celebs per subset")
                
                processes = []
                for gpu_id, data_subset in enumerate(data_subsets):
                    p = Process(target=worker_gpu, args=(gpu_id, data_subset))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
            
            elif args.parallelize_across_one_entity or args.sequentialize_across_entities:
                if args.face_recognition_model in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']:
                    insightfaceapp = None
                    insightface_recognition = None
                elif args.face_recognition_model in ['insightface_buffalo', 'insightface_resnet100']:
                    if args.sequentialize_across_entities:
                        import insightface
                        from insightface.app import FaceAnalysis
                        insightfaceapp = FaceAnalysis(providers=[('CUDAExecutionProvider', {'device_id': 0})])
                        insightfaceapp.prepare(ctx_id=0, det_size=(640, 640))
                        if args.face_recognition_model == 'insightface_resnet100':
                            insightface_recognition = insightface.model_zoo.get_model(args.insightface_model_path)     ## here we are going to use a different model for face recognition.
                            insightface_recognition.prepare(ctx_id=0)
                    elif args.parallelize_across_one_entity:
                        insightfaceapp = None
                        insightface_recognition = None
                
                for celeb_here in celebrity_list_remaining:
                    face_recognition_to_compare_training_and_generated_images(args, celebrity_here=celeb_here, only_generate_embeddings=True, insightfaceapp=insightfaceapp, insightface_recognition=insightface_recognition)


    elif args.store_similarity_matrix:
        face_recognition_to_compare_training_and_generated_images(args, only_generate_embeddings=False)


    elif args.prepare_montage_selected_images:
        selected_indices = "all"
        # selected_indices = [1,2,4,5,6,7,8,11,12,13,14,16,17,18,19,20,22,25,27,30,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,49,51,52,53,54,58,60,63,65,69]
        prepare_montage_selected_images(args, celeb_here=args.celebrity, selected_indices=selected_indices)


    elif args.plot_similarity_training_and_generated_images:
        assert not args.face_detection and not args.face_recognition_to_compare_training_and_generated_images
        if args.separate_plot_each_race:
            for gender in ['Man', 'Woman']:
                for race in ['White', 'Brown', 'Black']:
                    plot_similarity_training_and_generated_images(args, sort_according_to_quality=False, sort_according_to_effective_pixels=False, separate_plot_each_race=(gender, race))
        else:
            sort_according_to_face_count = True if args.set_of_people in ["celebrity", "politicians", "artists_group1", "artists_group2"] else False      ## art styles is sorted by the total number of actual paintings of that artist. 
            plot_similarity_training_and_generated_images(args, sort_according_to_quality=False, sort_according_to_effective_pixels=False, separate_plot_each_race=False, sort_according_to_face_count=sort_according_to_face_count, sort_according_to_total_face_pixels=False)
        exit()


    elif args.face_recognition_to_compare_training_images_of_different_groups:
        face_recognition_to_compare_training_images_of_different_groups(args)


    elif args.test_face_recognition_models_for_bias:
        if not args.plot_stacked_bar_chart_for_face_recognition_performance:
            assert args.face_recognition_model is not None, 'Please specify --face_recognition_model'
        assert sum([args.metric_fmr, args.metric_fnmr]) == 1, 'Please specify either --metric_fmr or --metric_fnmr'
        test_face_recognition_models_for_bias(args)


    else:
        raise Exception('Please specify either --face_recognition or --face_detection')

