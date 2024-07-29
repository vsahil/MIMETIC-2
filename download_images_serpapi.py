# from serpapi import GoogleSearch
import serpapi, json, os
import requests
from PIL import Image
from io import BytesIO


def create_json_file_for_celebrity(args):
    base_directory = "serpapi_json_files" if args.set_of_people == "celebrity" else "serpapi_json_files_politicians" if args.set_of_people == "politicians" else None
    os.makedirs(base_directory, exist_ok=True)
    save_file_results = f"{base_directory}/results_{args.celebrity_name.replace(' ', '-')}.json"
        
    if os.path.exists(save_file_results):
        print(f"File {save_file_results} already exists. Skipping...")
        return
    
    if os.path.exists(f"{args.celebrity_name.replace(' ', '-')}/"):
        print(f"Directory {args.celebrity_name.replace(' ', '-')} already exists. Skipping...")
        return
    
    params = {
        "engine": "google_images",
        "q": args.celebrity_name,
        "location": "Austin, TX, Texas, United States",
        "api_key": ""       ## put your API key here
        }

    search = serpapi.search(params)
    results = search.as_dict()
    ## save the results in a json file.
    with open(save_file_results, 'w') as f:
        json.dump(results, f)
    print("created json file for", args.celebrity_name)


def download_images_from_json(args, delete_directory_and_move_selected_images_directory=False, do_not_skip=False):
    download_base_folder = "/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet" if args.set_of_people == "celebrity" else "/gscratch/h2lab/vsahil/vlm-efficiency/high_quality_images_downloaded_from_internet_politicians" if args.set_of_people == "politicians" else None
    os.makedirs(download_base_folder, exist_ok=True)
    download_folder = f"{download_base_folder}/{args.celebrity_name.replace(' ', '-')}"
    
    if delete_directory_and_move_selected_images_directory:
        ## only delete the directory if the directory to be copied from has more than 1 image.
        if os.path.exists(f"../facial_detection_and_recognition/good_images_each_celeb/{args.celebrity_name.replace(' ', '-')}") and len(os.listdir(f"../facial_detection_and_recognition/good_images_each_celeb/{args.celebrity_name.replace(' ', '-')}")) > 1:
            os.system(f"rm -rf {download_folder}")
            print(f"Deleted {download_folder}")
            os.system(f"cp -r ../facial_detection_and_recognition/good_images_each_celeb/{args.celebrity_name.replace(' ', '-')} .")
            print(f"copied the good images for {args.celebrity_name} to the current directory")
            return
        else:
            print(f"Directory ../facial_detection_and_recognition/good_images_each_celeb/{args.celebrity_name.replace(' ', '-')} does not exist or has only 1 image. Skipping...")
            return
    
    serpapi_json_files_folder = "serpapi_json_files" if args.set_of_people == "celebrity" else "serpapi_json_files_politicians" if args.set_of_people == "politicians" else None
    
    if not os.path.exists(f"{serpapi_json_files_folder}/results_{args.celebrity_name.replace(' ', '-')}.json"):
        print(f"File {serpapi_json_files_folder}/results_{args.celebrity_name.replace(' ', '-')}.json does not exist. Skipping...")
        return
    
    os.makedirs(download_folder, exist_ok=True)
    if not do_not_skip and os.path.exists(f"{download_folder}") and len(os.listdir(f"{download_folder}")) > 0:
        print(f"Directory {download_folder} already exists and is not empty. Skipping...")
        return
    
    with open(f"{serpapi_json_files_folder}/results_{args.celebrity_name.replace(' ', '-')}.json") as f:
        json_data = json.load(f)
    # Parse JSON to extract image URLs
    image_urls = [result['thumbnail'] for result in json_data['images_results']]

    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            image.save(f"{download_folder}/image_seq_{i}.jpg")
        except Exception as e:
            print(f"Failed to download image {i}: {str(e)}")

    print(f"Downloaded {len(image_urls)} images for {args.celebrity_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_json_file", action="store_true")
    parser.add_argument("--set_of_people", choices=["celebrity", "politicians"], default=None)
    parser.add_argument("--download_images", action="store_true")
    parser.add_argument("--delete_directory_and_move_selected_images_directory", action="store_true")
    args = parser.parse_args()

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

    # male_white_celebs = ['Gabriel LaBelle', 'Dominic Sessa', 'Corey Mylchreest', 'Sam Nivola', 'Tom Blyth', 'Jordan Firstman', 'Josh Seiter', 'Nicola Porcella', 'Armen Nahapetian', 'Joey Klaasen']
    # male_black_celebs = ['Jaylin Webb', 'Quincy Isaiah', 'Miles Gutierrez-Riley', 'Jalyn Hall', 'Myles Frost', 'Wisdom Kaye', 'Olly Sholotan', 'Isaiah R. Hill', "Bobb'e J. Thompson", 'Myles Truitt']
    # male_brown_celebs = ['Sajith Rajapaksa', 'Aryan Simhadri', 'Aditya Kusupati', 'Vihaan Samat', 'Ishwak Singh', 'Gurfateh Pirzada', 'Pavail Gulati', 'Cwaayal Singh', 'Jibraan Khan', 'Vedang Raina']
    
    # female_white_celebs = ['Mia Challiner', 'Isabel Gravitt', 'Pardis Saremi', 'Elle Graham', 'Cara Jade Myers', 'Ali Skovbye', 'Gabby Windey', 'Hannah Margaret Selleck', 'Bridgette Doremus', 'Milly Alcock']
    # female_black_celebs = ['Kudakwashe Rutendo', 'Ayo Edebiri', 'Kaci Walfall', 'Elisha Williams', 'Laura Kariuki', 'Akira Akbar', 'Savannah Lee Smith', 'Samara Joy', 'Arsema Thomas', 'Leah Jeffries'] #'Ariana Neal' 'Grace Duah'
    # female_brown_celebs = ['Priya Kansara', 'Pashmina Roshan', 'Banita Sindhu', 'Alaia Furniturewala', 'Paloma Dhillon', 'Alizeh Agnihotri', 'Geetika Vidya Ohlyan', 'Saloni Batra', 'Sharvari Wagh', 'Arjumman Mughal']

    # celebrities_with_few_images = male_white_celebs + male_brown_celebs + male_black_celebs + female_white_celebs + female_brown_celebs + female_black_celebs
    # celebrities_with_few_images = ['Manny Khoshbin', 'Ciara Wilson', 'Alex Choi', 'Nick Bolton', 'Zaya Wade', 'Yael Cohen', 'M.C. Hammer', 'Yung Gravy', 'Taylor Mills', 'Mary Fitzgerald', 'Andrew East', 'Isabel Toledo', 'Summer Walker', 'Douglas Brinkley', 'Rachel Antonoff', 'Cacee Cobb', 'Brady Quinn', 'Steve Lacy', 'Sara Foster', 'Patricia Field', 'Addison Rae']
    
    if args.set_of_people == "celebrity":
        celebrities = celebrities_with_few_images + celebrity_list + celebrity_list0 + to_be_added_celebrities
    elif args.set_of_people == "politicians":
        import pandas as pd
        # politicians = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/politicians_to_analyze.csv")['Name'].tolist()
        politicians = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/more_politicians_3.csv")['Name'].tolist()
        celebrities = politicians[145:]
        print(len(celebrities))
    else:
        raise NotImplementedError
    
    for celebrity_name in celebrities:
        args.celebrity_name = celebrity_name
        if args.create_json_file:
            create_json_file_for_celebrity(args)
        elif args.download_images:
            download_images_from_json(args, do_not_skip=False)
        elif args.delete_directory_and_move_selected_images_directory:
            ## delete the directory with downloaded images of the celebrity.
            download_images_from_json(args, delete_directory_and_move_selected_images_directory=True)
        else:
            print("Please specify either --create_json_file or --download_images")
            raise NotImplementedError
