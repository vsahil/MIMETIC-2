import subprocess
import time, os, copy
import pandas as pd
# import extract_from_laion_database as laion_access

# cache_dir = "/home/nlp/royira/vlm-efficiency/"
cache_dir = "/gscratch/h2lab/vsahil/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir


def get_available_gpus():
    """
    Returns a list of GPU IDs that are currently not in use.
    """
    try:
        ## also get the GPU type
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], check=True, text=True, capture_output=True)
        output = result.stdout
        if "A100" in output:
            gpu_type = "A100"
        elif "RTX 6000" in output:
            gpu_type = "RTX 6000"
        elif "A40" in output:
            gpu_type = "A40"
        elif "L40" in output:
            gpu_type = "L40"
        else:
            gpu_type = "Unknown"
            
        # Run the nvidia-smi command and get the output
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv'], check=True, text=True, capture_output=True)
        output = result.stdout

        # Parse the output to get the memory usage of each GPU
        lines = output.strip().split('\n')[1:] # Skip the header line
        gpu_memory_used = [int(line.split()[0]) for line in lines]

        # Find the GPUs that are not in use (memory usage is 0)
        available_gpus = [i for i, mem in enumerate(gpu_memory_used) if (mem <= 3000)]

        return available_gpus, gpu_type

    except subprocess.CalledProcessError:
        print("Failed to run nvidia-smi.")
        return [], "Unknown"

    except Exception as e:
        print("Error:", e)
        return [], "Unknown"


def remove_gpus_with_process_launching_soon(available_gpus, active_processes):
    remove_device_ids = []
    for device_id in available_gpus:
        if device_id in active_processes:
            if active_processes[device_id].poll() is None:
                print(f"Device ID {device_id} is still in use.")
                remove_device_ids.append(device_id)
            else:
                del active_processes[device_id]
    
    for device_id in remove_device_ids:
        print("Original set of devices: ", available_gpus)
        available_gpus.remove(device_id)
        print("Final set of devices: ", available_gpus)
    return available_gpus


def get_device(active_processes, naive_available_gpus):
    available_gpus, _ = get_available_gpus()
    available_gpus = remove_gpus_with_process_launching_soon(available_gpus, active_processes)
    if naive_available_gpus:
        available_gpus = list(set(available_gpus).intersection(set(naive_available_gpus)))
    

    # If there are no available GPUs, wait and try again later
    # import ipdb; ipdb.set_trace()
    while not available_gpus:
        time.sleep(100)
        available_gpus, _ = get_available_gpus()
        available_gpus = remove_gpus_with_process_launching_soon(available_gpus, active_processes)
     
    return available_gpus[0]


def run_expts(args):
    
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

    if args.set_of_people == "celebrity":
        celebrity_list_to_process = celebrity_list + celebrity_list0 + celebrities_with_few_images + to_be_added_celebrities
        processed_celebrities = []
        prompts = [("a photorealistic close-up photograph of ", 1), ("High-resolution close-up image of ", 1), ("'s facial close-up", 2), ("Close-up headshot of ", 1), ("'s face portrait", 2)]
        
        for celebrity in celebrity_list_to_process:
            saving_directory = f"generated_images_{args.set_of_people}_{args.model_id}/{celebrity}/{prompts[args.prompt_id][0]}"
            if os.path.exists(saving_directory):
                if len(os.listdir(saving_directory)) >= 200:
                    processed_celebrities.append(celebrity)
        
        print("number of celebrity:", len(celebrity_list_to_process))
        
        for celebrity in processed_celebrities:       ## because I started generating with model 2, instead of 5. 
            celebrity_list_to_process.remove(celebrity)

    elif args.set_of_people == "politicians":
        # politicians = pd.read_csv(args.entities_path)['Name'].tolist()
        # politicians = ['Barack Obama', 'John Lewis', 'Theresa May', 'Narendra Modi', 'Kim Jong-un', 'David Cameron', 'Angela Merkel', 'Bill Clinton', 'Xi Jinping', 'Justin Trudeau', 'Emmanuel Macron', 'Nancy Pelosi', 'Arnold Schwarzenegger', 'Ron Paul', 'Shinzo Abe', 'Adolf Hitler', 'John Paul II', 'Tony Blair', 'Sachin Tendulkar', 'Nick Clegg', 'Newt Gingrich', 'Scott Morrison', 'Arvind Kejriwal', 'Ilham Aliyev', 'Jacob Zuma', 'Bashar al-Assad', 'Laura Bush', 'Sonia Gandhi', 'Kim Jong-il', 'Robert Mugabe', 'James Comey', 'Rodrigo Duterte', 'Pete Buttigieg', 'Lindsey Graham', 'Hosni Mubarak', 'Enda Kenny', 'Alexei Navalny', 'Rob Ford', 'Leo Varadkar', 'Evo Morales', 'Lee Hsien Loong', 'Henry Kissinger', 'Petro Poroshenko', 'Joko Widodo', 'Clarence Thomas', 'Rishi Sunak', 'Mohamed Morsi', 'Ashraf Ghani', 'Martin McGuinness', 'Viktor Orban', 'Uhuru Kenyatta', 'Mike Huckabee', 'Sheikh Hasina', 'Martin Schulz', 'Giuseppe Conte', 'John Howard', 'Benito Mussolini', 'Tulsi Gabbard', 'Dominic Raab', 'Michael D. Higgins', 'François Hollande', 'Yasser Arafat', 'Mark Rutte', 'Mahathir Mohamad', 'Juan Manuel Santos', 'Abiy Ahmed', 'William Prince', 'Lee Kuan Yew', 'Mikhail Gorbachev', 'Hun Sen', 'Jacques Chirac', "Martin O'Malley", 'Benazir Bhutto', 'Yoshihide Suga', 'John Major', 'Muammar Gaddafi', 'Jerry Springer', "Sandra Day O'Connor", 'Madeleine Albright', 'Thomas Mann', 'Paul Kagame', 'Simon Coveney', 'Grant Shapps', 'Sebastian Coe', 'Merrick Garland', 'Jean-Yves Le Drian', 'Nursultan Nazarbayev', 'Horst Seehofer', 'Liz Truss', 'Rowan Williams', 'Ellen Johnson Sirleaf', 'George Weah', 'Mark Sanford', 'Yoweri Museveni', 'Luigi Di Maio', 'Ben Wallace', 'Herman Van Rompuy', 'Daniel Ortega', 'Olaf Scholz', 'Beppe Grillo', 'Alassane Ouattara', 'Nicolás Maduro', 'Tamim bin Hamad Al Thani', 'Mary McAleese', 'Asif Ali Zardari', 'Joseph Goebbels', 'Nikol Pashinyan', 'Deb Haaland', 'Paul Biya', 'Abdel Fattah el-Sisi', 'Thabo Mbeki', 'Kyriakos Mitsotakis', 'Joseph Muscat', 'Micheál Martin', 'Rebecca Long-Bailey', 'Paschal Donohoe', 'Todd Young', 'Jean-Marie Le Pen', 'Nick Griffin', 'Zoran Zaev', 'Pierre Nkurunziza', 'Abhisit Vejjajiva', 'Maggie Hassan', 'Steven Chu', 'Juan Guaidó', 'Edi Rama', 'Mary Landrieu', 'Jyrki Katainen', 'Jens Spahn', 'John Dramani Mahama', 'Gina Raimondo', 'Alec Douglas-Home', 'Viktor Orbán', 'Anita Anand', 'Isaias Afwerki', 'James Cleverly', 'Ibrahim Mohamed Solih', 'Leymah Gbowee', 'Václav Havel', 'John Rawls', 'Jack McConnell', 'Romano Prodi', 'Eoghan Murphy', 'Vicky Leandros', 'Norodom Sihamoni', 'Nayib Bukele', 'Shirin Ebadi', 'Jusuf Kalla', 'George Eustice', 'Joachim von Ribbentrop', 'Peter Altmaier', 'Akbar Hashemi Rafsanjani', 'Paul Singer', 'Christian Stock', 'Moussa Faki', 'Dominique de Villepin', 'Michael Fabricant', 'Kim Dae-jung', 'Eamon Ryan', 'Shavkat Mirziyoyev', 'Denis Sassou-Nguesso', 'Werner Faymann', 'Kamla Persad-Bissessar', 'Ingrid Betancourt', 'Volodymyr Zelenskyy', 'Park Chung Hee', 'Elvira Nabiullina', 'Roselyne Bachelot', 'Heinz Fischer', 'Hideki Tojo', 'Anatoly Karpov', 'Marcelo Ebrard', 'Slavoj Žižek', 'Trent Lott', 'Alfred Rosenberg', 'Valentina Matviyenko', 'Gabi Ashkenazi', 'Kgalema Motlanthe', 'Winona LaDuke', 'Pedro Castillo', 'Peter Bell', 'Boyko Borisov', 'Almazbek Atambayev', 'Carl Bildt', 'Andry Rajoelina', 'Carl Schmitt', 'Ralph Gonsalves', 'Liam Byrne', 'Alok Sharma', 'Jean-Michel Blanquer', 'Robert Schuman', 'Shinzō Abe', 'Doris Leuthard', 'Jacques Delors', 'Floella Benjamin', 'Sauli Niinistö', 'Annalena Baerbock', 'Toomas Hendrik Ilves', 'Bob Kerrey', 'Alejandro Giammattei', 'Lionel Jospin', 'Murray McCully', 'Stefan Löfven', 'Salva Kiir Mayardit', 'Javier Solana', 'Cecil Williams', 'Shahbaz Bhatti', 'Marianne Thyssen', 'Marty Natalegawa', 'Roh Moo-hyun', 'John Diefenbaker', 'Antonio Inoki', 'CY Leung', 'Iván Duque', 'Tom Tancredo', 'Sigrid Kaag', 'Jim Bolger', 'Lou Barletta', 'Li Peng', 'Gennady Zyuganov', 'Laura Chinchilla', 'Chen Shui-bian', 'Sebastián Piñera', 'Gustavo Petro', 'Miguel Díaz-Canel', 'Alberto Fernández', 'Gerald Darmanin', 'Boutros Boutros-Ghali', 'Maia Sandu', 'Joschka Fischer', 'Ricardo Martinelli', 'Andrej Babiš', 'Dan Jarvis', 'Nikos Dendias', 'Chris Hipkins', 'Tawakkol Karman', 'Booth Gardner', 'Karin Kneissl', 'Mobutu Sese Seko', 'Alexander Haig', 'Alexander De Croo', 'Ahmed Aboul Gheit', 'Yasuo Fukuda', 'Jean-Luc Mélenchon', 'Jane Ellison', 'Diane Dodds', 'Helen Whately', 'Idriss Déby', 'Carmen Calvo', 'Patrice Talon', 'Dario Franceschini', 'Emma Bonino', 'Richard Ferrand', 'Andreas Scheuer', 'Moshe Katsav', 'K. Chandrashekar Rao', 'P. Harrison', 'Robert Habeck', 'Ann Linde', 'Jon Ashworth', 'Edward Scicluna', 'Stef Blok', 'Lawrence Gonzi', 'William Roper', 'Josep Rull', 'Sam Kutesa', 'Raja Pervaiz Ashraf', 'David Cairns', 'Ilir Meta', 'Perry Christie', 'Rinat Akhmetov', 'Ahmet Davutoğlu', 'Franck Riester', 'Nikos Christodoulides', 'Umberto Bossi', "Damien O'Connor", 'Sali Berisha', 'Lee Cheuk-yan', 'Alpha Condé', 'Alexander Newman', 'Annette Schavan', 'Yuri Andropov', 'Faure Gnassingbé', 'Bolkiah of Brunei', 'Peter Tauber', 'Helen Suzman', 'Karl-Theodor zu Guttenberg', 'Michael Brand', 'Ron Huldai', 'Mohamed Azmin Ali', 'François-Philippe Champagne', 'Marielle de Sarnez', 'Agostinho Neto', 'Kurt Waldheim', 'Mounir Mahjoubi', 'Juan Orlando Hernández', 'Angela Kane', 'Lech Wałęsa', 'Luis Lacalle Pou', 'Barbara Pompili', 'Margaritis Schinas', 'Tigran Sargsyan', 'Wolfgang Bosbach', 'Raed Saleh', 'Johanna Wanka', 'Michelle Donelan', 'Roberto Speranza', 'Traian Băsescu', 'Dara Calleary', 'Iurie Leancă', 'Ilona Staller', 'Micheline Calmy-Rey', 'Thomas Oppermann', 'Karine Jean-Pierre', 'Luciana Lamorgese', 'Michael Adam', 'Azali Assoumani', 'Paulo Portas', 'Svenja Schulze', 'Pita Sharples', 'Choummaly Sayasone', 'Federico Franco', 'Félix Tshisekedi', 'Roberta Metsola', 'Paul Myners', 'Nia Griffith', 'Kaja Kallas', 'Ahmad Vahidi', 'Hua Guofeng', 'Olga Rypakova', 'Audrey Tang', 'Otto Grotewohl', 'Oskar Lafontaine', 'Ivica Dačić', 'Isa Mustafa', 'Xiomara Castro', 'M. G. Ramachandran', 'Fernando Grande-Marlaska', 'Wopke Hoekstra', 'Tomáš Petříček', 'Egils Levits', 'Roland Koch', 'Joseph Deiss', 'Laurentino Cortizo', 'Alan García', 'Nikola Poposki', 'Evarist Bartolo', 'Reyes Maroto', 'Zuzana Čaputová', 'Sergei Stanishev', 'Plamen Oresharski', 'Ana Brnabić', 'Carlos Alvarado Quesada', 'Marek Biernacki', 'Vjekoslav Bevanda', 'Olivier Véran', 'Clare Moody', 'Matthias Groote', 'Giorgos Stathakis', 'Elena Bonetti', 'Marta Cartabia', 'Dina Boluarte', 'Milo Đukanović', 'Levan Kobiashvili', 'Isabel Celaá', 'Jarosław Gowin', 'José Luis Escrivá', 'Cora van Nieuwenhuizen', 'Ivan Mikloš', 'Arancha González Laya', 'Gernot Blümel', 'Viola Amherd', 'José Luis Ábalos', 'Deo Debattista', 'Alain Krivine', 'Zlatko Lagumdžija', 'Edward Argar', 'Adrian Năstase', 'Zdravko Počivalšek', 'Miroslav Kalousek', 'Gabriel Boric', 'Karel Havlíček', 'Juan Carlos Campo', 'Kiril Petkov', 'Elżbieta Rafalska', 'Tobias Billström', 'Miroslav Toman', 'Mihai Răzvan Ungureanu', 'Ivaylo Kalfin', 'Élisabeth Borne', 'Herbert Fux', 'Petru Movilă', 'Caroline Edelstam', 'Koichi Tani', 'Barbara Gysi', 'Ľubomír Jahnátek', 'Nuno Magalhães', 'Iñigo Méndez de Vigo', 'Goran Knežević', 'Martin Pecina', 'Björn Böhning', 'Milan Kujundžić', 'Hernando Cevallos', 'Božo Petrov', 'Ian Karan', 'Adriana Dăneasă', 'Ida Karkiainen', 'Zoran Stanković', 'Frederik François', 'Józef Oleksy', 'Camelia Bogdănici', 'Boris Tučić', 'Zbigniew Ćwiąkalski', 'Rafael Catalá Polo', 'Ljube Boškoski', 'Jerzy Kropiwnicki', 'Metin Feyzioğlu', 'Herbert Bösch', 'Zoltán Illés', 'Vivi Friedgut']
        politicians = pd.read_csv("celebrity_data/good_politicians_to_analyze.csv")['Name'].tolist()
        celebrity_list_to_process = politicians
        assert len(celebrity_list_to_process) == 418
        
    elif args.set_of_people == "cub_birds":
        cub_birds_df = pd.read_csv('birds_dataset/cub_birds_sorted.csv')
        ## there are three columns in this csv. name and second is the count of each equuivalent name. and the third is the sum of the counts
        most_popular_name = []  ## bird_name,
        for bird_name, bird_count in zip(cub_birds_df['bird_name'], cub_birds_df['count_of_individual_bird_name']):
            if len(bird_name.split(",")) == 1:
                most_popular_name.append(bird_name)
            else:
                ## get the name with the highest count.
                bird_name = bird_name.split(",")
                bird_count = bird_count.split(",")
                assert len(bird_name) == len(bird_count)
                bird_count = [int(count) for count in bird_count]
                max_count_index = bird_count.index(max(bird_count))
                most_popular_name.append(bird_name[max_count_index])
        celebrity_list_to_process = most_popular_name
        assert len(celebrity_list_to_process) == 200
    
    elif args.set_of_people == "animals":
        animals = pd.read_csv(args.entities_path)['animal_name'].tolist()
        celebrity_list_to_process = animals
    
    elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
        if args.set_of_people == "wikiart_artists":
            artists = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/artists_to_analyze.csv")['artist_name'].tolist()
        elif args.set_of_people == "artists_group1":
            artists = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")['artist_name'].tolist()
        elif args.set_of_people == "artists_group2":
            artists = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")['artist_name'].tolist()
        else:
            raise NotImplementedError
        
        assert len(artists) > 0 and len(artists) < 420, "Invalid number of artists"
        ## get the names of the artists whose generation of images are complete and hence can be skipped. saving_directory = f"generated_images_wikiart_artists/{args.celebrity}/{prompt}". Check if there are more than 200 images in this directory for that artist, if there is, then skip that artist. prompt is "a painting in the style of "
        prompts = [("a painting in the style of ", 1), ("an artwork in the style of ", 1), ("a sketch in the style of ", 1), ("a fine art piece in the style of ", 1),  ("an illustration in the style of ", 1)]
        
        processed_artists = []
        for artist in artists:
            saving_directory = f"generated_images_wikiart_artists/{artist}/{prompts[args.prompt_id][0]}"
            if os.path.exists(saving_directory):
                if len(os.listdir(saving_directory)) >= 200:
                    processed_artists.append(artist)

        # original_length = copy.deepcopy(len(artists))
        for artist in processed_artists:
            artists.remove(artist)
        # assert len(artists) == original_length - len(processed_artists), f"Length of artists: {len(artists)}, Length of processed artists: {len(processed_artists)}, Original length: {original_length}"
        print("Number of artists to process: ", len(artists))
        celebrity_list_to_process = artists

    else:
        raise NotImplementedError
    ## Replace the above list with any list of people you want to generate images for.
    
    print("number of entities to process: ", len(celebrity_list_to_process), celebrity_list_to_process[:5])
    if len(celebrity_list_to_process) == 0:
        print("No entities to process.")
        return
    
    if args.chunk_number is not None:
        max_chunks = args.max_gpus
        chunks = []
        for chunk_number in range(1, max_chunks+1):
            chunk_size = len(celebrity_list_to_process) // max_chunks
            start_index = (chunk_number - 1) * chunk_size
            end_index = start_index + chunk_size
            if chunk_number == max_chunks:
                chunk = celebrity_list_to_process[start_index:]
            else:
                chunk = celebrity_list_to_process[start_index:end_index]
            chunks.append(chunk)
                
        if args.chunk_number != 999:
            print(f"Chunk number: {args.chunk_number}, Chunk size: {len(celebrity_list_to_process)}")
            celebrity_list_to_process = chunks[args.chunk_number - 1][::-1]
        else:
            ## we know that other GPUs are running the 8 chunks, so we should start processing from the end of the 8 chunks. So take the last name from all the chunks, then the second last from all the chunks and so on. It is not just the reverse of the list, but the last name from each chunk is taken first, then the second last name from each chunk is taken and so on.
            celebrity_list_to_process = []
            max_length = max([len(chunk) for chunk in chunks])
            for index in range(max_length-1, -1, -1):
                for chunk in chunks:
                    if index < len(chunk):
                         celebrity_list_to_process.append(chunk[index])
        print("Total number of entities to process: ", len(celebrity_list_to_process))
    
    active_processes = {}

    if args.rotate_gpus:
        for celebrity in celebrity_list_to_process:
            if args.produce_images:
                command = f'python gen_images.py --total_images 200 --batch_size 100 --prompt_id {args.prompt_id} '      ## batch size 20 with gpu-rtx6k and 50 with a40. 70 with L40s. 100 with A100s
            elif args.get_clip_scores:
                command = 'python evaluated_generation.py --get_clip_score '
            elif args.get_blipv2_scores:
                command = 'python evaluated_generation.py --get_blipv2_score '
            elif args.get_clip_score_trustworthiness_score:
                command = 'python evaluated_generation.py --get_clip_score_trustworthiness_score '
            elif args.get_blipv2_score_trustworthiness_score:
                command = 'python evaluated_generation.py --get_blipv2_score_trustworthiness_score '
            elif args.get_hpsv2_scores:
                command = 'python evaluated_generation.py --get_hpsv2_score '
            elif args.get_blipv2_vqa_score_binary:
                command = 'python evaluated_generation.py --get_blipv2_vqa_score_binary '
            elif args.get_blipv2_vqa_score_ordinal:
                command = 'python evaluated_generation.py --get_blipv2_vqa_score_ordinal '
            elif args.get_blipv2_vqa_score_binary_trustworthiness_score:
                command = 'python evaluated_generation.py --get_blipv2_vqa_score_binary_trustworthiness_score '
            elif args.get_blipv2_vqa_score_ordinal_trustworthiness_score:
                command = 'python evaluated_generation.py --get_blipv2_vqa_score_ordinal_trustworthiness_score '
            elif args.get_fid_scores:
                command = 'python evaluated_generation.py --get_fid_score '
            elif args.get_fid_similarity_score:
                command = 'python evaluated_generation.py --get_fid_similarity_score '
            elif args.get_clip_similarity_score:
                command = 'python evaluated_generation.py --get_clip_similarity_score '
            elif args.get_instruct_blip_vqa_score_binary:
                command = 'python evaluated_generation.py --get_instruct_blip_vqa_score_binary '
            elif args.get_instruct_blip_vqa_score_ordinal:
                command = 'python evaluated_generation.py --get_instruct_blip_vqa_score_ordinal '
            elif args.get_instruct_blip_vqa_score_binary_trustworthiness_score:
                command = 'python evaluated_generation.py --get_instruct_blip_vqa_score_binary_trustworthiness_score '
            elif args.get_instruct_blip_vqa_score_ordinal_trustworthiness_score:
                command = 'python evaluated_generation.py --get_instruct_blip_vqa_score_ordinal_trustworthiness_score '
            else:
                raise NotImplementedError
            
            command += f' --celebrity "{celebrity}"'
            
            device_id = get_device(active_processes)

            command += f' --device {device_id}'
            # python run_gen_images.py --produce_images --prompt_id 1 --fixed_gpus --set_of_people politicians
            print("Running command:", command)
            active_processes[device_id] = subprocess.Popen(command, shell=True)
            # processes.append(subprocess.Popen(command, shell=True))
            time.sleep(3)       ## after implementing the device id, the sleep is not needed.
        
        for process in active_processes.values():
            process.wait()


    elif args.fixed_gpus:
        ## here we first get the GPU ids we want to use and then divide the celebrities among them, and launch them in parallel. Each GPU will have its own list of celebrities to process and that will avoid the need to load the model again and again.
        available_gpus, gpu_type = get_available_gpus()
        print("Available GPUs: ", available_gpus, "GPU type: ", gpu_type)
        if gpu_type == "A100":
            if args.model_id == "1":
                batch_size = 80
            elif args.model_id == "5":
                batch_size = 70
            elif args.model_id == "v2":
                batch_size = 67
            else:
                batch_size = 50
        elif gpu_type == "RTX 6000":
            batch_size = 20
        elif gpu_type == "A40":
            if args.model_id == "1":
                batch_size = 50
            elif args.model_id == "5":
                batch_size = 40
            elif args.model_id == "v2":
                batch_size = 34
            else:
                batch_size = 34
        elif gpu_type == "L40" or gpu_type == "L40s":
            if args.model_id == "1":
                batch_size = 34
            elif args.model_id == "5":
                batch_size = 34
            elif args.model_id == "v2":
                batch_size = 29
            else:
                batch_size = 29
        elif gpu_type == "Unknown":
            batch_size = 20
        
        num_gpus = len(available_gpus)
        num_celebrities = len(celebrity_list_to_process)
        celebrities_per_gpu = (num_celebrities + num_gpus - 1) // num_gpus  # This ensures an even distribution
        print("Celebrities per GPU: ", celebrities_per_gpu, 'Total to process: ', len(celebrity_list_to_process), 'and batch size: ', batch_size)

        # Initialize a new list to store the sublists for each GPU
        sublists_for_each_gpu = {gpu_id: [] for gpu_id in available_gpus}
        for index, gpu_id in enumerate(available_gpus):
            start_index = index * celebrities_per_gpu
            end_index = start_index + celebrities_per_gpu
            if index == num_gpus - 1:
                # For the last GPU, ensure it includes any remaining celebrities
                celebrities_this_gpu = celebrity_list_to_process[start_index:]
            else:
                celebrities_this_gpu = celebrity_list_to_process[start_index:end_index]
            sublists_for_each_gpu[gpu_id] = celebrities_this_gpu

        # Flatten the list of sublists to get a single list of all celebrities
        all_celebrities_combined = [celebrity for sublist in sublists_for_each_gpu.values() for celebrity in sublist]
        no_overlap = len(all_celebrities_combined) == len(celebrity_list_to_process)
        complete_coverage = len(all_celebrities_combined) == len(celebrity_list_to_process)
        print(len(all_celebrities_combined), len(celebrity_list_to_process))
      
        # Print results
        print(f"No Overlap: {no_overlap}")
        print(f"Complete Coverage: {complete_coverage}")
        if args.specific_gpu is not None:
            command = f'CUDA_VISIBLE_DEVICES={args.specific_gpu} '
        else:
            command = ''

        for gpu_id, sublist in sublists_for_each_gpu.items():
            if args.produce_images:
                command += f'python gen_images.py --total_images 200 --batch_size {batch_size} --prompt_id {args.prompt_id} '
            elif args.get_fid_scores:
                command += f'python evaluated_generation.py --get_fid_score --prompt_id {args.prompt_id} '
            
            celebrities_this_gpu = ",".join(sublist)   ## remove the square brackets from the list of celebrities.
            command += f' --celebrity "{celebrities_this_gpu}"'
            command += f' --device {gpu_id}'
            if args.produce_images:
                command += f' --model_id {args.model_id}'
            command += f' --set_of_people {args.set_of_people}'
            if args.num_inference_steps is not None:
                command += f' --num_inference_steps {args.num_inference_steps}'
            
            print("Running command:", command, end="\n\n")
            active_processes[gpu_id] = subprocess.Popen(command, shell=True)
            time.sleep(1)
        
        for process in active_processes.values():
            process.wait()
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--produce_images", action="store_true")
    parser.add_argument("--rotate_gpus", action="store_true")
    parser.add_argument("--fixed_gpus", action="store_true")
    parser.add_argument("--set_of_people", choices= ["celebrity", "politicians", "cub_birds", "animals", "wikiart_artists", "artists_group1", "artists_group2"], default=None)
    parser.add_argument("--chunk_number", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 999], help="Chunk number to process. 999 means all chunks.")
    parser.add_argument("--max_gpus", type=int, default=8, help="Maximum number of GPUs to use.", required=True)
    parser.add_argument("--specific_gpu", type=int, default=None, help="Specific GPU to use.")
    
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--model_id", type=str, default="1", choices=["1", "5", "v2"])
    parser.add_argument("--entities_path", type=str, required=False, help="Path to a CSV file containing a 'Name' column.")
    parser.add_argument("--get_clip_scores", action="store_true")
    parser.add_argument("--get_blipv2_scores", action="store_true")
    parser.add_argument("--get_clip_score_trustworthiness_score", action="store_true")
    parser.add_argument("--get_blipv2_score_trustworthiness_score", action="store_true")
    parser.add_argument("--get_hpsv2_scores", action="store_true")
    
    parser.add_argument("--get_blipv2_vqa_score_binary", action="store_true")
    parser.add_argument("--get_blipv2_vqa_score_ordinal", action="store_true")
    parser.add_argument("--get_blipv2_vqa_score_binary_trustworthiness_score", action="store_true")
    parser.add_argument("--get_blipv2_vqa_score_ordinal_trustworthiness_score", action="store_true")

    parser.add_argument("--get_instruct_blip_vqa_score_binary", action="store_true")
    parser.add_argument("--get_instruct_blip_vqa_score_ordinal", action="store_true")
    parser.add_argument("--get_instruct_blip_vqa_score_binary_trustworthiness_score", action="store_true")
    parser.add_argument("--get_instruct_blip_vqa_score_ordinal_trustworthiness_score", action="store_true")
    
    parser.add_argument("--get_fid_scores", action="store_true")
    parser.add_argument("--get_fid_similarity_score", action="store_true")
    parser.add_argument("--get_clip_similarity_score", action="store_true")
    parser.add_argument("--get_all_scores", action="store_true")

    args = parser.parse_args()

    if args.produce_images:
        assert args.set_of_people is not None
        assert sum([args.rotate_gpus, args.fixed_gpus]) == 1
        assert args.prompt_id is not None
        assert args.prompt_id in [0, 1, 2, 3, 4]
        # assert args.model_id == "5"
        assert args.model_id == "1"
        if args.chunk_number is not None and args.chunk_number != 999:
            assert args.chunk_number <= args.max_gpus

    run_expts(args)
