import extract_from_laion_database as laion_access
from extract_from_laion_database import count_documents_containing_phrases, get_documents_containing_phrases, get_indices
import os, requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


es = laion_access.es_init()


def process_celebrity_names():
    import pandas as pd
    df = pd.read_csv("celebrity_data/celebrities.csv")     ## rowid,Name,image_counts
    ## sort by descending order of image_counts
    df = df.sort_values(by=['image_counts'], ascending=False)
    # df.drop(columns=['rowid'], inplace=True)
    ## remove duplicates
    df.drop_duplicates(subset=['Name'], inplace=True)
    ## print to a new csv file
    ## remove all rows whose names are single words
    df = df[df['Name'].str.contains(" ")]
    df.to_csv("celebrities.csv", index=False)


def print_celebrity_name_counts(args):
    if args.set_of_people == "celebrities":
        df = pd.read_csv("celebrity_data/celebrities_sorted.csv")
        people_names = df['Name'].tolist()
    elif args.set_of_people == "politicians":
        # df = pd.read_csv("celebrity_data/sampled_politicians.csv")
        # people_names = df['Name'].tolist()
        # people_names = ['Abdelmalek Sellal', 'Abhisit Vejjajiva', 'Adolf Hitler', 'Agostinho Neto', 'Ahmad Vahidi', 'Ahmed Aboul Gheit', 'Ahmet Davutoğlu', 'Akbar Hashemi Rafsanjani', 'Akinori Eto', 'Alain Krivine', 'Alan García', 'Alassane Ouattara', 'Alec Douglas-Home', 'Alexander Haig', 'Alexander Newman', 'Alexei Navalny', 'Alfred Rosenberg', 'Ali Khamenei', 'Almazbek Atambayev', 'Anatoly Karpov', 'Andrew Cuomo', 'Andrus Ansip', 'Andry Rajoelina', 'Angela Kane', 'Annette Schavan', 'Antonio Inoki', 'Ariel Sharon', 'Arnold Schwarzenegger', 'Arvind Kejriwal', 'Asif Ali Zardari', 'Barack Obama', 'Barbara Gysi', 'Benazir Bhutto', 'Benito Mussolini', 'Beppe Grillo', 'Bill Clinton', 'Bob Kerrey', 'Bolkiah of Brunei', 'Booth Gardner', 'Boris Johnson', 'Boutros Boutros-Ghali', 'Boyko Borisov', 'Carl Bildt', 'Carl Schmitt', 'Cecil Williams', 'Chen Shui-bian', 'Chen Yi', 'Choummaly Sayasone', 'Christian Stock', 'Clarence Thomas', 'CY Leung', 'David Cairns', 'Deng Xiaoping', 'Denis Sassou-Nguesso', 'Diane Dodds', 'Dominique de Villepin', 'Doris Leuthard', 'Eduard Shevardnadze', 'Edward Lucas', 'Edward Scicluna', 'Ellen Johnson Sirleaf', 'Elvira Nabiullina', 'Emma Bonino', 'Enda Kenny', 'Erika Steinbach', 'Evo Morales', 'Federico Franco', 'Felix Klein', 'Floella Benjamin', 'Franz Stein', 'Gabi Ashkenazi', 'Gennady Zyuganov', 'Gerhard Schröder', 'Gordon Brown', 'Heinz Fischer', 'Helen Suzman', 'Henry Kissinger', 'Herbert Bösch', 'Herbert Fux', 'Herman Van Rompuy', 'Hideki Tojo', 'Hosni Mubarak', 'Hua Guofeng', 'Hun Sen', 'Ian Karan', 'Ilona Staller', 'Ingrid Betancourt', 'Isa Mustafa', 'Iurie Leancă', 'Jack McConnell', 'Jackie Chan', 'Jacob Zuma', 'Jacques Chirac', 'Jacques Delors', 'James Comey', 'Javier Solana', 'Jean-Luc Mélenchon', 'Jean-Marie Le Pen', 'Jerry Springer', 'Jim Bolger', 'Joachim von Ribbentrop', 'Johanna Wanka', 'John Diefenbaker', 'John Dramani Mahama', 'John Howard', 'John Kasich', 'John Lewis', 'John Major', 'John Maynard Keynes', 'John Paul II', 'John Rawls', 'John Roberts', 'Joschka Fischer', 'Joseph Deiss', 'Joseph Goebbels', 'Józef Oleksy', 'Juan Manuel Santos', 'Jusuf Kalla', 'Jyrki Katainen', 'K. Chandrashekar Rao', 'Kamla Persad-Bissessar', 'Karin Jöns', 'Karl-Theodor zu Guttenberg', 'Kenny Anthony', 'Kevin Rudd', 'Kgalema Motlanthe', 'Kim Dae-jung', 'Kim Jong-il', 'Kim Jong-un', 'Kirron Kher', 'Kurt Waldheim', 'Kwame Nkrumah', 'Laura Bush', 'Laura Chinchilla', 'Lawrence Gonzi', 'Lech Wałęsa', 'Lee Cheuk-yan', 'Lee Hsien Loong', 'Lee Kuan Yew', 'Levan Kobiashvili', 'Leymah Gbowee', 'Li Peng', 'Lindsey Graham', 'Lionel Jospin', 'Lou Barletta', 'Loukas Papademos', 'M. G. Ramachandran', 'Macky Sall', 'Madeleine Albright', 'Maggie Hassan', 'Mahathir Mohamad', 'Manfred Wörner', 'Mark Sanford', 'Martin McGuinness', "Martin O'Malley", 'Martin Schulz', 'Marty Natalegawa', 'Mary Landrieu', 'Mary McAleese', 'Matthias Groote', 'Meg Whitman', 'Michael Adam', 'Michael Brand', 'Michael D. Higgins', 'Michael Grimm', 'Michele Bachmann', 'Micheline Calmy-Rey', 'Mike Huckabee', 'Mikhail Gorbachev', 'Mitt Romney', 'Mobutu Sese Seko', 'Mohamed Azmin Ali', 'Mohamed Morsi', 'Moon Jae-in', 'Moshe Katsav', 'Moussa Faki', 'Muammar Gaddafi', 'Murray McCully', 'Nana Mouskouri', 'Nancy Pelosi', 'Newt Gingrich', 'Nick Clegg', 'Nick Griffin', 'Nicolás Maduro', 'Norodom Sihamoni', 'Nursultan Nazarbayev', 'Olga Rypakova', 'Oskar Lafontaine', 'Otto Grotewohl', 'P. Harrison', 'Park Chung Hee', 'Paul Myners', 'Paul Singer', 'Perry Christie', 'Peter Bell', 'Peter Lilley', 'Peter Tauber', 'Petro Poroshenko', 'Pierre Nkurunziza', 'Pita Sharples', 'Rachida Dati', 'Raed Saleh', 'Raja Pervaiz Ashraf', 'Rajiv Gandhi', 'Ralph Gonsalves', 'Ricardo Martinelli', 'Rinat Akhmetov', 'Rob Ford', 'Robert Fico', 'Robert Habeck', 'Robert Mugabe', 'Robert Schuman', 'Roh Moo-hyun', 'Roland Koch', 'Roland Wöller', 'Romano Prodi', 'Ron Huldai', 'Ron Paul', 'Rowan Williams', 'Sachin Tendulkar', 'Salva Kiir Mayardit', 'Sam Kutesa', "Sandra Day O'Connor", 'Sauli Niinistö', 'Sebastian Coe', 'Sebastián Piñera', 'Shahbaz Bhatti', 'Shavkat Mirziyoyev', 'Sheikh Hasina', 'Shinzō Abe', 'Shirin Ebadi', 'Slavoj Žižek', 'Sonia Gandhi', 'Steven Chu', 'Susan Rice', 'Tawakkol Karman', 'Thabo Mbeki', 'Theodore Roosevelt', 'Thomas Mann', 'Thomas Oppermann', 'Tigran Sargsyan', 'Todd Young', 'Tom Tancredo', 'Tony Blair', 'Toomas Hendrik Ilves', 'Traian Băsescu', 'Trent Lott', 'Tulsi Gabbard', 'Umberto Bossi', 'Václav Havel', 'Valentina Matviyenko', 'Vicky Leandros', 'Viktor Orbán', 'Viola Amherd', 'Werner Faymann', 'William Prince', 'William Roper', 'Winona LaDuke', 'Wolfgang Bosbach', 'Woodrow Wilson', 'Yasser Arafat', 'Yasuo Fukuda', 'Yoshihide Suga', 'Yuri Andropov', 'Zhao Leji', 'Joe Biden', 'Theresa May', 'Narendra Modi', 'David Cameron', 'Angela Merkel', 'Xi Jinping', 'Justin Trudeau', 'Benjamin Netanyahu', 'Emmanuel Macron', 'Shinzo Abe', 'Hassan Rouhani', 'Scott Morrison', 'Ilham Aliyev', 'Bashar al-Assad', 'Jacinda Ardern', 'Rodrigo Duterte', 'Pete Buttigieg', 'Michael Gove', 'Sergey Lavrov', 'Cyril Ramaphosa', 'Leo Varadkar', 'Jair Bolsonaro', 'Mohammed bin Salman', 'Muhammadu Buhari', 'Joko Widodo', 'Rishi Sunak', 'Ashraf Ghani', 'Viktor Orban', 'Uhuru Kenyatta', 'Giuseppe Conte', 'Dominic Raab', 'François Hollande', 'Mark Rutte', 'Abiy Ahmed', 'Paul Kagame', 'Simon Coveney', 'Grant Shapps', 'Merrick Garland', 'Jean-Yves Le Drian', 'Horst Seehofer', 'Liz Truss', 'George Weah', 'Yoweri Museveni', 'Luigi Di Maio', 'Heiko Maas', 'Ben Wallace', 'Michel Aoun', 'Daniel Ortega', 'Olaf Scholz', 'Tamim bin Hamad Al Thani', 'Harjit Sajjan', 'Nikol Pashinyan', 'Deb Haaland', 'Paul Biya', 'Abdel Fattah el-Sisi', 'Kyriakos Mitsotakis', 'Joseph Muscat', 'Micheál Martin', 'Simon Harris', 'Rebecca Long-Bailey', 'Nadine Dorries', 'Paschal Donohoe', 'Recep Tayyip Erdoğan', 'Zoran Zaev', 'Juan Guaidó', 'Edi Rama', 'Jens Spahn', 'Lisa Nandy', 'Gina Raimondo', 'Anita Anand', 'Isaias Afwerki', 'James Cleverly', 'Ibrahim Mohamed Solih', 'Cecilia Malmström', 'Eoghan Murphy', 'Oliver Dowden', 'Igor Dodon', 'Nayib Bukele', 'George Eustice', 'Peter Altmaier', 'Michael Fabricant', 'Eamon Ryan', 'Volodymyr Zelenskyy', 'Roselyne Bachelot', 'Marcelo Ebrard', 'Pedro Castillo', 'Liam Byrne', 'Alok Sharma', 'Jean-Michel Blanquer', 'Annalena Baerbock', 'Alejandro Giammattei', 'Stefan Löfven', 'Marianne Thyssen', 'Iván Duque', 'Sigrid Kaag', 'Gustavo Petro', 'Miguel Díaz-Canel', 'Alberto Fernández', 'Gerald Darmanin', 'Maia Sandu', 'Andrej Babiš', 'Dan Jarvis', 'Nikos Dendias', 'Chris Hipkins', 'Karin Kneissl', 'Alexander De Croo', 'Jane Ellison', 'Helen Whately', 'Idriss Déby', 'Patrice Talon', 'Carmen Calvo', 'Dario Franceschini', 'Richard Ferrand', 'Andreas Scheuer', 'Ann Linde', 'Jon Ashworth', 'Stef Blok', 'Josep Rull', 'Ilir Meta', 'Franck Riester', 'Nikos Christodoulides', "Damien O'Connor", 'Sali Berisha', 'Alpha Condé', 'Faure Gnassingbé', 'François-Philippe Champagne', 'Marielle de Sarnez', 'Mounir Mahjoubi', 'Juan Orlando Hernández', 'Luis Lacalle Pou', 'Barbara Pompili', 'Margaritis Schinas', 'Michelle Donelan', 'Roberto Speranza', 'Dara Calleary', 'Karine Jean-Pierre', 'Luciana Lamorgese', 'Azali Assoumani', 'Paulo Portas', 'Svenja Schulze', 'Félix Tshisekedi', 'Roberta Metsola', 'Nia Griffith', 'Kaja Kallas', 'Audrey Tang', 'Ivica Dačić', 'Xiomara Castro', 'Fernando Grande-Marlaska', 'Wopke Hoekstra', 'Tomáš Petříček', 'Egils Levits', 'Laurentino Cortizo', 'Nikola Poposki', 'Ibrahim Boubacar Keïta', 'Miquel Iceta', 'Jadranka Kosor', 'Evarist Bartolo', 'Reyes Maroto', 'Julia Klöckner', 'Zuzana Čaputová', 'Janez Janša', 'Sergei Stanishev', 'Plamen Oresharski', 'Ana Brnabić', 'Carlos Alvarado Quesada', 'Anna Ekström', 'Nadia Calviño', 'Marek Biernacki', 'Olivier Véran', 'Vjekoslav Bevanda', 'Sandro Gozi', 'Ferdinand Grapperhaus', 'Vincenzo Amendola', 'Ján Kubiš', 'Clare Moody', 'Lorenzo Guerini', 'Giorgos Stathakis', 'Olivier Dussopt', 'Marta Cartabia', 'Elena Bonetti', 'Fatos Nano', 'Dina Boluarte', 'Milo Đukanović', 'Isabel Celaá', 'José Luis Escrivá', 'Jarosław Gowin', 'Benoît Cœuré', 'Sophie Wilmès', 'Cora van Nieuwenhuizen', 'Ivan Mikloš', 'Arancha González Laya', 'Călin Popescu-Tăriceanu', 'Gernot Blümel', 'José Luis Ábalos', 'Deo Debattista', 'Amélie de Montchalin', 'Zlatko Lagumdžija', 'Vlora Çitaku', 'Edward Argar', 'Zsolt Semjén', 'Adrian Năstase', 'Assunção Cristas', 'Zdravko Počivalšek', 'Juan Carrasco', 'Clément Beaune', 'Miroslav Kalousek', 'Gabriel Boric', 'Ingrida Šimonytė', 'Karel Havlíček', 'Juan Carlos Campo', 'Elżbieta Rafalska', 'Kiril Petkov', 'Tobias Billström', 'Miroslav Toman', 'Mihai Răzvan Ungureanu', 'Élisabeth Borne', 'Ivaylo Kalfin', 'Pedro Francke', 'Ludivine Dedonder', 'Petru Movilă', 'Koichi Tani', 'Mairéad McGuinness', 'Caroline Edelstam', 'Ivan Bartoš', 'Kostas Skrekas', 'Ľubomír Jahnátek', 'Nuno Magalhães', 'Goran Knežević', 'Dana Drábová', 'Iñigo Méndez de Vigo', 'Martin Pecina', 'Björn Böhning', 'Grațiela Gavrilescu', 'Miklós Soltész', 'Cátia Batista', 'Milan Kujundžić', 'Károly Grósz', 'Hernando Cevallos', 'Péter Medgyessy', 'Aníbal Torres', 'Alena Schillerová', 'Božo Petrov', 'Boris Tučić', 'Zoran Stanković', 'Camelia Bogdănici', 'Zoltán Illés', 'Jerzy Kropiwnicki', 'Adriana Dăneasă', 'Frederik François', 'Ida Karkiainen', 'Ljube Boškoski', 'Vivi Friedgut', 'Zbigniew Ćwiąkalski', 'Rafael Catalá Polo', 'Metin Feyzioğlu']
        # assert len(people_names) == 520
        people_names = pd.read_csv("celebrity_data/good_politicians_to_analyze.csv")['Name'].tolist()
        assert len(people_names) == 418
    # elif args.set_of_people == "birds":
    #     birds = pd.read_csv("birds_dataset/all_birds.csv")['birds'].tolist()
    #     people_names = birds
    elif args.set_of_people == "all_animals":
        df = pd.read_csv("animal_domain/sorted_sampled_animals.csv", sep='|')
        people_names = df['animal_name'].tolist()
    elif args.set_of_people == "imagenet_animals" or args.set_of_people == "birds" or args.set_of_people == "cub_birds":
        if args.set_of_people == "imagenet_animals":
            df = pd.read_csv("birds_dataset/imagenet_hypernyms.csv", sep='|')      ## Class Name|Hypernym 1|Hypernym 2|Hypernym 3|Hypernym 4|Hypernym 5|Hypernym 6|Hypernym 7
            df = df[df['Hypernym 7'] == "animal"]
            assert len(df) == 398
            animal_names_original = df['Class Name'].tolist()
        elif args.set_of_people == "birds":
            df = pd.read_csv("birds_dataset/modified_classes.csv", sep='|')
            animal_names_original = df['bird_name'].tolist()
        elif args.set_of_people == "cub_birds":
            df = pd.read_csv("birds_dataset/modified_cub_classes.csv")
            animal_names_original = df['bird_name'].tolist()
        ## note that in several of the class names there are multiple names for the same entity, so we want to split them by comma and get the count of each of the names
        animal_names = [name.split(",") for name in animal_names_original]
        animal_names = [name.strip() for sublist in animal_names for name in sublist]
        people_names = animal_names
    elif args.set_of_people == "cars":
        df = pd.read_csv("cars_dataset/cars_names.csv")
        people_names = df['CarName'].tolist()
    elif args.set_of_people == "fruits":
        df = pd.read_csv("fruits_dataset/fruits.csv")
        people_names = df['fruit_name'].tolist()
    elif args.set_of_people == "artist_styles":
        df = pd.read_csv("art_styles/full_artist_list.csv", sep='|')
        people_names = df['Names of Artists'].tolist()
    elif args.set_of_people == "wikiart_artists":
        df = pd.read_csv("art_styles/style_similarity_somepalli/wikiart_artists.csv")
        people_names = df['artist'].tolist()
    elif args.set_of_people == "artists_group1":
        df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
        people_names = df['artist_name'].tolist()
    elif args.set_of_people == "artists_group2":
        df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
        people_names = df['artist_name'].tolist()
    else:
        raise NotImplementedError
    
    ## get the count of each celebrity name
    people_counts = []
    people_selected_names = []
    for name in people_names:
        ## filter for names with length greater than 1 and less than 4 (2-3), splitting by space
        # if (len(name.split(" ")) < 2 or len(name.split(" ")) > 3) and (args.set_of_people == "celebrities" or args.set_of_people == "artist_styles"):
            # continue
        people_selected_names.append(name)
        # count_this_name = count_documents_containing_phrases("re_laion2b-en-*", name, es=es)
        count_this_name = count_documents_containing_phrases(args.laion_dataset, name, es=es)
        print(name, count_this_name)
        people_counts.append(count_this_name)
        
    if args.set_of_people == "celebrities":
        if args.laion_dataset == "re_laion2b-en-*":
            df['counts_in_laion2b-en'] = people_counts
            df.to_csv("celebrity_data/celebrities_sorted_new.csv", index=False)
        elif args.laion_dataset == "re_laion2b*":
            df['counts_in_laion5b'] = people_counts
            df = df.sort_values(by=['counts_in_laion5b'], ascending=False)
            df.to_csv("celebrity_data/celebrities_sorted_laion5b.csv", index=False)
    elif args.set_of_people == "politicians":
        if args.laion_dataset == "re_laion2b-en-*":
            df = pd.DataFrame({'Name': people_selected_names, 'counts_in_laion2b-en': people_counts})
            df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
            print("number of selected politicians: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
            df.to_csv("celebrity_data/good_images_politicians_sorted.csv", index=False)
        elif args.laion_dataset == "re_laion2b*":
            df = pd.DataFrame({'Name': people_selected_names, 'counts_in_laion5b': people_counts})
            df = df.sort_values(by=['counts_in_laion5b'], ascending=False)
            print("number of selected politicians: ", len(df), "max count: ", max(df['counts_in_laion5b']), "min count: ", min(df['counts_in_laion5b']))
            df.to_csv("celebrity_data/good_images_politicians_sorted_laion5b.csv", index=False)
    # elif args.set_of_people == "birds":
        # df = pd.DataFrame({'Name': people_selected_names, 'counts_in_laion2b-en': people_counts})
        # df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
        # print("number of selected birds: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
        # df.to_csv("birds_dataset/birds_sorted.csv", index=False) 
    elif args.set_of_people == "all_animals":
        df = pd.DataFrame({'animal_name': people_selected_names, 'counts_in_laion2b-en': people_counts})
        df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
        print("number of selected animals: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
        print(df)
        # df.to_csv("animal_datasets/animal_names_sorted.csv", index=False)
    elif args.set_of_people == "imagenet_animals" or args.set_of_people == "birds" or args.set_of_people == "cub_birds":
        final_counts = []
        for seq, orig_name in enumerate(animal_names_original):
            final_counts.append([])
            final_counts[-1].append(orig_name)
            counts = []
            for subname in orig_name.split(","):
                counts.append(people_counts[people_selected_names.index(subname.strip())])
            final_counts[-1].append(",".join([str(count) for count in counts]))
            final_counts[-1].append(sum(counts))
            
        assert len(final_counts) == len(animal_names_original)
        if args.set_of_people == "imagenet_animals":
            df = pd.DataFrame(final_counts, columns=["animal_name", "count_of_individual_animal_name", "sum_count_of_animal_name"])
            df = df.sort_values(by=['sum_count_of_animal_name'], ascending=False)
            print("number of selected animals: ", len(df), "max count: ", max(df['sum_count_of_animal_name']), "min count: ", min(df['sum_count_of_animal_name']))
            df.to_csv("birds_dataset/imagenet_hypernyms_sorted.csv", index=False)
        elif args.set_of_people == "birds":
            df = pd.DataFrame(final_counts, columns=["bird_name", "count_of_individual_bird_name", "sum_count_of_bird_name"])
            df = df.sort_values(by=['sum_count_of_bird_name'], ascending=False)
            print("number of selected birds: ", len(df), "max count: ", max(df['sum_count_of_bird_name']), "min count: ", min(df['sum_count_of_bird_name']))
            df.to_csv("birds_dataset/cub_and_nabirds_sorted.csv", index=False)
        elif args.set_of_people == "cub_birds":
            df = pd.DataFrame(final_counts, columns=["bird_name", "count_of_individual_bird_name", "sum_count_of_bird_name"])
            df = df.sort_values(by=['sum_count_of_bird_name'], ascending=False)
            print("number of selected birds: ", len(df), "max count: ", max(df['sum_count_of_bird_name']), "min count: ", min(df['sum_count_of_bird_name']))
            df.to_csv("birds_dataset/cub_birds_sorted.csv", index=False)
    elif args.set_of_people == "cars":
        df = pd.DataFrame({'CarName': people_selected_names, 'counts_in_laion2b-en': people_counts})
        df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
        print("number of selected cars: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
        df.to_csv("cars_dataset/cars_sorted.csv", index=False)
    elif args.set_of_people == "fruits":
        df = pd.DataFrame({'fruit_name': people_selected_names, 'counts_in_laion2b-en': people_counts})
        df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
        print("number of selected fruits: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
        df.to_csv("fruits_dataset/fruits_sorted.csv", index=False)
    elif args.set_of_people == "artist_styles":
        df = pd.DataFrame({'Names of Artists': people_selected_names, 'counts_in_laion2b-en': people_counts})
        df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
        print("number of selected artists: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
        df.to_csv("art_styles/artist_names_sorted.csv", index=False)
    elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
        if args.laion_dataset == "re_laion2b-en-*":
            assert args.set_of_people == "wikiart_artists"
            df = pd.DataFrame({'artist_name': people_selected_names, 'counts_in_laion2b-en': people_counts})
            df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
            ## get the counts of these artists in the wikiart dataset. Use the column "count" in file wikiart_artists.csv to get the counts
            wikiart_df = pd.read_csv("art_styles/style_similarity_somepalli/wikiart_artists.csv")
            wikiart_counts = []
            for artist in people_selected_names:
                wikiart_counts.append(wikiart_df[wikiart_df['artist'] == artist]['count'].tolist()[0])
            df['counts_in_wikiart'] = wikiart_counts
            print("number of selected artists: ", len(df), "max count: ", max(df['counts_in_laion2b-en']), "min count: ", min(df['counts_in_laion2b-en']))
            df.to_csv("art_styles/style_similarity_somepalli/wikiart_artists_sorted.csv", index=False)
        elif args.laion_dataset == "re_laion2b*":
            df = pd.DataFrame({'artist_name': people_selected_names, 'counts_in_laion5b': people_counts})
            df = df.sort_values(by=['counts_in_laion5b'], ascending=False)
            if args.set_of_people == "wikiart_artists":
                df.to_csv("art_styles/style_similarity_somepalli/wikiart_artists_sorted_laion5b.csv", index=False)
            elif args.set_of_people == "artists_group1":
                df.to_csv("art_styles/style_similarity_somepalli/final_artists_group1_sorted_laion5b.csv", index=False)
            elif args.set_of_people == "artists_group2":
                df.to_csv("art_styles/style_similarity_somepalli/final_artists_group2_sorted_laion5b.csv", index=False)
    else:
        raise NotImplementedError
        

def get_celebrity_first_names_total_count():
    '''
    We will use this function to determine the commonality of the first names of the celebs. My hypothesis if that the first name is very common then the model might perform worse for them as it is difficult to distinguish a particular celeb. It might also the ratio of the number of times this celeb is mentioned by the number of times the first name is mentioned. 
    '''
    celebs_and_full_name_counts = pd.read_csv("celebrity_data/celebrities_sorted.csv", header=0)       ## its header is Name,counts_in_laion2b-en
    ## for a random sample of 100 celebs from this list, get the count of their full name and assert that it is equal to the count loaded from the csv file
    import random
    random.seed(42)
            ## this is just a sanity check, we are not selecting the 100 celebs randomly, we are selecting them in increasing order of their counts
    random_celebs = random.sample(celebs_and_full_name_counts['Name'].tolist(), 100)
    for celeb in random_celebs:
        full_name_count = count_documents_containing_phrases("re_laion2b-en-*", celeb, es=es)
        assert full_name_count == celebs_and_full_name_counts[celebs_and_full_name_counts['Name'] == celeb]['counts_in_laion2b-en'].tolist()[0], "celeb: {}, full_name_count: {}, celebs_and_full_name_counts: {}".format(celeb, full_name_count, celebs_and_full_name_counts[celebs_and_full_name_counts['Name'] == celeb]['counts_in_laion2b-en'].tolist()[0])

    ## get the first names of the celebs
    ## now we want to understand the difference between the function count_documents_containing_phrases and count_total_occurrences_of_unigrams
    ## count_total_occurrences_of_unigrams is a function that counts the number of times a unigram occurs in the entire corpus. It is not a function that counts the number of documents in which a unigram occurs. Okay then we are only interested in the number of documents in which a unigram occurs.
    ## get the first names of the celebs
    celebs_and_full_name_counts['first_name'] = celebs_and_full_name_counts['Name'].apply(lambda x: x.split(" ")[0])
    ## get the count of the first names of the celebs
    first_name_counts = []
    for seq_id, name in enumerate(celebs_and_full_name_counts['first_name'].tolist()):
        print(seq_id, name)
        first_name_counts.append(count_documents_containing_phrases("re_laion2b-en-*", name, es=es))
    celebs_and_full_name_counts['first_name_counts_in_laion2b-en'] = first_name_counts
    ## save the df
    celebs_and_full_name_counts.to_csv("celebrities_sorted.csv", index=False)
    print("Total number of celebs: ", len(celebs_and_full_name_counts))


def print_most_popular_celebrities():
    df = pd.read_csv("celebrity_data/celebrities_sorted.csv")
    df = df.sort_values(by=['counts_in_laion2b-en'], ascending=False)
    # df.drop(columns=['image_counts'], inplace=True)
    # df.to_csv("celebrities_sorted.csv", index=False)

    ## make a barplot of the distribution of the counts. Divide the counts by the total number of documents in the corpus, to get the percentage of documents that contain the celebrity name
    counts = df['counts_in_laion2b-en'].tolist()
    datasets = get_indices()
    total_size = int(datasets["re_laion2b-en-1"]['docs.count']) + int(datasets["re_laion2b-en-2"]['docs.count'])
    assert total_size > 2.3 * 10 ** 9       ## 2.3 billion
    percentages = counts
    # percentages = [i * 100. / total_size for i in counts]
    ## make a plot now
    import matplotlib.pyplot as plt
    print(len(percentages), max(percentages), min(percentages))
    plt.figure(figsize=(10, 7))
    ## I want the celebrities to be sorted by the percentage of documents containing their names. The most popular celebrity should be on the rightmost. Don't print the names of any celebrity. 
    percentages = sorted(percentages)
    plt.bar(range(len(percentages)), percentages, color='blue')
    plt.xticks(range(len(percentages)), [])
    # plt.ylabel("Percentage of images in Laion-2B-EN dataset of a celebrity")
    plt.ylabel('Number of Captions containing the Celebrity')
    plt.xlabel("Celebrities")

    # ## add the names of the most popular celebrities to the x-axis of the plot
    # ## most popular
    most_popular = df.head(1)
    most_popular_names = most_popular['Name'].tolist()
    # most_popular_counts = most_popular['counts_in_laion2b-en'].tolist()
    # most_popular_percentages = [i * 100. / total_size for i in most_popular_counts]
    # # import ipdb; ipdb.set_trace()
    # ## least popular
    # least_popular = df.tail(1)
    # least_popular_names = least_popular['Name'].tolist()
    # least_popular_counts = least_popular['counts_in_laion2b-en'].tolist()
    # least_popular_percentages = [i * 100. / total_size for i in least_popular_counts]
    ## add every 100th xtick with the name of the celebrity
    ## sort the df in ascending order of the counts
    df = df.sort_values(by=['counts_in_laion2b-en'], ascending=True)

    celebrities_to_generate = []
    # import ipdb; ipdb.set_trace()

    x_axis = []
    names_axis = []
    for i in range(len(percentages)):
        if i % 101 == 0 and i != 0:
            x_axis.append(i)
            names_axis.append(df.iloc[i]['Name'])
            celebrities_to_generate.append(df.iloc[i]['Name'])
    x_axis.append(len(percentages) - 1)
    
    names_axis.append(most_popular_names[0])        ## add the most popular celebrity to the x-axis
    celebrities_to_generate.append(most_popular_names[0])

    
    plt.xticks(x_axis, names_axis, rotation=80)

    # plt.title("Distribution of the percentages of images in Laion-2B-EN dataset of a celebrity")
    # plt.savefig("celebrity_data/celebrities_distribution.png")
    plt.title("Histogram of the number of captions of each celebrity in the Laion-2B-rn dataset")
    plt.tight_layout()
    plt.savefig("celebrity_data/celebrities_distribution.pdf")

    print(celebrities_to_generate, len(celebrities_to_generate))


def get_celebrity_count_distribution(args, full_celebrity_list):
    if args.set_of_people == "celebrities":
        count_df = pd.read_csv("celebrity_data/celebrities_sorted.csv")
    elif args.set_of_people == "politicians":
        count_df = pd.read_csv("celebrity_data/politicians_sorted.csv")
    else:
        raise NotImplementedError

    # for celebrity in full_celebrity_list:
    #     count = count_df[count_df['Name'] == celebrity]['counts_in_laion2b-en'].tolist()[0]
    #     print(celebrity, count)
    # print(len(full_celebrity_list))
    ## now we want to get more people from the df who are not in the full_celebrity_list and their counts are between 490 and 1500
    ## print the number of such people
    consider_df = count_df[(count_df['counts_in_laion2b-en'] >= 490) & (count_df['counts_in_laion2b-en'] <= 2500)]
    to_be_added = []
    for celebrity in consider_df['Name'].tolist():
        if celebrity not in full_celebrity_list:
            print(celebrity, consider_df[consider_df['Name'] == celebrity]['counts_in_laion2b-en'].tolist()[0])
            to_be_added.append(celebrity)
    print(len(to_be_added), len(full_celebrity_list) + len(to_be_added) )
    print(to_be_added[::-1])
            

def download_celebrity_images(celebrities, only_download_captions_for_already_downloaded_images=False):
    from extract_from_laion_database import phrase_download_image

    for seq, i in enumerate(celebrities):
        if seq % 50 == 0:
            print(i)

    # for celebrity in celebrities:
    #     phrase_download_image(celebrity, max_num=500)

    ## download the images in parallel across multiple processes - start 48 processes at a time
    import multiprocessing as mp
    processes = []
    for celebrity in celebrities:
        processes.append(mp.Process(target=phrase_download_image, args=(celebrity, 1000, only_download_captions_for_already_downloaded_images)))
        if len(processes) == 24:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            processes = []
    
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def count_landmarks(args):
    df = pd.read_csv("df_landmarks_combined.csv")       ## landmark,landmark_nopunc,count,count_nopunc,count_max
    ## get the list of landmarks using landmark_nopunc and print the count using count_nopunc
    landmarks = df['landmark_nopunc'].tolist()
    counts_measured = df['count_nopunc'].tolist()
    assert len(landmarks) == len(counts_measured)
    for landmark, count in zip(landmarks, counts_measured):
        laion2b_count = count_documents_containing_phrases("re_laion2b-en-*", landmark, es=es)
        # print(landmark, laion2b_count, count)
        assert laion2b_count == count, "landmark: {}, laion2b_count: {}, count: {}".format(landmark, laion2b_count, count)


def remove_or_rename_single_image(image, celebrity_dir, actual_removal, download_directory):
    import os
    from PIL import Image
    
    if image.endswith(".csv") or (image.startswith("open_clip_image_features") and image.endswith(".pt")):
        return
    try:
        img = Image.open(os.path.join(celebrity_dir, image))
        ## get the shape of the image
        width, height = img.size
        if width == 1 or height == 1:
            if actual_removal: os.remove(os.path.join(celebrity_dir, image))
            print("removed because width of height is 1", os.path.join(celebrity_dir, image), img.size)
            return
        ## some images are truncated, we need to remove them. Get image sizes
        try:
            img.convert('RGB')
        except:
            if actual_removal: os.remove(os.path.join(celebrity_dir, image))
            print("remove because img cannot be converted to RGB ", os.path.join(celebrity_dir, image), img.size)
            return
        file_size_in_bytes = os.path.getsize(os.path.join(celebrity_dir, image))
        if file_size_in_bytes < 1000:
            if actual_removal: os.remove(os.path.join(celebrity_dir, image))
            print("Remove because image file is small ", os.path.join(celebrity_dir, image), file_size_in_bytes)
            return
    except:
        if actual_removal: os.remove(os.path.join(celebrity_dir, image))
        print("Remove because cannot be opened by Image ", os.path.join(celebrity_dir, image))
        return

    ## Also open by cv2
    import cv2
    try:
        img = cv2.imread(os.path.join(celebrity_dir, image))
        if img is None:
            if actual_removal: os.remove(os.path.join(celebrity_dir, image))
            print("cannot be opened by cv2", os.path.join(celebrity_dir, image))
            return
    except:
        if actual_removal: os.remove(os.path.join(celebrity_dir, image))
        print("Remove the image that cannot be opened by cv2 ", os.path.join(celebrity_dir, image))
        return
    
    ## each image is of the form: f'image_seq_{seq}_' + os.path.basename(url), we want to remove the os.path.basename(url) part. Note that there can be multiple underscores in the name in the url part
    
    if not download_directory == "art_styles/style_similarity_somepalli/wikiart_images":
        assert image.startswith("image_seq_")
        seq_number = int(image.split("_")[2])
        new_filename = f"image_seq_{seq_number}"
    else:
        new_filename = image

    ## get the extension of the image
    with Image.open(os.path.join(celebrity_dir, image)) as img:
        extension = img.format.lower()
        if extension == "jpeg":
            extension = "jpg"
        new_filename = f"{new_filename}.{extension}"
        # print("renaming", os.path.join(celebrity_dir, image), os.path.join(celebrity_dir, new_filename))
        if actual_removal: os.rename(os.path.join(celebrity_dir, image), os.path.join(celebrity_dir, new_filename))
        

def remove_image_single_celebrity(celebrity, download_directory):
    celebrity_dir = os.path.join(download_directory, celebrity)
    actual_removal = True
    for image in os.listdir(celebrity_dir):
        remove_or_rename_single_image(image, celebrity_dir, actual_removal, download_directory)
        

def remove_images_that_cannot_be_opened_by_pil(args, celebrities, download_directory, max_cores):
    ## open images using PIL and remove the ones that cannot be opened
    import multiprocessing as mp
    if args.parallelize_across_entities:        
        processes = []
        max_processes = min(max_cores, len(celebrities))
        for celebrity in celebrities:
            # print(celebrity)
            processes.append(mp.Process(target=remove_image_single_celebrity, args=(celebrity, download_directory, )))
            if len(processes) == max_processes:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
                processes = []

        for process in processes:
            process.start()
        for process in processes:
            process.join()
    
    elif args.parallelize_across_one_entity:        ## in this case we will parallelize the remove process across the images of one celebrity
        assert len(celebrities) == 1
        celebrity = celebrities[0]
        celebrity_dir = os.path.join(download_directory, celebrity)
        actual_removal = True
        images = os.listdir(celebrity_dir)
        
        if len(images) <= 100:
            ## remove the images in a single process
            for image in images:
                remove_or_rename_single_image(image, celebrity_dir, actual_removal, download_directory)
        else:
            max_processes = min(max_cores, len(images))
            processes = []
            for image in images:
                processes.append(mp.Process(target=remove_or_rename_single_image, args=(image, celebrity_dir, actual_removal, download_directory)))
                if len(processes) == max_processes:
                    for process in processes:
                        process.start()
                    for process in processes:
                        process.join()
                    processes = []
                    
            for process in processes:
                process.start()
            for process in processes:
                process.join()


if __name__ == "__main__":    
    celebrity_list = ['Aditya Kusupati', 'Bridgette Doremus', 'Joey Klaasen', 'Hannah Margaret Selleck', 'Mary Lee Pfeiffer', 'Miguel Bezos', 'Ben Mallah', 'Tara Lynn Wilson', 'Eric Porterfield', 'Quinton Reynolds', 'Avani Gregg', 'David Portnoy', 'Kyle Cooke', 'Sue Aikens', 
                  'Danielle Cohn', 'Marcus Freeman', 'Ryan Upchurch', 'Greg Fishel', 'Wayne Carini', 'Cole LaBrant', 'Ella Emhoff', 'Manny Khoshbin', 'Lexi Underwood', 'Ciara Wilson', 'Scott Yancey', 'Alex Choi', 'Shenae Grimes-Beech', 'Nicole Fosse', 'Isaak Presley', 
                  'Yamiche Alcindor', 'Matthew Garber', 'Rege-Jean Page', 'Nick Bolton', 'Will Reeve', 'Madison Bailey', 'Maitreyi Ramakrishnan', 'Zaya Wade', 'Sessilee Lopez', 'Albert Lin', 'Frank Sheeran', 'Fred Stoller', 'Trevor Wallace', 'Madelyn Cline', 
                  'Angie Varona', 'Yael Cohen', 'Bailey Sarian', 'Zack Bia', 'M.C. Hammer', 'Irwin Winkler', 'Danny Koker', 'Tony Lopez', 'Sasha Calle', 'Maggie Rizer', 'Jillie Mack', "Olivia O'Brien", 'Joanna Hillman', 'Belle Delphine', 'Chase Stokes', 'Kyla Weber', 
                  'Alexandra Cooper', 'Jordan Chiles', 'Bob Keeshan', "Dixie D'Amelio", 'Daisy Edgar-Jones', 'Yung Gravy', 'Jamie Bochert', 'Forrest Fenn', 'Barbie Ferreira', 'DJ Kool Herc', 'Bregje Heinen']

    celebrity_list0 = ['Dan Peña', 'Thandiwe Newton', 'Alexa Demie', 'Paul Mescal', 'Jeremy Bamber', 'Malgosia Bela', 'Lacey Evans', 'Tao Okamoto', 'Ashleigh Murray', 'Nico Tortorella', 'Annie Murphy', 'Jimmy Buffet', 'Marsai Martin', 'Sofia Hellqvist', 'Skai Jackson', 'Doja Cat', 
                    'Mackenzie Foy', 'Lana Condor', 'Selita Ebanks', 'Jay Baruchel', 'Leslie Odom Jr.', 'Joey Fatone', 'Karen Elson', 'Penélope Cruz', 'Lily Donaldson', 'Lauren Daigle', 'James Garner', 'Chris Farley', 'Mark Consuelos', 'Eric Bana', 'Ray Liotta', 'Robert Kraft', 
                    'Riley Keough', 'Brody Jenner', 'Ross Lynch', 'Joel McHale', 'Melanie Martinez', 'Hunter Hayes', 'Gwendoline Christie', 'Zachary Levi', 'Troye Sivan', 'Marisa Tomei', 'Kevin James', 'Cuba Gooding Jr.', 'Sterling K. Brown', 'Rose Leslie', 'Clive Owen', 'Nick Offerman', 
                    'Ansel Elgort', 'Natalie Dormer', 'Terrence Howard', 'Sacha Baron Cohen', 'Joe Rogan', 'Olivia Culpo', 'Meg Ryan', 'Brendon Urie', 'Forest Whitaker', 'Matthew Perry', 'Sarah Paulson', 'Saoirse Ronan', 'Felicity Jones', 'Kurt Russell', 'Kacey Musgraves', 'Tony Romo', 
                    'Ashlee Simpson', 'Greta Thunberg', 'Carey Mulligan', 'Ashley Olsen', 'Martin Scorsese', 'Scott Disick', 'Victoria Justice', 'Mandy Moore', 'Jason Statham', 'Samuel L. Jackson', 'David Tennant', 'Jay Leno', 'Bob Ross', 'Bella Thorne', 'Steven Spielberg', 'Malcolm X', 
                    'Kendrick Lamar', 'Jessica Biel', 'Nick Jonas', 'Naomi Campbell', 'Mark Wahlberg', 'Cate Blanchett', 'Cameron Diaz', 'Dwayne Johnson', 'Floyd Mayweather', 'Oprah Winfrey', 'Ronald Reagan', 'Ben Affleck', 'Anne Hathaway', 'Stephen King', 'Johnny Depp', 'Abraham Lincoln', 'Kate Middleton', 'Donald Trump']

    celebrities_with_few_images = ['Sajith Rajapaksa', 'Mia Challiner', 'Yasmin Finney', 'Gabriel LaBelle', 'Isabel Gravitt', 'Pardis Saremi', 'Dominic Sessa', 'India Amarteifio', 'Aryan Simhadri', 'Arsema Thomas', 'Sam Nivola', 'Corey Mylchreest', 'Diego Calva', 'Armen Nahapetian', 'Jaylin Webb', 
                                'Gabby Windey', 'Amaury Lorenzo', 'Kudakwashe Rutendo', 'Cwaayal Singh', 'Chintan Rachchh', 'Adwa Bader', 'Vedang Raina', 'Delaney Rowe', 'Aria Mia Loberti', 'Florence Hunt', 'Tom Blyth', 'Kris Tyson', 'Tioreore Ngatai-Melbourne', 'Cody Lightning', 'Mason Thames', 
                                'Samara Joy', 'Wisdom Kaye', 'Jani Zhao', 'Elle Graham', 'Priya Kansara', 'Boman Martinez-Reid', 'Park Ji-hu', 'Cara Jade Myers', 'Banks Repeta', 'Ali Skovbye', 'Nicola Porcella', 'Keyla Monterroso Mejia', 'Pashmina Roshan', 'Jeff Molina', 'Woody Norman', 'Leah Jeffries', 'Lukita Maxwell', 'Jordan Firstman', 'Josh Seiter', 'Ayo Edebiri']
    
    to_be_added_celebrities = ['David Brinkley', 'Kate Chastain', 'Portia Freeman', 'Taylor Mills', 'Mary Fitzgerald', "Miles O'Brien", 'Andrew East', 'Una Stubbs', 'Nicola Coughlan', 'Matthew Bomer', 'Jim Walton', "Charli D'Amelio", 'Dan Lok', 'Nicholas Braun', 'Connor Cruise', 'Matt Stonie', 'Grant Achatz', 'Eiza González', 'Anna Cleveland', 'Melanie Iglesias', 
                               'Jacquetta Wheeler', 'Austin Abrams', 'The ACE Family', 'Julia Stegner', 'Miles Heizer', 'Devon Sawa', 'Julie Chrisley', 'Barry Weiss', "Sha'Carri Richardson", 'Elliot Page', 'Emma Barton', 'Tommy Dorfman', 'Joe Lacob', 'Stephen tWitch Boss', 'Kendra Spears', 'Sam Elliot', 'Luka Sabbat', 'Hunter Schafer', 
                               'Marcus Lemonis', 'Danneel Harris', 'Ivan Moody', 'Tammy Hembrow', 'Ethan Suplee', 'Hunter McGrady', 'Mat Fraser', 'Sydney Sweeney', 'Isabel Toledo', 'Allison Stokke', 'Kim Zolciak-Biermann', 'Phoebe Dynevor', 'Jenna Ortega', 'Lew Alcindor', 'Emma Chamberlain', 'Summer Walker', 'Mariacarla Boscono', 'Justina Machado', 'Erin Foster', 
                               'Hero Fiennes-Tiffin', 'Douglas Brinkley', 'Sunisa Lee', 'Tina Knowles-Lawson', 'Peter Firth', 'Lauren Bush Lauren', 'Presley Gerber', 'Rachel Antonoff', 'Kieran Culkin', 'Annie LeBlanc', 'Michael Buffer', 'Jacquelyn Jablonski', 'Paz de la Huerta', 'Chris McCandless', 'Arthur Blank', 'Jay Ellis', 'Cacee Cobb', 'Ziyi Zhang', 'Indya Moore', 
                               'Bill Skarsgård', 'Thomas Doherty', 'Elisa Sednaoui', 'Margherita Missoni', 'Jared Followill', 'Taylor Tomasi Hill', 'Fei Fei Sun', 'Deana Carter', 'Hannah Bronfman', 'Anwar Hadid', 'Olivia Rodrigo', 'Frankie Grande', 'Diane Ladd', 'Sophia Lillis', 'Genesis Rodriguez', 'Richard Rawlings', 'Jill Wagner', 'Brady Quinn', 'Simu Liu', 'Diane Guerrero', 
                               'Matt McGorry', 'Hunter Parrish', 'Eddie Hall', 'Daria Strokous', 'Peter Hermann', 'The Koch Brothers', 'Sasha Pivovarova', 'Liu Yifei', 'Danielle Jonas', 'Magdalena Frackowiak', 'Mark Richt', 'Crystal Renn', 'Amanda Gorman', 'Dan Crenshaw', 'Todd Chrisley', 'Eddie McGuire', 'Lori Harvey', 'Beanie Feldstein', 'David Muir', 'Martin Starr', 'Quvenzhané Wallis', 
                               'Lo Bosworth', 'Emma Corrin', 'Steve Lacy', 'Liza Koshy', 'Zak Bagans', 'Bob Morley', 'Jenna Marbles', 'Trevor Jackson', 'Jodie Turner-Smith', 'Michael Peña', 'Rhea Perlman', 'Luke Hemsworth', 'Rodrigo Santoro', 'Chris Klein', 'Lauren Bushnell', 'Oliver Hudson', 'Danny Thomas', 'Diego Boneta', 'Sara Foster', 'Michiel Huisman', 'Dan Bilzerian', 'Patricia Field', 
                               'Charles Melton', 'AJ Michalka', 'Stella Tennant', 'Lindsay Wagner', 'Justice Smith', 'Addison Rae', 'Scott Speedman', 'Christiano Ronaldo', 'Hanne Gaby Odiele', 'The Kennedy Family', 'Michaela Coel', 'John Corbett', 'Lindsay Ellingson', 'Ryan Guzman', 'Marvin Sapp', 'John Wayne Gacy', 'Joyner Lucas', 'Rachelle Lefevre', 'Jessica Seinfeld', 'Sam Taylor Johnson', 
                               'Normani Kordei', 'Danielle Fishel', 'Robert Irvine', 'Sandra Cisneros', 'Alek Wek', 'Torrey Devitto', 'Christine Quinn', 'Jacob Sartorius', 'Chris Watts', 'Allison Holker', 'Mamie Gummer', 'Gregg Sulkin', 'Dree Hemingway', 'Alyson Stoner', 'Omar Epps', 'Jacob Elordi', 'David Dobrik', 'Cam Gigandet', 'Léa Seydoux', 'Tai Lopez', 'Alexander Skarsgård', 'Sarah Chalke', 
                               'Phoebe Bridgers', 'Lindsay Price', 'Josh Peck', 'Bernard Arnault']

    import pandas as pd
    
    politicians = pd.read_csv("celebrity_data/sampled_politicians.csv")['Name'].tolist()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--laion_dataset", type=str, default="re_laion2b-en-*", choices=["re_laion2b-en-*", "re_laion2b*"], help="The second one counts across LAION-5B and the first one for LAION-2B-en")
    parser.add_argument("--set_of_people", type=str, default=None, choices=["celebrities", "politicians", "birds", "imagenet_animals", "cars", "fruits", "all_animals", "cub_birds", "artist_styles", "wikiart_artists", "wikiart_images", "caption_assumption", "artists_group1", "artists_group2"])
    parser.add_argument("--consider_alias", action="store_true")
    parser.add_argument("--print_celebrity_name_counts", action="store_true")
    parser.add_argument("--get_celebrity_first_names_total_count", action="store_true")
    parser.add_argument("--download_images_celebs", action="store_true", help="This downloads the images, but does not remove the images that cannot be opened by PIL -- you need to do that. It renames the ones without extension to with extension. And finally it also downloads the captions for the downloaded images ")
    parser.add_argument("--download_captions_for_downloaded_images", action="store_true", help='If this is true, we download the captions for the already downloaded images, this is done for the older celebrity list when we did not download their text captions')
    parser.add_argument("--remove_and_rename_images", action="store_true")
    parser.add_argument("--print_most_popular_celebrities", action="store_true")
    parser.add_argument("--get_celebrity_count_distribution", action="store_true")
    parser.add_argument("--get_count_landmarks", action="store_true")
    parser.add_argument("--parallelize_across_entities", action="store_true", help="this will parallelize the process of cleaning images across CPUs for all entities. ")
    parser.add_argument("--parallelize_across_one_entity", action="store_true", help="this will parallelize the process of cleaning images across CPUs for one entity. ")
    args = parser.parse_args()

    if args.print_celebrity_name_counts:
        print_celebrity_name_counts(args)
    
    elif args.get_celebrity_first_names_total_count:
        get_celebrity_first_names_total_count()

    elif args.download_images_celebs:
        # find the list of celebrities in the new_celebrity_list that do not have a directory in "downloaded_images"
        missing_celebrities = []
        for celebrity in less_popular_celebrity_list:
            if not os.path.exists(os.path.join("downloaded_images", celebrity)):
                missing_celebrities.append(celebrity)
            else:
                # print("We already have images for ", celebrity)
                pass

        print(missing_celebrities, len(missing_celebrities), len(less_popular_celebrity_list))
        download_celebrity_images(missing_celebrities)
    
    elif args.download_captions_for_downloaded_images:
        download_celebrity_images(new_celebrity_list, only_download_captions_for_already_downloaded_images=False)

    elif args.remove_and_rename_images:
        assert args.parallelize_across_entities or args.parallelize_across_one_entity
        max_cores = 80
        assert args.set_of_people is not None
        if args.set_of_people == "celebrities":
            if not args.consider_alias:
                celebrityyyy = to_be_added_celebrities + celebrity_list + celebrity_list0 + celebrities_with_few_images
                # celebrityyyy = ['Bridgers', 'Dynevor', 'Hellqvist', 'Speedman', 'Sartorius', 'Stegner', 'Biermann', 'Boneta', 'Sawa', 'Gacy', 'Ryan', 'Thomas', 'Simpson', 'Leslie', 'Russell', 'Maxwell', 'Harvey', 'Buffer', 'Price', 'Evans']
                celebrityyyy = ['Danneel', 'Normani', 'Penélope', 'Avani', 'Cacee', 'Simu', 'Michiel', 'Léa', 'Skai', 'Selita', 'Forest', 'Samuel', 'David', 'Joe', 'Nick', 'Nicole', 'Steve', 'John', 'Cody', "Olivia"]
                # remove_images_that_cannot_be_opened_by_pil(args, celebrityyyy, download_directory="all_downloaded_images", max_cores=max_cores)
                if args.parallelize_across_entities:
                    remove_images_that_cannot_be_opened_by_pil(args, celebrityyyy, download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images", max_cores=max_cores)
                elif args.parallelize_across_one_entity:
                    for celebrity in celebrityyyy:
                        remove_images_that_cannot_be_opened_by_pil(args, [celebrity], download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images", max_cores=max_cores)
                else:
                    raise NotImplementedError
            else:
                alias_df = pd.read_csv("celebrity_data/celebrity_counts_with_aliases.csv", sep="|")     ##celeb|full_name_count|aliases_and_counts|alias_total_count|Outlier
                all_aliases = alias_df['aliases_and_counts'].tolist()       ## this is a column where everything is a dictionary (as a string)
                all_aliases = [eval(alias) for alias in all_aliases]        ## convert the string to a dictionary
                assert [type(alias) for alias in all_aliases] == [dict]*len(all_aliases)
                all_aliases = [alias for alias_dict in all_aliases for alias in alias_dict.keys()]
                # all_aliases = ['Kool Herc', 'Kool DJ Herc', 'Clive Campbell', 'M. Walker', 'Yamiche', 'Koker', 'Kirschner', 'Mary Belle', 'Varona', 'Bari Weiss', 'Keeshan', 'Ramakrishnan', 'Maitreyi', 'Bregje', 'Heinen', 'Stoller', 'Winkler', 'Brinkley', 'Ferreira', 'Angie', 'Jackson', 'Shenae Grimes', 'Skarsgård']
                all_aliases = ['Walker', 'Summer', 'Delphine', 'Belle', 'Barry', 'Babygirl', 'Marjani', 'David', 'Weiss', 'Irwin'][::-1]
                print(len(all_aliases))
                if args.parallelize_across_entities:
                    remove_images_that_cannot_be_opened_by_pil(args, all_aliases, download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images", max_cores=max_cores)
                elif args.parallelize_across_one_entity:
                    for alias in all_aliases:
                        remove_images_that_cannot_be_opened_by_pil(args, [alias], download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images", max_cores=max_cores)
                else:
                    raise NotImplementedError
                
        elif args.set_of_people == "politicians":
            politicians_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")       # Name,counts_in_laion2b-en
            politicians = politicians_df['Name'].tolist()
            assert len(politicians) == 420
            if args.parallelize_across_entities:
                ## here get the ones with counts_in_laion2b-en < 2500
                politicians = politicians_df[politicians_df['counts_in_laion2b-en'] < 2500]['Name'].tolist()
                print(len(politicians))
                remove_images_that_cannot_be_opened_by_pil(args, politicians, download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images_politicians", max_cores=max_cores)
            elif args.parallelize_across_one_entity:
                ## here get the ones with counts_in_laion2b-en >= 2500
                politicians = politicians_df[politicians_df['counts_in_laion2b-en'] >= 2500]['Name'].tolist()
                print(len(politicians), politicians)
                for politician in politicians:
                    remove_images_that_cannot_be_opened_by_pil(args, [politician], download_directory="/gscratch/scrubbed/vsahil/all_downloaded_images_politicians", max_cores=max_cores)
            else:
                raise NotImplementedError
            
        elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
            # alias_df = pd.read_csv("art_styles/style_similarity_somepalli/artists_to_analyze.csv")
            # alias_df = alias_df.rename(columns={"counts_in_laion2b-en": "full_name_count"})
            # print(len(alias_df[alias_df['full_name_count'] == 0]), "number of artists with 0 caption count")
            #     ## get the number of celebs between 0 and 100 caption count
            # print(len(alias_df[(alias_df['full_name_count'] > 0) & (alias_df['full_name_count'] <= 100)]), "number of artists between 0 and 100 caption count")
            # print(len(alias_df[(alias_df['full_name_count'] > 100) & (alias_df['full_name_count'] <= 500)]), "number of artists between 100 and 500 caption count")
            # print(len(alias_df[(alias_df['full_name_count'] > 500) & (alias_df['full_name_count'] <= 1000)]), "number of artists between 500 and 1000 caption count")
            # print(len(alias_df[(alias_df['full_name_count'] > 1000) & (alias_df['full_name_count'] <= 5000)]), "number of artists between 1000 and 5000 caption count")
            # ## greater than 5000
            # print(len(alias_df[alias_df['full_name_count'] > 5000]), "number of artists with caption count greater than 5000")
            # exit()
            # if args.set_of_people == "wikiart_artists":
            wikiart_df1 = pd.read_csv("art_styles/style_similarity_somepalli/artists_to_analyze.csv")
            if args.set_of_people == "artists_group1":
                wikiart_df = pd.read_csv("art_styles/style_similarity_somepalli/final_artists_group1.csv")
            elif args.set_of_people == "artists_group2":
                wikiart_df = pd.read_csv("art_styles/style_similarity_somepalli/final_artists_group2.csv")
            wikiart_artists = wikiart_df['artist_name'].tolist()
            ## only keep the artists in wikiart_artists that are not in wikiart_df1
            wikiart_artists = [artist for artist in wikiart_artists if artist not in wikiart_df1['artist_name'].tolist()]
            print(len(wikiart_artists))
            if args.parallelize_across_entities:
                remove_images_that_cannot_be_opened_by_pil(args, wikiart_artists, download_directory="art_styles/style_similarity_somepalli/all_artists_images", max_cores=max_cores)
            elif args.parallelize_across_one_entity:
                for artist in wikiart_artists:
                    remove_images_that_cannot_be_opened_by_pil(args, [artist], download_directory="art_styles/style_similarity_somepalli/all_artists_images", max_cores=max_cores)
            else:
                raise NotImplementedError
        
        elif args.set_of_people == "wikiart_images":
            wikiart_styles = os.listdir("art_styles/style_similarity_somepalli/wikiart_images")
            assert len(wikiart_styles) == 27
            remove_images_that_cannot_be_opened_by_pil(args, wikiart_styles, download_directory="art_styles/style_similarity_somepalli/wikiart_images", max_cores=max_cores)
        
        elif args.set_of_people == "caption_assumption":
            entities = ["and", "the", "a", "an", "in", "for", "on", "at", "by", "to", "of", "it", "as"]
            output_folder = "all_downloaded_images_for_captions"
            output_folder = "/gscratch/scrubbed/vsahil/" + output_folder
            for entity in entities:
                remove_images_that_cannot_be_opened_by_pil(args, [entity], download_directory=output_folder, max_cores=max_cores)

        else:
            raise NotImplementedError

    elif args.print_most_popular_celebrities:
        print_most_popular_celebrities()

    elif args.get_celebrity_count_distribution:
        full_celebrity_list = celebrities_with_few_images + celebrity_list + celebrity_list0
        get_celebrity_count_distribution(args, full_celebrity_list)

    elif args.get_count_landmarks:
        count_landmarks(args)

    else:
        raise NotImplementedError
