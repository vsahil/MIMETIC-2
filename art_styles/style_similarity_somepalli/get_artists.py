import pandas as pd
import requests
from bs4 import BeautifulSoup

## Step 1: Extract the artists from the wikiart dataset
def extract_artists_from_wikiart():
    df = pd.read_csv("wikiart.csv")
    artists = df["artist"].unique()
    styles = df["label"].unique()
    print("Number of artists: ", len(artists))
    print("Number of styles: ", len(styles))
    ## get the count of each artists and save the name of the artist and the count in a csv file
    artist_count = df["artist"].value_counts()
    artist_count.to_csv("wikiart_artists.csv")

## Step2: use get_laion_captions.py to get the counts of the each artist. Save as wikiart_artists_sorted.csv. 

def sample_artists_for_analysis():
    ## now use the sorted list of artists to get the counts in laion2b of 0, between 1-100, between 100-500, between 500-1000, between 1000-5000, between 5000-10000, and above. Just print it. Use column counts_in_laion2b-en for counts
    df = pd.read_csv("wikiart_artists_sorted.csv")      ## artist_name,counts_in_laion2b-en,counts_in_wikiart
    ## remove the rows where name of artist is: "arman", "mabuse", "corneille", "erro", "menez"
    df = df[~df["artist_name"].isin(["arman", "mabuse", "corneille", "erro", "menez"])]
    
    def contains_non_english_characters(name):
        ## print the ascii value of each character in the name
        # print(name, [ord(char) for char in name])
        return not all((65 <= ord(char) <= 90) or (97 <= ord(char) <= 122) or (ord(char) == 32) for char in name)

    # Filtering artist names that contain non-standard English characters
    non_english_artists = [name for name in df['artist_name'].tolist() if contains_non_english_characters(name)]

    print(non_english_artists)

    print("Total Number of artists: ", len(df))
    print("Number of artists with 0 counts in laion2b: ", len(df[df["counts_in_laion2b-en"] == 0]))
    print("Number of artists with counts between 1-100 in laion2b: ", len(df[(df["counts_in_laion2b-en"] > 0) & (df["counts_in_laion2b-en"] <= 100)]))
    print("Number of artists with counts between 100-500 in laion2b: ", len(df[(df["counts_in_laion2b-en"] > 100) & (df["counts_in_laion2b-en"] <= 500)]))
    print("Number of artists with counts between 500-1000 in laion2b: ", len(df[(df["counts_in_laion2b-en"] > 500) & (df["counts_in_laion2b-en"] <= 1000)]))
    print("Number of artists with counts between 1000-5000 in laion2b: ", len(df[(df["counts_in_laion2b-en"] > 1000) & (df["counts_in_laion2b-en"] <= 5000)]))
    print("Number of artists with counts between 5000-10000 in laion2b: ", len(df[(df["counts_in_laion2b-en"] > 5000) & (df["counts_in_laion2b-en"] <= 10000)]))
    print("Number of artists with counts above 10000 in laion2b: ", len(df[df["counts_in_laion2b-en"] > 10000]))

    ## get the name and count of all artists with one word name only split by space
    # df["name_split"] = df["artist_name"].apply(lambda x: x.split())
    # df["name_len"] = df["name_split"].apply(lambda x: len(x))
    # df = df[df["name_len"] == 1]
    # print(df)


    ## sample 20 artists with 0 counts in laion2b, 30 artists with counts between 1-100 in laion2b, 70 artists with counts between 100-500 in laion2b, 125 artists with counts between 500-1000 in laion2b, 70 artists with counts between 1000-5000 in laion2b, 50 artists with counts between 5000-10000 in laion2b, and 50 artists with counts above 10000 in laion2b. Total = 20+30+70+125+70+50+50 = 415. 
    # Get the counts and the artists names and save them to a csv file.
    ## set the seed for reproducibility
    import numpy as np
    np.random.seed(42)

    artists_0 = df[df["counts_in_laion2b-en"] == 0]
    artists_0 = artists_0.sample(20)
    assert len(artists_0) == 20
    artists_1_100 = df[(df["counts_in_laion2b-en"] > 0) & (df["counts_in_laion2b-en"] <= 100)]
    artists_1_100 = artists_1_100.sample(30)
    assert len(artists_1_100) == 30
    artists_100_500 = df[(df["counts_in_laion2b-en"] > 100) & (df["counts_in_laion2b-en"] <= 500)]
    artists_100_500 = artists_100_500.sample(70)
    assert len(artists_100_500) == 70
    artists_500_1000 = df[(df["counts_in_laion2b-en"] > 500) & (df["counts_in_laion2b-en"] <= 1000)]
    artists_500_1000 = artists_500_1000.sample(125)
    assert len(artists_500_1000) == 125
    artists_1000_5000 = df[(df["counts_in_laion2b-en"] > 1000) & (df["counts_in_laion2b-en"] <= 5000)]
    artists_1000_5000 = artists_1000_5000.sample(70)
    assert len(artists_1000_5000) == 70
    artists_5000_10000 = df[(df["counts_in_laion2b-en"] > 5000) & (df["counts_in_laion2b-en"] <= 10000)]
    artists_5000_10000 = artists_5000_10000.sample(50)
    assert len(artists_5000_10000) == 50
    artists_above_10000 = df[df["counts_in_laion2b-en"] > 10000]
    artists_above_10000 = artists_above_10000.sample(50)
    assert len(artists_above_10000) == 50

    artists = pd.concat([artists_0, artists_1_100, artists_100_500, artists_500_1000, artists_1000_5000, artists_5000_10000, artists_above_10000])
    ## sort the artists by counts_in_laion2b-en in descending order
    artists = artists.sort_values(by="counts_in_laion2b-en", ascending=False)
    ## get the sum of the counts for thses artists
    print("Total counts for the sampled artists: ", artists["counts_in_laion2b-en"].sum())
    artists.to_csv("artists_to_analyze.csv", index=False)


def correct_number_of_wikiart_images():
    df = pd.read_csv("artists_to_analyze.csv")      ## artist_name,counts_in_laion2b-en,counts_in_wikiart,count_paintings
    ## correct the number of counts_in_wikiart for each artist in the artists_to_analyze.csv file. Copy the value from "wikiart_artists.csv"
    wikiart_artists = pd.read_csv("wikiart_artists.csv")        ## artist,count
    wikiart_artists = wikiart_artists[["artist", "count"]]
    wikiart_artists.columns = ["artist_name", "counts_in_wikiart"]
    ## I think we should replace the old counts_in_wikiart with the new counts_in_wikiart. Currently I am getting two columns: counts_in_wikiart_x and counts_in_wikiart_y. I will remove the old column and keep the new one.
    df = df.drop(columns=["counts_in_wikiart"])
    df = pd.merge(df, wikiart_artists, on="artist_name", how="left")
    df.to_csv("artists_to_analyze.csv", index=False)


def get_style_each_artist():
    df1 = pd.read_csv("wikiart.csv")       ## label,artist,name,split
    ## here we have labels for each art work (name), we want to get the label for each artist, which we will do by majority aggregation across their art works. 
    ## get the count of each label for each artist
    df2 = df1.groupby(["artist", "label"]).size().reset_index(name="count")
    ## get the label with the maximum count for each artist
    df2 = df2.sort_values(by=["artist", "count"], ascending=False)
    import ipdb; ipdb.set_trace()
    df2 = df2.drop_duplicates(subset="artist", keep="first")
    df2 = df2[["artist", "label"]]
    df2.columns = ["artist_name", "style"]
    df2.to_csv("wikiart_artists_sorted2.csv", index=False)
    

def merge_files_to_get_style():
    df1 = pd.read_csv("wikiart_artists_sorted.csv")     ## artist_name,counts_in_laion2b-en,counts_in_wikiart
    df2 = pd.read_csv("wikiart_artists_sorted2.csv")        ## artist_name,style
    
    ## assert that all the artist_name in df1 are in artist column in df2
    assert all(df1["artist_name"].isin(df2["artist_name"]))
    
    ## and then merge the two dfs, such that we get the label for each artist from df2 in df1
    ## make sure the length of the merged df is same as the length of df1
    df = pd.merge(df1, df2, on="artist_name", how="inner")
    assert len(df) == len(df1), f"Length of df: {len(df)} and df1: {len(df1)} are not equal."
    ## assert that df and df1 only differ in one column
    df_temp = df.drop(columns=["style"])
    assert df_temp.equals(df1)
    df.to_csv("wikiart_artists_sorted.csv", index=False)


def get_sampled_artist_style():
    sampled_df = pd.read_csv("artists_to_analyze.csv")      ## artist_name,counts_in_laion2b-en,count_paintings_in_laion_images,counts_in_wikiart_original_dataset,downloaded_images_from_wikiart_website,count_this_artist_paintings_in_laion_images
    print(sampled_df['style'].value_counts())
    exit()
    
    ## get the style for each artist in the sampled_df
    df = pd.read_csv("wikiart_artists_sorted.csv")       ## artist_name,counts_in_laion2b-en,counts_in_wikiart,style
    df = df[["artist_name", "style"]]
    ## assert that all the artist_name in sampled_df are in artist column in df
    assert all(sampled_df["artist_name"].isin(df["artist_name"])), f"these artists are in sampled df but not in df: {sampled_df[~sampled_df['artist_name'].isin(df['artist_name'])]}"
    ## and then merge the two dfs, such that we get the label for each artist from df in sampled_df
    ## make sure the length of the merged df is same as the length of sampled_df
    df = pd.merge(sampled_df, df, on="artist_name", how="inner")
    assert len(df) == len(sampled_df), f"Length of df: {len(df)} and sampled_df: {len(sampled_df)} are not equal."
    ## assert that df and sampled_df only differ in one column
    df_temp = df.drop(columns=["style"])
    assert df_temp.equals(sampled_df)
    # df.to_csv("artists_to_analyze.csv", index=False)
    
    ## get the count of each style in the df
    style_count = df["style"].value_counts()
    print(style_count)


def get_artists_from_wikiarts_website():
    ## we will sample artists from the wikiarts webiste that are not among the art styles that we have already sampled. Expressionism, Impressionism, Romanticism, Realism, Minimalism, Abstract Expressionism, Pop Art, Baroque, Cubism, Post Impressionism, Symbolism, Rococo, Color Field Painting
    # Also do not sample artists from the styles that are not paintings like: , Art Nouveau Modern, Ukiyo e, New Realism
    ## Remaining popular art styles: Surrealism, Contemporary, Neo-Expressionism, Post-Minimalism, Neoclassicism, 
    
    ## we will make two groups of art styles and get artists for each group.
    
    # Group 1 (Traditional and Historical Focus)
        # Romanticism
        # Impressionism
        # Realism
        # Baroque
        # Neoclassicism
        # Rococo
        # Academic Art
        # Symbolism
        # Cubism
        # Post-Impressionism

    # Group 2 (Modern and Abstract Focus)
        # Expressionism
        # Surrealism
        # Abstract Expressionism
        # Pop Art
        # Art Informel
        # Post-Painterly Abstraction
        # Neo-Expressionism
        # Post-Minimalism
        # Neo-Impressionism
        # Neo-Romanticism
        
    ## download the names of the artists from wikiarts website for these art styles.
    def get_artists_by_movement(url):
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the <ul> tag containing the list of artists
            artist_list = soup.select_one('main > div:nth-of-type(3) > ul')
            
            # Extract the text from each <li> tag within the <ul>
            artist_names = [li.text.strip() for li in artist_list.find_all('li')]
            
            return artist_names
        else:
            print("Failed to retrieve data. Status code:", response.status_code)


    group1_styles = ['Romanticism', 'Impressionism', 'Realism', 'Baroque', 'Neoclassicism', 'Rococo', 'Academic Art', 'Symbolism', 'Cubism', 'Naturalism']
    group2_styles = ['Expressionism', 'Surrealism', 'Abstract Expressionism', 'Pop Art', 'Art Informel', 'Post-Painterly Abstraction', 'Neo-Expressionism', 'Post-Minimalism', 'Neo-Impressionism', 'Neo-Romanticism', 'Post-Impressionism']
    
    artists_in_this_group = []
    for style in group1_styles:
        # URL of the art movement page
        url = f'https://www.wikiart.org/en/artists-by-art-movement/{style.lower().replace(" ", "-")}/text-list'
        # Get the list of artists
        artists = get_artists_by_movement(url)
        if artists:
            ## each element in the list is like 'George Stubbs\n, British\n, 1724 - 1806\n, 82 artworks', we need to get the name and the number of art works of the artist and print in a csv file.
            list_of_artists = []
            for artist in artists:
                artist_name = artist.split("\n")[0]
                num_artworks = artist.split("\n")[-1]
                assert "artwork" in num_artworks, f"{artist}, {num_artworks}"
                num_artworks = int(num_artworks.split()[-2])
                list_of_artists.append((artist_name, num_artworks))
        else:
            print("No artists found or an error occurred.")

        artists_in_this_group.extend(list_of_artists)
        print(len(list_of_artists), " in ", style, " Total artists till now: ", len(artists_in_this_group))
    
    print(artists_in_this_group, len(artists_in_this_group))
    ## save the artists_in_this_group to a csv file: first column is the artist name and the second column is their number of art works
    df = pd.DataFrame(artists_in_this_group, columns=["artist_name", "count_paintings"])
    df.to_csv("artists_in_group1.csv", index=False)


def get_artist_counts_in_the_two_groups():
    ## task 1, drop the names of the artists that have less than 8 artworks 
    group1_df = pd.read_csv("artists_in_group1.csv")        ## artist_name,count_paintings
    ## drop the duplicate rows
    group2_df = pd.read_csv("artists_in_group2.csv")        ## artist_name,count_paintings
    # group2_df = group2_df.drop_duplicates(subset="artist_name", keep="first")
    # group2_df.to_csv("artists_in_group2.csv", index=False)
    # group1_df = group1_df[group1_df["count_paintings"] >= 8]
    # group2_df = group2_df[group2_df["count_paintings"] >= 8]
    # group1_df.to_csv("artists_in_group1.csv", index=False)
    # group2_df.to_csv("artists_in_group2.csv", index=False)
    
    ## get counts in the LAION2B dataset for these artists.
    import sys
    sys.path.append("../../")
    from extract_from_laion_database import count_documents_containing_phrases
    import extract_from_laion_database as laion_access
    es = laion_access.es_init()
    
    artist_name_count = []
    for seq, artist in enumerate(group2_df["artist_name"].tolist()):
        count_this_name = count_documents_containing_phrases("re_laion2b-en-*", artist, es=es)
        artist_name_count.append((artist, count_this_name))
        if seq % 20 == 0:
            print(f"counted {seq} artists")
    
    ## create a new column in group1_df with the count of their names
    artist_name_count_df = pd.DataFrame(artist_name_count, columns=["artist_name", "count_this_artist_paintings"])
    assert artist_name_count_df.shape == group2_df.shape
    df = pd.merge(group2_df, artist_name_count_df, on="artist_name", how="inner")
    assert df.shape[0] == group2_df.shape[0], f"Shape of df: {df.shape} and group1_df: {group2_df.shape} are not equal."
    df.to_csv("artists_in_group2.csv", index=False)
    

def sample_artists_for_both_groups():
    group1_df = pd.read_csv("artists_in_group1.csv")    ## artist_name,count_wikiart_artworks,counts_in_laion2b-en
    group2_df = pd.read_csv("artists_in_group2.csv")    ## artist_name,count_wikiart_artworks,counts_in_laion2b-en
    ## find the artists that are in both group1_df and group2_df
    # common_artists = group1_df[group1_df["artist_name"].isin(group2_df["artist_name"])]
    # print(common_artists, group1_df.shape, group2_df.shape)
    ## remove all the common artists from group1_df
    # group1_df = group1_df[~group1_df["artist_name"].isin(common_artists["artist_name"])]
    # group1_df.to_csv("artists_in_group1.csv", index=False)
    ## drop all the artists in group1 df whose count_wikiart_artworks is less than 8
    # group1_df = group1_df[group1_df["count_wikiart_artworks"] >= 8]
    # group1_df.to_csv("artists_in_group1.csv", index=False)
    # ## drop all the artists in group2 df whose count_wikiart_artworks is less than 8
    # group2_df = group2_df[group2_df["count_wikiart_artworks"] >= 8]
    # group2_df.to_csv("artists_in_group2.csv", index=False)
    
    already_done_artists = pd.read_csv("artists_to_analyze.csv")    ## artist_name,counts_in_laion2b-en, and many other columns
    already_done_artists = already_done_artists[["artist_name","counts_in_laion2b-en"]]
    
    # Convert artist names in both DataFrames to lowercase for case-insensitive comparison
    group1_df['artist_name'] = group1_df['artist_name'].str.lower()
    group2_df['artist_name'] = group2_df['artist_name'].str.lower()
    already_done_artists['artist_name'] = already_done_artists['artist_name'].str.lower()
    ## find the artists in already done that are in group1, convert the artist name in group1_df to lower
    already_done_group1 = already_done_artists[already_done_artists["artist_name"].isin(group1_df["artist_name"])]
    # print(already_done_group1)
    ## find the artists in already done that are in group2
    already_done_group2 = already_done_artists[already_done_artists["artist_name"].isin(group2_df["artist_name"])]
    # print(already_done_group2)    
    
    def sample_remaining_artists(df, already_done):
        ## remove artists from df that are in already_done
        df = df[~df["artist_name"].isin(already_done["artist_name"])]
            
        ## sample 15 artists with 0 counts in laion2b, 60 artists with counts between 1-100 in laion2b, 120 artists with counts between 100-500 in laion2b, 80 artists with counts between 500-1000 in laion2b, 65 artists with counts between 1000-5000 in laion2b, 40 artists with counts between 5000-10000 in laion2b, and 40 artists with counts above 10000 in laion2b. Total = 15+60+120+80+65+40+40 = 420.

        ## get the counts range of the artists in already_done list, so that we only need to sample the remaining artists.
        ## convert "counts_in_laion2b-en" to count range for this task
        already_done["count_range"] = pd.cut(already_done["counts_in_laion2b-en"], bins=[0, 1, 100, 500, 1000, 5000, 10000, float('inf')], labels = ["0", "1-100", "100-500", "500-1000", "1000-5000", "5000-10000", "above 10000"], right=False)
        # print(already_done["count_range"].value_counts())
        
        more_artists_to_sample_with_0_count = 15 - len(already_done[already_done["count_range"] == "0"])
        more_artists_to_sample_with_1_to_100_count = 70 - len(already_done[already_done["count_range"] == "1-100"])
        more_artists_to_sample_with_100_to_500_count = 140 - len(already_done[already_done["count_range"] == "100-500"])
        more_artists_to_sample_with_500_to_1000_count = 65 - len(already_done[already_done["count_range"] == "500-1000"])
        more_artists_to_sample_with_1000_to_5000_count = 65 - len(already_done[already_done["count_range"] == "1000-5000"])
        more_artists_to_sample_with_5000_to_10000_count = 40 - len(already_done[already_done["count_range"] == "5000-10000"])
        more_artists_to_sample_with_above_10K_count = 40 - len(already_done[already_done["count_range"] == "above 10000"])
        
        # print(more_artists_to_sample_with_0_count, more_artists_to_sample_with_1_to_100_count, more_artists_to_sample_with_100_to_500_count, more_artists_to_sample_with_500_to_1000_count, more_artists_to_sample_with_1000_to_5000_count, more_artists_to_sample_with_5000_to_10000_count, more_artists_to_sample_with_above_10K_count)

        ## set the seed for reproducibility
        import numpy as np
        np.random.seed(0)
        
        artists_0 = df[df["counts_in_laion2b-en"] == 0]
        artists_0 = artists_0.sample(more_artists_to_sample_with_0_count)
        # assert len(artists_0) == 20
        artists_1_100 = df[(df["counts_in_laion2b-en"] > 0) & (df["counts_in_laion2b-en"] <= 100)]
        artists_1_100 = artists_1_100.sample(more_artists_to_sample_with_1_to_100_count)
        # assert len(artists_1_100) == 30
        artists_100_500 = df[(df["counts_in_laion2b-en"] > 100) & (df["counts_in_laion2b-en"] <= 500)]
        artists_100_500 = artists_100_500.sample(more_artists_to_sample_with_100_to_500_count)
        # assert len(artists_100_500) == 70
        artists_500_1000 = df[(df["counts_in_laion2b-en"] > 500) & (df["counts_in_laion2b-en"] <= 1000)]
        artists_500_1000 = artists_500_1000.sample(min(more_artists_to_sample_with_500_to_1000_count, len(artists_500_1000)))
        # assert len(artists_500_1000) == 125
        artists_1000_5000 = df[(df["counts_in_laion2b-en"] > 1000) & (df["counts_in_laion2b-en"] <= 5000)]
        artists_1000_5000 = artists_1000_5000.sample(more_artists_to_sample_with_1000_to_5000_count)
        # assert len(artists_1000_5000) == 70
        artists_5000_10000 = df[(df["counts_in_laion2b-en"] > 5000) & (df["counts_in_laion2b-en"] <= 10000)]
        artists_5000_10000 = artists_5000_10000.sample(min(more_artists_to_sample_with_5000_to_10000_count, len(artists_5000_10000)))
        # assert len(artists_5000_10000) == 50
        artists_above_10000 = df[df["counts_in_laion2b-en"] > 10000]
        artists_above_10000 = artists_above_10000.sample(min(more_artists_to_sample_with_above_10K_count, len(artists_above_10000)))
        # assert len(artists_above_10000) == 50
        
        # print("Total artists with count 0: ", len(artists_0) + len(already_done[already_done["count_range"] == "0"]))
        # print("Total artists with count between 1 and 100:", len(artists_1_100) + len(already_done[already_done["count_range"] == "1-100"]))
        # print("Total artists with count between 100 and 500:", len(artists_100_500) + len(already_done[already_done["count_range"] == "100-500"]))
        # print("Total artists with count between 500 and 1000:", len(artists_500_1000) + len(already_done[already_done["count_range"] == "500-1000"]))
        # print("Total artists with count between 1000 and 5000:", len(artists_1000_5000) + len(already_done[already_done["count_range"] == "1000-5000"]))
        # print("Total artists with count between 5000 and 10000:", len(artists_5000_10000) + len(already_done[already_done["count_range"] == "5000-10000"]))
        # print("Total artists with count above 10000:", len(artists_above_10000) + len(already_done[already_done["count_range"] == "above 10000"]))

        artists_full = pd.concat([artists_0, artists_1_100, artists_100_500, artists_500_1000, artists_1000_5000, artists_5000_10000, artists_above_10000, already_done])
        
        ## make count range a column in artists_full
        artists_full["count_range"] = pd.cut(artists_full["counts_in_laion2b-en"], bins=[0, 1, 100, 500, 1000, 5000, 10000, float('inf')], labels = ["0", "1-100", "100-500", "500-1000", "1000-5000", "5000-10000", "above 10000"], right=False)
        print(artists_full["count_range"].value_counts())
        return artists_full
        
    
    sampled_group1 = sample_remaining_artists(group1_df, already_done_group1)
    sampled_group1 = sampled_group1.sort_values(by="counts_in_laion2b-en", ascending=False)
    sampled_group1.to_csv("final_artists_group1.csv", index=False)
    
    sampled_group2 = sample_remaining_artists(group2_df, already_done_group2)
    sampled_group2 = sampled_group2.sort_values(by="counts_in_laion2b-en", ascending=False)
    sampled_group2.to_csv("final_artists_group2.csv", index=False)
    

def print_artists_names():
    group1_df = pd.read_csv("final_artists_group1.csv")    ## artist_name,count_wikiart_artworks,counts_in_laion2b-en,count_range,count_paintings_in_laion_images,downloaded_images_from_wikiart_website,count_artworks_in_laion_images,count_this_artist_artworks_in_laion_images
    group2_df = pd.read_csv("final_artists_group2.csv")    ## artist_name,count_wikiart_artworks,counts_in_laion2b-en,count_range,count_paintings_in_laion_images,downloaded_images_from_wikiart_website,count_artworks_in_laion_images,count_this_artist_artworks_in_laion_images
    
    group_df = group2_df
    ## print just then names of artists in group1_df in descending order of counts_in_laion2b-en
    group_df = group_df.sort_values(by="counts_in_laion2b-en", ascending=False)
    # # Change the names to title case
    # group_df['artist_name'] = group_df['artist_name'].str.title()
    # ## print the names without the single quotes
    # print(", ".join(group_df["artist_name"].tolist()))
    # print the number of artists in group_df with counts_in_laion2b-en: 0, between 1-100, between 100-500, between 500-1000, between 1000-5000, between 5000-10000, and above 10000. Just print in new line without anything else. Print it in this order.
    print(len(group_df[group_df["counts_in_laion2b-en"] == 0]))
    print(len(group_df[(group_df["counts_in_laion2b-en"] > 0) & (group_df["counts_in_laion2b-en"] <= 100)]))
    print(len(group_df[(group_df["counts_in_laion2b-en"] > 100) & (group_df["counts_in_laion2b-en"] <= 500)]))
    print(len(group_df[(group_df["counts_in_laion2b-en"] > 500) & (group_df["counts_in_laion2b-en"] <= 1000)]))
    print(len(group_df[(group_df["counts_in_laion2b-en"] > 1000) & (group_df["counts_in_laion2b-en"] <= 5000)]))
    print(len(group_df[(group_df["counts_in_laion2b-en"] > 5000) & (group_df["counts_in_laion2b-en"] <= 10000)]))
    print(len(group_df[group_df["counts_in_laion2b-en"] > 10000]))
    

if __name__ == "__main__":
    # extract_artists_from_wikiart()
    # sample_artists_for_analysis()
    # correct_number_of_wikiart_images()
    # merge_files_to_get_style()
    # get_style_each_artist()
    # get_sampled_artist_style()
    # get_artists_from_wikiarts_website()
    # get_artist_counts_in_the_two_groups()
    # sample_artists_for_both_groups()
    print_artists_names()
