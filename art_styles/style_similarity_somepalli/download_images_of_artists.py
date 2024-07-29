import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
from unidecode import unidecode


def download_images_from_section(main_url, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Send a request to the URL
    response = requests.get(main_url)
    if response.status_code != 200:
        print(f"Failed to fetch the webpage for {main_url}")
        return
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the specific div with artworks
    artworks_div = soup.find_all('div', {"ng-controller": "MasonryCtrl"})[0]
    
    # Extracting the JSON-like structure from 'ng-init' attribute
    raw_data = artworks_div['ng-init']
    
    # Use regular expression to find all image URLs
    image_urls = re.findall(r'https?://[^"]+\.jpg', raw_data)

    # Download the first 20 images based on these URLs
    for i, img_url in enumerate(image_urls[:200], start=1):
        try:
            img_data = requests.get(img_url).content
            img_name = f"image_{i}.jpg"
            with open(os.path.join(output_folder, img_name), 'wb') as file:
                file.write(img_data)
            print(f"Downloaded {img_name}")
        except Exception as e:
            print(f"Failed to download image {i}: {e}")


def download_images_from_text_list(url2, output_folder, max_images_to_download=50):
    # Fetch the page with the list of artworks
    response = requests.get(url2)
    if response.status_code != 200:
        print(f"Failed to fetch the list page for {url2}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the specific 'ul' containing the list of artwork links
    artworks_ul = soup.select_one('body > div:nth-of-type(1) > div:nth-of-type(1) > section > main > div:nth-of-type(2) > ul')
    if not artworks_ul:
        print("Failed to find the artworks list on the page")
        return

    # Find all 'a' tags within this 'ul' that contain links to individual artworks
    artwork_links = artworks_ul.find_all('a', href=True)
    
    # Prepare full URLs for each artwork page
    image_pages = [f"https://www.wikiart.org{link['href']}" for link in artwork_links]
    
    print(f"Found {len(image_pages)} artwork pages to download for {output_folder}")

    ## if there are already len(image_pages) or 50 images in the output_folder, skip downloading for this artist
    if len(os.listdir(output_folder)) >= min(len(image_pages), max_images_to_download):
    # if len(os.listdir(output_folder)) >= max_images_to_download:
        print(f"Skipping downloading for {output_folder} as there are already {len(os.listdir(output_folder))} images in the folder.")
        return
    
    # Limit to the first 20 artwork links
    for i, page_url in enumerate(image_pages[:max_images_to_download], start=1):
        try:
            # Fetch each artwork page
            page_response = requests.get(page_url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            # Use the specific path to locate the image
            image_tag = page_soup.select_one('body > div:nth-of-type(2) > div:nth-of-type(1) > section:nth-of-type(1) > main > div:nth-of-type(2) > aside > div:nth-of-type(1) > img')
            if image_tag and 'src' in image_tag.attrs:
                image_url = image_tag['src']
                
                # Download the image
                img_data = requests.get(image_url).content
                img_name = f"image_{i}.jpg"
                with open(os.path.join(output_folder, img_name), 'wb') as file:
                    file.write(img_data)
                print(f"Downloaded {img_name}")
        except Exception as e:
            print(f"Failed to download image {i} from {page_url}: {e}")


def download_images_from_wikiart(download_images_from_section, download_images_from_text_list, artist_df, use_url1, use_url2, executor):
    ## wikiarts has some of the artists named incorrectly and therefore the url download fails, here is the dictionary from correct names to incorrect names to download images, and then replace back. 
    # replacement_list = {'joaquín sorolla': "joaqu-n-sorolla", 'andré gill': "andre-gill", 'françois gérard': "francois-gerard", 'pedro américo': 'pedro-americo', 'jean-étienne liotard': "jean-etienne-liotard", ''}
    replacement_list = {'apollinary goravsky': 'apollinariy-goravskiy', 'petro kholodny': 'petro-kholodny-elder', 'alexei korzukhin': 'aleksey-ivanovich-korzukhin', 'jérôme-martin langlois': 'jerome-martin-langlois'}
    
    for artist_name in artist_df["artist_name"].tolist():
        if artist_name in replacement_list:
            artist_name = replacement_list[artist_name]
        else:
            if "'" in artist_name:
                artist_name = artist_name.replace("'", ' ')
            if "." in artist_name:
                artist_name = artist_name.replace(".", ' ')
        
            artist_name = unidecode(artist_name.strip().lower().replace('   ', '-').replace('   ', '-').replace('  ', '-').replace(' ', '-'))
        
        # URL of the page to scrape
        url1 = f"https://www.wikiart.org/en/{artist_name}"
        url2 = f"https://www.wikiart.org/en/{artist_name}/all-works/text-list"

        # # Folder to save the downloaded images
        output_dir = f"wikiart_images_downloaded/{artist_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        ## if there are already 11 or more images downloaded for an artist in the output_dir, skip downloading for this artist
        if len(os.listdir(output_dir)) >= 11:
            print(f"Skipping downloading for {artist_name} as there are already {len(os.listdir(output_dir))} images in the folder.")
            continue

        print(f"Downloading images for {artist_name} at {output_dir}, {use_url1}, {use_url2}")
        
        if executor:
            if use_url1:
                executor.submit(download_images_from_section, url1, output_dir)
            elif use_url2:
                executor.submit(download_images_from_text_list, url2, output_dir)
        else:
            if use_url1:
                download_images_from_section(url1, output_dir)
            elif use_url2:
                download_images_from_text_list(url2, output_dir)


if __name__ == "__main__":
    # artist_df = pd.read_csv("artists_to_analyze.csv")       ## artist_name,counts_in_laion2b-en,counts_in_wikiart,count_paintings
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_url1", action="store_true", help="Use the first URL to download images")
    parser.add_argument("--use_url2", action="store_true", help="Use the second URL to download images")
    parser.add_argument("--get_downloaded_image_count", action="store_true", help="Get the count of downloaded images for each artist")
    parser.add_argument("--get_low_count_artists", action="store_true", help="Get the list of artists with counts_in_wikiart < 11")
    parser.add_argument("--group", choices=["artist_group1", "artist_group2"], help="Group of artists to download images for which group", required=True)
    args = parser.parse_args()
    
    ## we have now decided to download images for all the artists from wikiarts website instead of just the artists with counts_in_wikiart < 11

    assert sum([args.use_url1, args.use_url2, args.get_downloaded_image_count, args.get_low_count_artists]) == 1, "Exactly one of use_url1, use_url2, get_downloaded_image_count, get_low_count_artists must be True"
    
    if args.group == "artist_group1":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
    elif args.group == "artist_group2":
        artist_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
    else:
        raise NotImplementedError
    
    if args.get_downloaded_image_count:
        replacement_list = {'apollinary goravsky': 'apollinariy-goravskiy', 'petro kholodny': 'petro-kholodny-elder', 'alexei korzukhin': 'aleksey-ivanovich-korzukhin', 'jérôme-martin langlois': 'jerome-martin-langlois'}
        ## Add the downloaded images to the artist_df, for all artists having counts_in_wikiart < 11, get the number of images downloaded, and for rest the downloaded images is 0. 
        total_images = {}
        downloaded_images = {}
        for artist_name in artist_df["artist_name"].tolist():
            # this_artist_images_in_wikiart = artist_df[artist_df["artist_name"] == artist_name]["counts_in_wikiart"].values[0]
            if artist_name in replacement_list:
                artist_name_downloaded_folder = replacement_list[artist_name]
            else:
                artist_name_downloaded_folder = unidecode(artist_name.strip().lower().replace("'", ' ').replace(".", ' ').replace('   ', '-').replace('   ', '-').replace('  ', '-').replace(' ', '-'))
            downloaded_images_folder = f"wikiart_images_downloaded/{artist_name_downloaded_folder}"
            if not os.path.exists(downloaded_images_folder):
                downloaded_images[artist_name] = 0
            else:
                ## get the images inside the downloaded_images_folder, these must end with .jpg
                images = [f for f in os.listdir(downloaded_images_folder) if f.endswith(".jpg")]
                downloaded_images[artist_name] = len(images)
            print(f"Artist: {artist_name} had {len(images)} images downloaded.")
            # total_images[artist_name] = this_artist_images_in_wikiart + downloaded_images[artist_name]
        artist_df["downloaded_images_from_wikiart_website"] = artist_df["artist_name"].map(downloaded_images)
        # artist_df["total_images"] = artist_df["artist_name"].map(total_images)
        # artist_df.to_csv("artists_to_analyze.csv", index=False)
        if args.group == "artist_group1":
            artist_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv", index=False)
        elif args.group == "artist_group2":
            artist_df.to_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv", index=False)
        else:
            raise NotImplementedError
        
    elif args.get_low_count_artists:
        ## get the name of all artists whose counts_in_wikiart is less than 11
        artists = artist_df[artist_df["downloaded_images_from_wikiart_website"] < 8]
        print(artists["artist_name"].tolist(), len(artists["artist_name"].tolist()))
        ## print the name of the artist and their number of images downloaded and counts_in_wikiart
        # for artist_name in artists["artist_name"].tolist():
        #     print(f"Artist: {artist_name}, downloaded_images: {artist_df[artist_df['artist_name'] == artist_name]['downloaded_images_from_wikiart_website'].values[0]}, counts_in_wikiart: {artist_df[artist_df['artist_name'] == artist_name]['counts_in_wikiart'].values[0]}")
        ## remove these 6 artists from the artists_to_analyze.csv file
        # artist_df = artist_df[artist_df["downloaded_images_from_wikiart_website"] >= 11]
        # artist_df.to_csv("artists_to_analyze.csv", index=False)
    
    else:
        artist_df = artist_df[artist_df["artist_name"].isin(['johannes vermeer', 'john trumbull', 'erin hanson'])] # -- used url1 for them as url2 was not working 
        print(artist_df)
        num_threads = 40
        if num_threads > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                download_images_from_wikiart(download_images_from_section, download_images_from_text_list, artist_df, args.use_url1, args.use_url2, executor)
        else:
            download_images_from_wikiart(download_images_from_section, download_images_from_text_list, artist_df, args.use_url1, args.use_url2, None)
