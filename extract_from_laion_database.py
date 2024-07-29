import logging
import os
from functools import cache
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Union
import requests
import yaml
import unicodedata, string
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
import pandas as pd
import random
random.seed(42)

logger = logging.getLogger(__name__)

# PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
DEFAULT_CONFIG_LOCATION = "/gscratch/h2lab/vsahil/vlm-efficiency/es_config.yml"

@cache
def es_init(config: Path = DEFAULT_CONFIG_LOCATION, timeout: int = 30) -> Elasticsearch:
    """
    :param config: Path to the config yaml file, containing `cloud_id` and `api_key` fields.
    :return: Authenticated ElasticSearch client.
    """
    with open(config) as file_ref:
        config = yaml.safe_load(file_ref)

    cloud_id = config["cloud_id"]
    api_key = config.get("api_key", os.getenv("ES_API_KEY", None))
    if not api_key:
        raise RuntimeError(
            f"Please specify ES_API_KEY environment variable or add api_key to {DEFAULT_CONFIG_LOCATION}."
        )

    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key,
        retry_on_timeout=True,
        http_compress=True,
        request_timeout=timeout,
    )

    return es


def get_indices(
    return_mapping: bool = False, es: Optional[Elasticsearch] = None
) -> Dict:
    """
    :param return_mapping: Whether to return mapping along with index information.
    :return: Dictionary of existing indices.
    """
    es = es or es_init()

    indices = es.cat.indices(format="json")
    exclude = [
        "search-test",
        "test-index-2",
        "metrics-endpoint.metadata_current_default",
    ]
    indices = {
        index["index"]: {key: index[key] for key in ["docs.count"]}
        for index in indices
        if not index["index"].startswith(".") and not index["index"] in exclude
    }

    if return_mapping:
        mappings = es.indices.get_mapping(index=list(indices.keys()))
        for key in mappings:
            indices[key]["properties"] = list(
                mappings[key]["mappings"]["properties"].keys()
            )

    return indices


def _query_documents_contain_phrases(
    phrases: Union[str, List[str]],
    all_phrases: bool = False,
    do_score: bool = False,
    is_regexp: bool = False,
) -> Dict:
    if isinstance(phrases, str):
        phrases = [phrases]
    if all_phrases:
        which_bool = "must" if do_score else "filter"
        minimum_should_match = None
    else:
        which_bool = "should"
        minimum_should_match = 1

    if is_regexp:
        match_query = []
        for phrase in phrases:
            match_query.append(
                {
                    "regexp": {
                        "text": {
                            "value": phrase,
                            "case_insensitive": True,
                            "flags": "ALL",
                        }
                    }
                }
            )
        # minimum_should_match = None
    else:
        match_query = []
        for phrase in phrases:
            match_query.append({"match_phrase": {"text": phrase}})

    query = {
        "bool": {which_bool: match_query, "minimum_should_match": minimum_should_match}
    }
    return query


def count_documents_containing_phrases(
    index: str,
    phrases: Union[str, List[str]],
    all_phrases: bool = False,
    is_regexp: bool = False,
    subset_filter: bool = True,
    es: Optional[Elasticsearch] = None,
) -> int:
    """
    :param index: Name of the index
    :param phrases: A single string or a list of strings to be matched in the `text` field
        of the index.
    :param all_phrases: Whether the document should contain all phrases (AND clause) or any
        of the phrases (OR clause).
    :param is_regexp: Whether the phrases are regular expressions. Note that spaces in regular
        expressions are not supported by ElasticSearch, so if you want to do an exact match for
        spans longer than a single term, set this to False. In most cases, using exp1|exp2 is better
        than specifying [exp1, exp2] as two different `phrases`.
    :return: The number of documents matching the conditions.

    Examples:

        count_documents_containing_phrases("test-index", "legal")  # single term
        count_documents_containing_phrases("test-index", ["legal", "license"])  # list of terms
        count_documents_containing_phrases("test-index", ["terms of use", "legally binding"])  # list of word sequences

        # The documents should contain both `winter` and `spring` in the text.
        count_documents_containing_phrases("test-index", ["winter", "spring"], all_phrases=True)

    """
    es = es or es_init()

    query = _query_documents_contain_phrases(phrases, all_phrases, is_regexp=is_regexp)
    if index == "c4" and subset_filter:
        if "filter" in query["bool"]:
            query["bool"]["filter"].append({"term": {"subset": "en"}})
        else:
            query["bool"]["filter"] = {"term": {"subset": "en"}}

    result = es.count(index=index, query=query)

    return result["count"]


# def multiple_count_documents_containing_phrases(
#     index: str,
#     phrases: Union[str, List[str]],
# ):
#     if isinstance(phrases, str):
#         phrases = [phrases]
#     es = es_init()
#     num_shards = len(es.cat.shards(index=index, format="json"))
#     final_counts = [0 for _ in range(len(phrases))]
#     for shard in range(num_shards):
#         # TODO: for the time being its all in memory. but we might want to do some
#         # map-reduce.
#         final_counts = []
#         queries = []
#         for phrase in phrases:
#             queries.append({"index": index, "search_type": "query_then_fetch"})
#             queries.append({"size": 0, "query": {"match_phrase": {"text": phrase}}})

#         results = es.msearch(searches=queries, search_type="query_then_fetch", routing=f"_shards:{shard}")
#         # todo: change to a generator?
#         counts = [r["hits"]["total"]["value"] for r in results["responses"]]
#         final_counts = [sum(x) for x in zip(final_counts, counts)]
#     return final_counts


# def get_documents_containing_phrases(
#     index: str,
#     phrases: Union[str, List[str]],
#     all_phrases: bool = False,
#     num_documents: int = 10,
#     is_regexp: bool = False,
#     return_all_hits: bool = False,
#     sort_field: str = "date",
#     last_id: Optional[any] = None,  # Adding last_id parameter
#     subset_filter: bool = True,
#     es: Optional[Elasticsearch] = None,
# ) -> Generator[Dict, None, None]:
#     """
#     :param index: Name of the index
#     :param phrases: A single string or a list of strings to be matched in the `text` field
#         of the index.
#     :param all_phrases: Whether the document should contain all phrases (AND clause) or any
#         of the phrases (OR clause).
#     :param num_documents: The number of document hits to return.
#     :param is_regexp: Whether the phrases are regular expressions. Note that spaces in regular
#         expressions are not supported by ElasticSearch, so if you want to do an exact match for
#         spans longer than a single term, set this to False. In most cases, using exp1|exp2 is better
#         than specifying [exp1, exp2] as two different `phrases`.
#     :param return_all_hits: Whether to return all hits beyond maximum 10k results. This will return an
#         iterator.
#     :return: An iterable (of length `num_documents` if `return_all_hits` is False),
#         containing the relevant hits.

#     Examples:

#         get_documents_containing_phrases("test-index", "legal", num_documents=50)  # single term, get 50 documents
#         get_documents_containing_phrases("test-index", ["legal", "license"])  # list of terms
#         get_document_containing_phrases("test-index", ["terms of use", "legally binding"])  # list of word sequences

#         # The documents should contain both `winter` and `spring` in the text.
#         get_documents_containing_phrases("test-index", ["winter", "spring"], all_phrases=True)
#     """
#     es = es or es_init()

#     # import ipdb; ipdb.set_trace()
#     query = _query_documents_contain_phrases(phrases, all_phrases, is_regexp=is_regexp)
#     if index == "c4" and subset_filter:
#         if "filter" in query["bool"]:
#             query["bool"]["filter"].append({"term": {"subset": "en"}})
#         else:
#             query["bool"]["filter"] = [{"term": {"subset": "en"}}]

#     # Modify the query to start after the last_id if provided
#     if last_id is not None:
#         if "range" not in query:
#             query["range"] = {}
#         query["range"][sort_field] = {"gt": last_id}
    
#     # if return_all_hits:
#     #     sort = [{sort_field: "asc"}]
#     #     pit = es.open_point_in_time(index=index, keep_alive="1m")
#     #     results = es.search(index=index, query=query, size=num_documents, sort=sort)["hits"]["hits"]
#     #     yield from results
        
#     #     while len(results) > 0:
#     #         # todo: perhaps we need to refresh pit?
#     #         results = es.search(
#     #             index=index,
#     #             query=query,
#     #             size=num_documents,
#     #             sort=sort,
#     #             search_after=results[-1]["sort"],
#     #         )["hits"]["hits"]
#     #         yield from results
#     #     try:
#     #         es.close_point_in_time(id=pit["id"])
#     #     except NotFoundError:
#     #         pass

#     if return_all_hits:
#         sort = [{sort_field: {"order": "asc"}}]
#         # Open PIT
#         pit = es.open_point_in_time(index=index, keep_alive="1m")
#         body = {
#             "query": query,
#             "sort": sort,
#             "size": num_documents,
#             "pit": {"id": pit["id"], "keep_alive": "1m"}
#         }
#         # Adjust the initial search to use PIT
#         results = es.search(body=body)["hits"]["hits"]
#         yield from results
        
#         while len(results) > 0:
#             search_after_param = results[-1]["sort"]
#             # Continue searching with PIT and search_after
#             body = {
#                 "query": query,
#                 "sort": sort,
#                 "size": num_documents,
#                 "search_after": search_after_param,
#                 "pit": {"id": pit["id"], "keep_alive": "1m"}
#             }
#             results = es.search(body=body)["hits"]["hits"]
#             yield from results

#         # Close PIT outside the while loop
#         try:
#             es.close_point_in_time(id=pit["id"])
#         except NotFoundError:
#             pass

#     else:
#         yield from es.search(index=index, query=query, size=num_documents)["hits"]["hits"]


def count_documents_for_each_phrase(
    index: str,
    phrases: Union[str, Iterable[str], Iterable[List[str]]],
    batch_size: int = 500,
    timeout: str = "60s",
    all_phrases: bool = False,
    subset_filter: bool = True,
    es: Optional[Elasticsearch] = None,
):
    if isinstance(phrases, str):
        phrases = [phrases]

    if all_phrases:
        try:
            assert isinstance(phrases, Iterable)
            assert isinstance(phrases[0], List)
            assert isinstance(phrases[0][0], str)
        except AssertionError:
            raise AssertionError(
                "`all_phrases` is set to True, please provide lists of lists."
            )
    else:
        try:
            assert isinstance(phrases, Iterable)
            assert isinstance(phrases[0], str)
        except AssertionError:
            raise AssertionError(
                "`all_phrases` is set to False, please provide a list of strings."
            )
    es = es or es_init()
    # num_shards = len(es.cat.shards(index=index, format="json"))
    final_counts = []

    done = False
    generator = iter(phrases)
    while not done:
        queries = []
        for i, phrase in enumerate(generator):
            if not isinstance(phrase, List):
                phrase = [phrase]
            match_query = []
            for phr in phrase:
                match_query.append({"match_phrase": {"text": phr}})

            if index == "c4" and subset_filter:
                match_query.append({"term": {"subset": "en"}})

            queries.append({"index": index, "search_type": "query_then_fetch"})
            queries.append(
                {
                    "stored_fields": [],
                    "timeout": timeout,
                    "track_scores": False,
                    "track_total_hits": True,
                    "query": {"bool": {"filter": match_query}},
                }
            )
            if i == batch_size:
                break
        if len(queries) == 0:
            done = True
            break
        results = es.msearch(
            index=index,
            searches=queries,
            search_type="query_then_fetch",
            rest_total_hits_as_int=True,
        )
        final_counts += [r["hits"]["total"] for r in results["responses"]]
    return final_counts


def count_total_occurrences_of_unigrams(
    index: str,
    unigrams: Union[str, List[str]],
    subset_filter: bool = True,
    es: Optional[Elasticsearch] = None,
) -> Dict[str, int]:
    """
    :param index: Name of the index
    :param terms: A single unigram or a list of unigrams to be matched in the `text` field
        of the index.
    :return: The total number of occurrences of each unigram in `terms` across all documents.

    Examples:

        count_total_occurrences_of_unigrams("test-index", "legal")  # single term
        count_total_occurrences_of_unigrams("test-index", ["legal", "license"])  # list of terms

    """
    if isinstance(unigrams, str):
        unigrams = [unigrams]

    es = es or es_init()

    # We use individual shards for counting total occurrences, because elasticsearch's default behavior
    # is to return term statistics for a randomly selected shard. For more information on term vector behaviour, please
    # see the following:
    # https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-termvectors.html#docs-termvectors-api-behavior
    num_shards = len(es.cat.shards(index=index, format="json"))

    logger.debug(f"Total number of shards in '{index}': {num_shards}")

    term_freq_dict = {}
    for term in unigrams:
        query = {"bool": {"filter": {"match": {"text": term}}}}
        if index == "c4" and subset_filter:
            query = {
                "bool": {
                    "filter": [{"match": {"text": term}}, {"term": {"subset": "en"}}]
                }
            }
        else:
            query = {"bool": {"filter": {"match": {"text": term}}}}

        total_freq = 0
        for i in range(num_shards):
            documents = es.search(
                index=index,
                query=query,
                preference=f"_shards:{i}",
                stored_fields=[],
                track_total_hits=False,
            )

            if len(documents["hits"]["hits"]) > 0:
                doc_id = documents["hits"]["hits"][0]["_id"]

                term_vector = es.termvectors(
                    index=index,
                    id=doc_id,
                    fields=["text"],
                    positions=False,
                    term_statistics=True,
                    preference=f"_shards:{i}",
                )

                ttf = term_vector["term_vectors"]["text"]["terms"][term]["ttf"]
                logger.debug(f"Total term frequency for shard {i}: {ttf}")
                total_freq += ttf

        logger.info(
            f"The term: '{term}' occurs {total_freq} times across all documents in '{index}'."
        )
        term_freq_dict[term] = total_freq
    return term_freq_dict


def get_documents_containing_phrases(
    index: str,
    phrases: Union[str, List[str]],
    all_phrases: bool = False,
    num_documents: int = 10,
    is_regexp: bool = False,
    return_all_hits: bool = False,
    sort_field: str = "date",
    subset_filter: bool = True,
    es: Optional[Elasticsearch] = None,
    scroll_size: int = 10000,
    scroll_timeout: str = "1m",
) -> Generator[Dict, None, None]:
    es = es or es_init()
    query = _query_documents_contain_phrases(phrases, all_phrases, is_regexp=is_regexp)

    if index == "c4" and subset_filter:
        if "filter" in query["bool"]:
            query["bool"]["filter"].append({"term": {"subset": "en"}})
        else:
            query["bool"]["filter"] = [{"term": {"subset": "en"}}]

    if return_all_hits:
        sort = [{sort_field: "asc"}]
        pit = es.open_point_in_time(index=index, keep_alive=scroll_timeout)
        response = es.search(index=index, query=query, size=scroll_size, sort=sort, scroll=scroll_timeout,)
        hits = response["hits"]["hits"]
        yield from hits

        scroll_id = response["_scroll_id"]  # Get _scroll_id from the top-level response
        while len(hits) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
            hits = response["hits"]["hits"]
            yield from hits
            scroll_id = response["_scroll_id"]  # Update scroll_id for the next batch
        try:
            es.close_point_in_time(id=pit["id"])
        except NotFoundError:
            pass
    else:
        yield from es.search(index=index, query=query, size=num_documents)["hits"]["hits"]


# def get_documents_containing_phrases(
#     index: str,
#     phrases: Union[str, List[str]],
#     all_phrases: bool = False,
#     num_documents: int = 10,
#     is_regexp: bool = False,
#     return_all_hits: bool = False,
#     sort_field: str = "date",
#     subset_filter: bool = True,
#     es: Optional[Elasticsearch] = None,
# ) -> Generator[Dict, None, None]:
#     """
#     :param index: Name of the index
#     :param phrases: A single string or a list of strings to be matched in the `text` field
#         of the index.
#     :param all_phrases: Whether the document should contain all phrases (AND clause) or any
#         of the phrases (OR clause).
#     :param num_documents: The number of document hits to return.
#     :param is_regexp: Whether the phrases are regular expressions. Note that spaces in regular
#         expressions are not supported by ElasticSearch, so if you want to do an exact match for
#         spans longer than a single term, set this to False. In most cases, using exp1|exp2 is better
#         than specifying [exp1, exp2] as two different `phrases`.
#     :param return_all_hits: Whether to return all hits beyond maximum 10k results. This will return an
#         iterator.
#     :return: An iterable (of length `num_documents` if `return_all_hits` is False),
#         containing the relevant hits.

#     Examples:

#         get_documents_containing_phrases("test-index", "legal", num_documents=50)  # single term, get 50 documents
#         get_documents_containing_phrases("test-index", ["legal", "license"])  # list of terms
#         get_document_containing_phrases("test-index", ["terms of use", "legally binding"])  # list of word sequences

#         # The documents should contain both `winter` and `spring` in the text.
#         get_documents_containing_phrases("test-index", ["winter", "spring"], all_phrases=True)
#     """
#     es = es or es_init()

#     query = _query_documents_contain_phrases(phrases, all_phrases, is_regexp=is_regexp)
#     if index == "c4" and subset_filter:
#         if "filter" in query["bool"]:
#             query["bool"]["filter"].append({"term": {"subset": "en"}})
#         else:
#             query["bool"]["filter"] = [{"term": {"subset": "en"}}]
    
#     if return_all_hits:
#         sort = [{sort_field: "asc"}]
#         pit = es.open_point_in_time(index=index, keep_alive="1m")
#         results = es.search(index=index, query=query, size=num_documents, sort=sort)["hits"]["hits"]
#         yield from results
        
#         while len(results) > 0:
#             # todo: perhaps we need to refresh pit?
#             results = es.search(
#                 index=index,
#                 query=query,
#                 size=num_documents,
#                 sort=sort,
#                 search_after=results[-1]["sort"],
#             )["hits"]["hits"]
#             yield from results
#         try:
#             es.close_point_in_time(id=pit["id"])
#         except NotFoundError:
#             pass
#     else:
#         yield from es.search(index=index, query=query, size=num_documents)["hits"]["hits"]


def download_image(url, output_folder, filename):
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    # Get the file name from the URL (this assumes URLs end with filename)
    filename = os.path.join(output_folder, filename)
    
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192*4):
            file.write(chunk)

    return filename


def handle_image_download(only_download_captions_for_already_downloaded_images, documents, output_folder, seq, url):
    filename = f'image_seq_{seq}_' #+ os.path.basename(url)
    try:
        if only_download_captions_for_already_downloaded_images:
            filepath = os.path.join(output_folder, filename)
            if os.path.exists(filepath):
                # print(f"Image {url} already downloaded. only downloading caption")
                caption_this_image = documents[seq]["_source"]["text"]
                # image_captions.append((filepath, caption_this_image))
                return filepath, caption_this_image
        else:
            filepath = download_image(url, output_folder=output_folder, filename=filename)
            # print(f"Downloaded {url} to {filepath}")
            ## if download is successful, get the caption and save it in the csv file
            caption_this_image = documents[seq]["_source"]["text"]
            # image_captions.append((filepath, caption_this_image))
            return filepath, caption_this_image
        
    except Exception as e:
        # print(f"Error downloading {url}. Error: {e}")
        pass
    return None, None


def phrase_download_image(phrase, max_num_documents=10, output_folder_general=None, only_download_captions_for_already_downloaded_images=False, parallelize_download_across_images_of_one_entity=False, return_all_hits=None, max_cores=40):
    output_folder_this_celeb = f"{output_folder_general}/{phrase}"
    os.makedirs(output_folder_this_celeb, exist_ok=True)
    ## if there are already 50 documents inside the folder, then skip downloading images
    if len(os.listdir(output_folder_this_celeb)) >= 25:
        print(f"Already downloaded {len(os.listdir(output_folder_this_celeb))} images for {phrase}. Skipping download")
        return
    
    es = es_init()
    num_documents = count_documents_containing_phrases("re_laion2b-en-*", phrase, es=es)
    total_desired_documents = min(num_documents, max_num_documents)
    print(f"total desired documents for {phrase} is:", total_desired_documents)
    index = "re_laion2b-en-*"
    if return_all_hits is None:
        return_all_hits = num_documents > 10000
    else:
        if num_documents > 10000 and not return_all_hits:
            total_desired_documents = 10000
        else: raise NotImplementedError
        
    print("return_all_hits:", return_all_hits)
    documents_batch = get_documents_containing_phrases(index,phrase,num_documents=total_desired_documents,es=es,return_all_hits=return_all_hits)
    documents = list(documents_batch)
    ## randomly shuffle the documents and take the first 200K documents
    random.shuffle(documents)
    documents = documents[:max_images_to_download]
    print(f"Downloading images for {phrase} with {len(documents)} images")

    urls = [doc["_source"]["url"] for doc in documents]
    ## along with the downloading images, get the captions for the images. And save them in a csv file with the image and caption in the output folder

    image_captions = []
    image_caption_file = f'{output_folder_this_celeb}/image_captions.csv'

    if not parallelize_download_across_images_of_one_entity:
        for seq, url in enumerate(urls):
            filepath, caption_this_image = handle_image_download(only_download_captions_for_already_downloaded_images, documents, output_folder_this_celeb, seq, url)
            if filepath and caption_this_image:
                image_captions.append((filepath, caption_this_image))
    else:
        max_parallel_processes = max_cores
        from multiprocessing import Pool

        args_for_download = [(only_download_captions_for_already_downloaded_images, documents, output_folder_this_celeb, seq, url) for seq, url in enumerate(urls)]

        # Create a pool of worker processes
        with Pool(processes=max_parallel_processes) as pool:
            # Use starmap instead of map. starmap allows the function to receive multiple arguments from the iterable
            results = pool.starmap(handle_image_download, args_for_download)

        # Filter out failed downloads (where result is None or both elements are None) and extend the image_captions list
        image_captions.extend([(filepath, caption) for filepath, caption in results if filepath and caption])

    ## save the captions in the csv file. And wrap the caption in double quotes. Separate character is !#!# to separate the image and caption in the csv file
    with open(image_caption_file, 'w') as f:
        f.write("image|caption\n")
        for image_caption in image_captions:
            # sep_char = "!#!#"
            # wrap the caption in double quotes
            caption = image_caption[1]
            caption = caption.replace('"', "'")
            caption = f'"{caption}"'
            image_name = f'"{image_caption[0]}"'
            line_to_write = image_name + "|" + caption + "\n"
            f.write(line_to_write)
    
    print(f"Downloaded {len(image_captions)} images for {phrase} and saved the captions in {image_caption_file}")


def count_files_and_directories(root_dir):
    total_directories = 0
    directories_less_than_1k = 0

    set_ents = set()
    for subdir, dirs, files in os.walk(root_dir):
        total_directories += 1
        file_count = len(files)
        # print(subdir, file_count)

        if file_count > 200:
        # if file_count < 1000:
            directories_less_than_1k += 1
        set_ents.add(subdir)
    # print(len(set_ents))
    return total_directories, directories_less_than_1k


def worker(celebrity_here):
    phrase_download_image(phrase=celebrity_here, max_num_documents=1000)


def normalize(s):
    # normalize to decomposed form, then remove nonspacing marks
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # remove all types of punctuation and convert to lower case
    s = s.translate(str.maketrans('', '', string.punctuation + string.whitespace))
    # convert to lower case for uniformity
    return s.lower()


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_images", action="store_true")
    parser.add_argument("--parallelize_download_across_entities", action="store_true")
    parser.add_argument("--parallelize_download_across_images_of_one_entity", action="store_true")
    parser.add_argument("--print_count", action="store_true")
    parser.add_argument("--set_of_people", type=str, default=None, choices=['celebrity', 'politician', 'birds', "wikiart_artists", "caption_assumption", "artists_group1", "artists_group2"], required=True)
    parser.add_argument("--also_consider_aliases", action="store_true")
    parser.add_argument("--dataset", type=str, default="laion2ben", choices=['laion2ben', 'laion5b'])
    parser.add_argument("--dataindex", type=str, default="re_laion2b-en-*", choices=['re_laion2b-en-*', 're_laion2b*', '*laion*'])
    args = parser.parse_args()
    
    if args.dataset == "laion2ben":
        database = "re_laion2b-en-*"
    elif args.dataset == "laion5b":
        raise NotImplementedError("laion5b is not implemented yet")
    else:   raise ValueError("Invalid dataset")
    
    assert sum([args.parallelize_download_across_entities, args.parallelize_download_across_images_of_one_entity]) <= 1, "Only one of the parallelization flags can be set"
    if sum([args.parallelize_download_across_entities, args.parallelize_download_across_images_of_one_entity]) == 1:    assert args.download_images, "Parallelization can only be done when downloading images"
    assert args.print_count or args.download_images, "Either print the count or download images"

    es = es_init()

    # people_with_few_images = ['Coco Jones', 'Shameik Moore', 'Jayme Lawson', 'Letitia Wright', 'Ariana DeBose', 'Hailey Kilgore', 'Jaden Michael', 'Quincy Isaiah', 'Marsai Martin', 'Miles Gutierrez-Riley', 'Caleb McLaughlin', 'Saniyya Sidney', 'Tyrel Jackson Williams', 'Kaci Walfall', 'Myles Truitt', 'Priah Ferguson', 'Michael Evans Behling', 'Elisha Williams', 'Elisha EJ Williams', 'Laura Kariuki', 'Olly Sholotan', 'Akira Akbar', 'Ava Grey', 'Tati Gabrielle', 'Savannah Lee Smith', 'Grace Duah', 'Jalyn Hall', 'Lyric Ross', 'Lonnie Chavis', 'Algee Smith', 'Keith Powers', 'Jeremy Pope', 'Kelvin Harrison Jr.', 'Jharrel Jerome', 'Micheal Ward', 'Myles Frost', 'Asante Blackk']
    # people_with_few_images = ['Chosen Jacobs', 'Jaden Smith', 'Khylin Rhambo', 'Trevor Jackson', 'Jacob Latimore', "Bobb'e J. Thompson", 'Danny Boyd Jr.', 'Isaiah R. Hill', 'Kwesi Boakye', 'Skylan Brooks', 'Ariana Neal']
    # people_with_few_images = ['Shanaya Kapoor', 'Tara Sutaria', 'Banita Sindhu', 'Sanjana Sanghi', 'Pranutan Bahl', 'Shirley Setia', 'Alaia F.', 'Palak Tiwari', 'Isabelle Kaif', 'Mahikaa Rampal', "Krystle D'Souza", 'Manushi Chillar', 'Paloma Dhillon', 'Shehnaaz Gill', 'Alizeh Agnihotri', 'Saiee Manjrekar', 'Sobhita Dhulipala', 'Shivani Raghuvanshi', 'Tripti Dimri', 'Shivaleeka Oberoi', 'Radhika Madan', 'Geetika Vidya Ohlyan', 'Saloni Batra', 'Banita Sandhu', 'Zaira Wasim', 'Dhvani Bhanushali', 'Sharvari Wagh', 'Kavya Thapar', 'Arjumman Mughal', 'Adah Sharma', 'Mawra Hocane'] 
    # people_with_few_images = ['Ahaan Panday', 'Ahan Shetty', 'Ishaan Khatter', 'Siddhant Chaturvedi', 'Abhimanyu Dassani', 'Rohit Saraf', 'Vihaan Samat', 'Shantanu Maheshwari', 'Adarsh Gourav', 'Babil Khan', 'Ishwak Singh', 'Gurfateh Pirzada', 'Pavail Gulati', 'Laksh Lalwani', 'Jibraan Khan', ]    
    male_white_celebs = ['Gabriel LaBelle', 'Dominic Sessa', 'Corey Mylchreest', 'Sam Nivola', 'Tom Blyth', 'Jordan Firstman', 'Josh Seiter', 'Nicola Porcella', 'Armen Nahapetian', 'Joey Klaasen']
    male_black_celebs = ['Jaylin Webb', 'Quincy Isaiah', 'Miles Gutierrez-Riley', 'Jalyn Hall', 'Myles Frost', 'Wisdom Kaye', 'Olly Sholotan', 'Isaiah R. Hill', "Bobb'e J. Thompson", 'Myles Truitt']
    male_brown_celebs = ['Sajith Rajapaksa', 'Aryan Simhadri', 'Aditya Kusupati', 'Vihaan Samat', 'Ishwak Singh', 'Gurfateh Pirzada', 'Pavail Gulati', 'Cwaayal Singh', 'Jibraan Khan', 'Vedang Raina']
    female_white_celebs = ['Gabby Windey', 'Mia Challiner', 'Isabel Gravitt', 'Pardis Saremi', 'Elle Graham', 'Cara Jade Myers', 'Ali Skovbye', 'Hannah Margaret Selleck', 'Bridgette Doremus', 'Milly Alcock']
    female_black_celebs = ['Kudakwashe Rutendo', 'Ayo Edebiri', 'Kaci Walfall', 'Elisha Williams', 'Laura Kariuki', 'Akira Akbar', 'Savannah Lee Smith', 'Samara Joy', 'Arsema Thomas', 'Leah Jeffries']
    female_brown_celebs = ['Priya Kansara', 'Pashmina Roshan', 'Banita Sindhu', 'Alaia Furniturewala', 'Paloma Dhillon', 'Alizeh Agnihotri', 'Geetika Vidya Ohlyan', 'Saloni Batra', 'Sharvari Wagh', 'Arjumman Mughal']
    
    # group_order = ['male_white_celebs', 'female_white_celebs', 'male_brown_celebs', 'female_brown_celebs', 'male_black_celebs', 'female_black_celebs']
    # entities = []
    # for group in group_order:
    #     for celeb_here in eval(group):
    #         entities.append(celeb_here)
   
    return_all_hits = None
   
    if args.set_of_people == 'celebrity':
        entities = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/celebrity_counts_with_aliases.csv", sep="|")['celeb'].tolist()
        # entities = ['Kool Herc', 'Kool DJ Herc', 'Clive Campbell', 'M. Walker', 'Yamiche', 'Koker', 'Kirschner', 'Mary Belle', 'Varona', 'Bari Weiss', 'Keeshan', 'Ramakrishnan', 'Maitreyi', 'Bregje', 'Heinen', 'Stoller', 'Winkler', 'Brinkley', 'Ferreira', 'Angie', 'Jackson', 'Shenae Grimes', 'Skarsgård']
        # entities = ['Walker', 'Summer', 'Delphine', 'Belle', 'Barry', 'Babygirl', 'Marjani', 'David', 'Weiss', 'Irwin']
        # entities = ['Deb Haaland', 'Lil Nas X', 'amichai chikli']
        # entities = ['Phoebe Bridgers', 'Phoebe Dynevor', 'Sofia Hellqvist', 'Scott Speedman', 'Jacob Sartorius', 'Julia Stegner', 'Kim Zolciak-Biermann', 'Diego Boneta', 'Devon Sawa', 'John Wayne Gacy', 'Meg Ryan', 'Danny Thomas', 'Ashlee Simpson', 'Rose Leslie', 'Kurt Russell', 'Lukita Maxwell', 'Lori Harvey', 'Michael Buffer', 'Lindsay Price', 'Lacey Evans']
        # entities = ['Bridgers', 'Dynevor', 'Hellqvist', 'Speedman', 'Sartorius', 'Stegner', 'Biermann', 'Boneta', 'Sawa', 'Gacy', 'Ryan', 'Thomas', 'Simpson', 'Leslie', 'Russell', 'Maxwell', 'Harvey', 'Buffer', 'Price', 'Evans'][::-1]
        # entities = ['Danneel Harris', 'Normani Kordei', 'Penélope Cruz', 'Avani Gregg', 'Cacee Cobb', 'Simu Liu', 'Michiel Huisman', 'Léa Seydoux', 'Skai Jackson', 'Selita Ebanks', 'Forest Whitaker', "Olivia O'Brien", 'Samuel L. Jackson', 'Cody Lightning', 'David Dobrik', 'Joe Rogan', 'Nick Jonas', 'Nicole Fosse', 'Steve Lacy', 'John Corbett']
        entities = ['Penélope Cruz', 'Penelope Cruz'] #['Danneel', 'Normani', 'Penélope', 'Avani', 'Cacee', 'Simu', 'Michiel', 'Léa', 'Skai', 'Selita', 'Forest', "Olivia", 'Samuel', 'Cody', 'David', 'Joe', 'Nick', 'Nicole', 'Steve', 'John']
        # assert len(entities) == 399
        if args.also_consider_aliases:
            if os.path.exists("celebrity_data/celebrity_counts_with_aliases.csv"):
                alias_df = pd.read_csv("celebrity_data/celebrity_counts_with_aliases.csv", sep="|")     ##celeb|full_name_count|aliases_and_counts|alias_total_count|Outlier
                assert alias_df.shape == (len(entities), 5)
                all_aliases = alias_df['aliases_and_counts'].tolist()       ## this is a column where everything is a dictionary (as a string)
                all_aliases = [eval(alias) for alias in all_aliases]        ## convert the string to a dictionary
                assert [type(alias) for alias in all_aliases] == [dict]*len(all_aliases)
                ### now take the keys of all the dictionaries and make a set of all the aliases
                # all_aliases_names = set([alias for alias_dict in all_aliases for alias in alias_dict.keys()])
                if args.download_images:
                    all_aliases_names = []
                    if args.parallelize_download_across_entities:
                        ## here get the names of the aliases where the count is less than 10K, get the aliases from the dictionary all_aliases
                        for alias_dict in all_aliases:
                            for alias, count in alias_dict.items():
                                if count < 10000:
                                    all_aliases_names.append(alias)
                        all_aliases_names = list(set(all_aliases_names))
                    elif args.parallelize_download_across_images_of_one_entity:
                        selected_alias_and_counts = {}
                        for alias_dict in all_aliases:
                            for alias, count in alias_dict.items():
                                if count >= 10000:
                                    all_aliases_names.append(alias)
                                    selected_alias_and_counts[alias] = count
                        ## print the selected aliases and counts, sorted in ascending order of counts
                        selected_alias_and_counts = dict(sorted(selected_alias_and_counts.items(), key=lambda item: item[1]))
                        print(selected_alias_and_counts, len(selected_alias_and_counts), len(all_aliases_names), "\n\n")
                        all_aliases_names = list(selected_alias_and_counts.keys())
                        print(all_aliases_names)
                        
            elif os.path.exists("celebrity_data/celebrity_aliases.csv"):
                alias_df = pd.read_csv("celebrity_data/celebrity_aliases.csv", sep="|")
                assert alias_df.shape == (len(entities), 2)

            else:
                alias_file = "name_aliases_celebrity.pkl"
                import pickle
                with open(alias_file, 'rb') as f:
                    name_aliases = pickle.load(f)
                assert len(name_aliases) == len(entities)
                ## get all the aliases of everyone as a dict where the key is the main name and the value is the list of aliases. Note that aliases can also be empty
                all_aliases = {}
                for i, entity in enumerate(entities):
                    if isinstance(name_aliases[entity], list):
                        assert len(name_aliases[entity]) == 0
                        all_aliases[entity] = []
                    elif isinstance(name_aliases[entity], dict):
                        ## there can be multiple keys in the dict and we will merge the list of aliases for all the keys
                        all_aliases[entity] = []
                        for key, value in name_aliases[entity].items():
                            all_aliases[entity].extend(value)
                        ## remove duplicates
                        all_aliases[entity] = list(set(all_aliases[entity]))
                    else:
                        raise ValueError(f"Invalid type for name_aliases[{entity}]: {name_aliases[entity]}")
                
                ## print the output to a file where the first column is the main name and the second column is the list of aliases
                # with open("celebrity_data/celebrity_aliases.csv", 'w') as f:
                #     for entity, aliases in all_aliases.items():
                #         f.write(f"{entity}|{aliases}\n")
                # print("Saved the aliases in celebrity_aliases.csv")
                print("rerun now")
                exit()
            ## get the list of all the aliases
        
        output_folder = "all_downloaded_images"
        # if args.also_consider_aliases:
        output_folder = "/gscratch/scrubbed/vsahil/" + output_folder
    
    elif args.set_of_people == 'politician':
        politicians_previous = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/sampled_politicians_sorted.csv")  ## Name,counts_in_laion2b-en,count_range
        entities = politicians_previous['Name'].tolist()
        politicians_new = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/celebrity_data/good_politicians_to_analyze.csv")   # Name,counts_in_laion2b-en
        
        ## find the politcians in the new list which are not in the old list
        new_politicians = politicians_new['Name'].tolist()
        old_politicians = politicians_previous['Name'].tolist()
        # new_politicians = [politician for politician in new_politicians if politician not in old_politicians]
        print(len(new_politicians))
        
        output_folder = "all_downloaded_images_politicians"
        output_folder = "/gscratch/scrubbed/vsahil/" + output_folder
        if args.download_images:
            ## when we are parallelizing across entities, only consider entities whose caption counts are less than 10K, for entities with counts more than 10K we will use parallelization across images of one entity
            if args.parallelize_download_across_entities:
                entities = politicians_new[(politicians_new['counts_in_laion2b-en'] < 5100) & (politicians_new['counts_in_laion2b-en'] > 50)]['Name'].tolist()
                # entities = politicians_new[politicians_new['counts_in_laion2b-en'] < 5100]['Name'].tolist()
                ## get the intersection of the entities and the new politicians
                entities = list(set(entities).intersection(set(new_politicians)))
            elif args.parallelize_download_across_images_of_one_entity:
                entities = politicians_new[politicians_new['counts_in_laion2b-en'] >= 5100]['Name'].tolist()
                ## get the intersection of the entities and the new politicians
                entities = list(set(entities).intersection(set(new_politicians)))
            else:   raise NotImplementedError
            
    elif args.set_of_people == 'birds':
        birds = pd.read_csv("birds_dataset/all_birds.csv")['birds'].tolist()
        print(len(birds))
        entities = birds
        output_folder = "birds_dataset/all_birds_images"
    
    elif args.set_of_people == "wikiart_artists" or args.set_of_people == "artists_group1" or args.set_of_people == "artists_group2":
        if args.set_of_people == "wikiart_artists":
            artists_df = pd.read_csv("art_styles/style_similarity_somepalli/artists_to_analyze.csv")
        elif args.set_of_people == "artists_group1":
            artists_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group1.csv")
        elif args.set_of_people == "artists_group2":
            artists_df = pd.read_csv("/gscratch/h2lab/vsahil/vlm-efficiency/art_styles/style_similarity_somepalli/final_artists_group2.csv")
            
        artists_df = artists_df[['artist_name','counts_in_laion2b-en']]
        assert artists_df.shape[1] == 2
        assert len(artists_df) > 0 and len(artists_df) < 420, "Invalid number of artists"
        # entities = ["Nicolas Toussaint Charlet", "Nicolas-Toussaint Charlet", "Nicholas-Toussaint Charlet", "charlet nicolas toussaint", "charlet n. t.", "Charlet", "Nicolas T. Charlet", "charlet nicolas-toussaint", "n. t. charlet", "Nicolas Charlet", "Nicolas Toussaint"]
        entities = ["Wilhelm Von Kaulbach", "Von Kaulbach", "Wilhelm Eliodorus", "Wilhelm Kaulbach", "Bernhard Wilhelm Eliodorus Kaulbach", "Bernhard Wilhelm Eliodorus","Wilhelm Von Kaulbach","Wilhelm Kaulbach","prof. wilhelm von kaulbach","wilhelm v. kaulbach","Wilh. v. Kaulbach","wilhelm kaulbach","w. v. kaulbach","w. kaulbach","professor wilhelm von kaulbach","w. von kaulbach","wilh. von kaulbach","wm. kaulbach"]
        
        if args.download_images:
            if args.parallelize_download_across_entities:
                entities = artists_df[artists_df['counts_in_laion2b-en'] < 5100]['artist_name'].tolist()
            elif args.parallelize_download_across_images_of_one_entity:
                entities = artists_df[artists_df['counts_in_laion2b-en'] >= 5100]['artist_name'].tolist()
            else:   raise NotImplementedError

        # entities = ['piero dorazio', 'dorazio', 'david batchelor', 'batchelor', 'oswaldo guayasamin', 'guayasamin'] #, 'alexander liberman', 'liberman', 'jacob isaakszoon van ruisdael', 'jacob isaakszoon', 'isaakszoon van', 'Jacob van Ruisdael', 'louise elisabeth vigee le brun', 'elisabeth vigee le brun', 'vigee le brun', 'Louise Élisabeth Vigée Le Brun', 'Madame Le Brun', 'martiros saryan']
        output_folder = "art_styles/style_similarity_somepalli/all_artists_images"
        
    elif args.set_of_people == "caption_assumption":
        entities = ["and", "the", "a", "an", "in", "for", "on", "at", "by", "to", "of", "it", "as"]
        output_folder = "all_downloaded_images_for_captions"
        output_folder = "/gscratch/scrubbed/vsahil/" + output_folder
        return_all_hits = False
        
    else:
        raise ValueError("Invalid set of people")
    
    if args.download_images:
        if not args.also_consider_aliases:
            print(len(entities), output_folder, return_all_hits)
        else:
            print(len(entities), output_folder, len(all_aliases_names))
        
    if args.print_count:
        total_images_to_download_for_aliases = 0
        last_name_junior = False
        
        for entity in entities:
            # count_this_friend_full_name = count_documents_containing_phrases("re_laion2b-en-*", entity, es=es)
            count_this_friend_full_name = count_documents_containing_phrases(args.dataindex, entity, es=es)
            
            if args.also_consider_aliases:
                ## get the aliases of the entity
                if os.path.exists("celebrity_data/celebrity_aliases.csv"):
                    alias_df = pd.read_csv("celebrity_data/celebrity_aliases.csv", sep="|")
                    aliases = alias_df[alias_df['celeb'] == entity]['aliases'].values[0]
                    assert aliases[0] == "[" and aliases[-1] == "]", f"Invalid format for aliases: {aliases} for {entity}"
                    ## aliases will either be an empty list or a list of strings, obviously after converting it to the correct type
                    aliases = eval(aliases)
                    ## we should remove any alias where the celeb is a substring of the alias because the count of the celeb will already include the count of the alias
                    aliases = [alias for alias in aliases if normalize(entity) not in normalize(alias)]
                    count_of_aliases = {}
                    num_images_to_download = 0
                    for alias in aliases:
                        count_this_friend_alias = count_documents_containing_phrases("re_laion2b-en-*", alias, es=es)
                        count_of_aliases[alias] = count_this_friend_alias
                        num_images_to_download += min(count_this_friend_alias, 100000)
                    total_images_to_download_for_aliases += num_images_to_download
                    print(f"Full name: {entity}: {count_this_friend_full_name}")
                    print(f"Aliases: {count_of_aliases}")
                    outlier = 'FALSE'
                    if sum(count_of_aliases.values()) > count_this_friend_full_name:
                        outlier = 'TRUE'
                    ## save all of this in a file. First column with the main celeb name and the second colum will be the main name count. The third column will be the dict with the alias and their counts
                    if not os.path.exists("celebrity_data/celebrity_counts_with_aliases.csv"):
                        with open("celebrity_data/celebrity_counts_with_aliases.csv", 'w') as f:
                            f.write("celeb|full_name_count|aliases_and_counts|alias_total_count|Outlier\n")
                    
                    # with open("celebrity_data/celebrity_counts_with_aliases.csv", 'a') as f:
                    #     f.write(f"{entity}|{count_this_friend_full_name}|{count_of_aliases}|{sum(count_of_aliases.values())}|{outlier}\n")
            else:
                print(f"Full name: '{entity}': {count_this_friend_full_name}")
                if args.set_of_people == "celebrity":
                    ## print the full name count, first name count, last name count in a df and save it in a csv file
                    ## if there is hyphen in name, repalce it with space and then split the name
                    new_entity = entity.replace("-", " ")
                    first_name = new_entity.split()[0]
                    last_name = new_entity.split()[-1]
                    if last_name == "Jr.":
                        last_name = new_entity.split()[-2]
                        last_name_junior = True
                    first_name_count = count_documents_containing_phrases("re_laion2b-en-*", first_name, es=es)
                    last_name_count = count_documents_containing_phrases("re_laion2b-en-*", last_name, es=es)
                    ## if there are middle names, get the count of the highest count middle name
                    if len(new_entity.replace("Jr.", "").split()) > 2:
                        middle_names = new_entity.split()[1:-1]
                        middle_name_counts = [count_documents_containing_phrases("re_laion2b-en-*", middle_name, es=es) for middle_name in middle_names]
                        middle_name_counts = sorted(middle_name_counts, reverse=True)
                        middle_name_count = middle_name_counts[0]
                    else:
                        middle_name_count = 0

                    # with open("celebrity_data/celebrity_counts_all_names.csv", 'a') as f:
                    #     ## add header if the file is empty
                    #     if os.stat("celebrity_data/celebrity_counts_all_names.csv").st_size == 0:
                    #         f.write("celeb|full_name_count|first_name|first_name_count|last_name|last_name_count|middle_name_count\n")
                    #     f.write(f"{entity}|{count_this_friend_full_name}|{first_name}|{first_name_count}|{last_name}|{last_name_count}|{middle_name_count}\n")
            
            # if args.set_of_people == "politician":
            #     ## assert that the count is same here and what is in the file
            #     count_in_file = politicians[politicians['Name'] == entity]['counts_in_laion2b-en'].values[0]
            #     assert count_this_friend_full_name == count_in_file, f"Count mismatch for {entity}: {count_this_friend_full_name} != {count_in_file}"

        print(f"Total images to download for aliases: {total_images_to_download_for_aliases}")

    elif args.download_images:
        max_images_to_download = 100000
        max_cores = 80
        os.makedirs(output_folder, exist_ok=True)
        if args.also_consider_aliases:
           entities = all_aliases_names
        if len(entities) == 1:
            args.parallelize_download_across_entities = False
        
        if args.parallelize_download_across_entities:
            assert not args.parallelize_download_across_images_of_one_entity, "Parallelization across entities and images of one entity cannot be done together"
            from multiprocessing import Pool
            pool = Pool(processes=min(max_cores, len(entities)))
            for celebrity in entities:
                pool.apply_async(phrase_download_image, args=(celebrity, max_images_to_download, output_folder, False, args.parallelize_download_across_images_of_one_entity, return_all_hits))
            pool.close()
            pool.join()
        
        else:
            for celebrity in entities[::-1][30:]:
                print(celebrity)
                phrase_download_image(phrase=celebrity, max_num_documents=max_images_to_download, output_folder_general=output_folder, only_download_captions_for_already_downloaded_images=False, parallelize_download_across_images_of_one_entity=args.parallelize_download_across_images_of_one_entity, return_all_hits=return_all_hits, max_cores=max_cores)

