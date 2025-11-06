#!/usr/bin/env python3

import argparse
from functools import cache
import json
from operator import invert, le
import string
import pickle
from typing import TypedDict
from nltk.stem import PorterStemmer
from collections import defaultdict
from pathlib import Path

class InvertedIndex:
    index: defaultdict[str, set[int]] = defaultdict(set)
    docmap: defaultdict[int, Movie] = defaultdict()

    def __add_document(self: InvertedIndex, doc_id: int, term: str, stop_words: list[str]) -> None:
        tokenized = process_text(term, stop_words)
        for token in tokenized:
            self.index[token].add(doc_id)

    def get_document(self: InvertedIndex, term: str) -> list[int]:
        doc_ids = self.index[term.lower()]

        return sorted(doc_ids) 
    
    def build(self: InvertedIndex) -> None:
        with open('data/movies.json', 'r') as file:
            data = json.load(file)

        stop_words = read_stopwords()
        movies: list[Movie] = data['movies']

        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}", stop_words)
            self.docmap[movie['id']] = movie

    def save(self: InvertedIndex) -> None:
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        index_cache_file = f"{cache_dir}/index.pkl"
        with open(index_cache_file, 'wb') as index_file:
            pickle.dump(self.index, index_file)

        doc_cache_file = f"{cache_dir}/docmap.pkl"
        with open(doc_cache_file, 'wb') as doc_file:
            pickle.dump(self.docmap, doc_file)

    def load(self: InvertedIndex) -> None:
        index_path_path = Path("cache/index.pkl")

        if not index_path_path.exists():
            raise FileNotFoundError(f"{index_path_path} was not found")

        with open(index_path_path, 'rb') as f:
            self.index = pickle.load(f)

        doc_cache_file = Path("cache/docmap.pkl")
        if not doc_cache_file.exists():
            raise FileNotFoundError(f"{doc_cache_file} was not found")

        with open(doc_cache_file, 'rb') as f:
            self.docmap = pickle.load(f)


class Movie(TypedDict):
    id: int
    title: str
    description: str

def read_stopwords() -> list[str]:
    file = open("data/stopwords.txt")
    stop_words = file.read().split('\n')

    return stop_words


def tokenize(query: str) -> list[str]:
    tokenized = query.split(' ')

    return tokenized

def remove_stop_words(tokenized: list[str], stop_words: list[str]) -> list[str]:
    result: list[str] = []
    for curr_token in tokenized:
        if curr_token not in stop_words:
            result.append(curr_token)

    return result

def process_text(current: str, stop_words: list[str]) -> list[str]:
    # first make it lower case
    processed = current.lower()
    
    # create the translation table and remove punctuation
    translate_table = str.maketrans('', '', string.punctuation)
    processed = processed.translate(translate_table)

    # tokenize the string now
    tokenized = tokenize(processed)

    # remove the stop words from the tokenized list
    tokenized = remove_stop_words(tokenized, stop_words)

    # put each token to its stem
    stemmer = PorterStemmer()
    tokenized = [stemmer.stem(token) for token in tokenized]

    return tokenized

def matches(query_tokens: list[str], result_tokens: list[str]) -> bool:
    return any(query_token in result_token 
               for query_token in query_tokens
               for result_token in result_tokens)


def search(query: str, num_results: int) -> list[Movie]:
    with open('data/movies.json', 'r') as file:
        data = json.load(file)

    stop_words = read_stopwords()

    results: list[Movie] = []
    movies: list[Movie] = data['movies']
    
    for movie in movies:
        processed_query = process_text(query, stop_words)
        processed_title = process_text(movie['title'], stop_words)
        if matches(processed_query, processed_title):
            results.append(movie)
    
    results.sort(key=lambda movie: movie['id'])
    return results[:num_results]

def search_command(query: str, limit: int = 5) -> list[Movie]:
    index = InvertedIndex()
    index.load()
    query_tokens = process_text(query, read_stopwords())
    seen: set[int] = set()
    results: list[Movie] = []

    for token in query_tokens:
        matching_ids = index.get_document(token)
        for doc_id in matching_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            document = index.docmap[doc_id]
            results.append(document)
            if len(results) >= 5:
                return results
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    search_parser = subparsers.add_parser("search", help="Search movies BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build out the cache")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for result in results:
                print(f"{result['title']} - {result['id']}")
            # inverted_index = InvertedIndex()
            # try:
            #     inverted_index.load()
            #     query_tokens = process_text(args.query, read_stopwords())
            #     results: list[Movie] = []
            #     for query_token in query_tokens:
            #         found_docs = inverted_index.get_document(query_token)
            #         for doc_id in found_docs:
            #             results.append(inverted_index.docmap[doc_id])
            #             if len(results) >= 5:
            #                 break
            #
            #     for result in results:
            #         print(f"{result['title']}")
            # except FileNotFoundError as e:
            #     print(e)

            # search_results = search(args.query, 5)
            # for index, result in enumerate(search_results):
            #     print(f"{index + 1}: {result['title']}")
            pass
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            docs = sorted(inverted_index.index['merida'])
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

