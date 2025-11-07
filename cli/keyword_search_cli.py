#!/usr/bin/env python3

import argparse
import json
import math
import string
import pickle
from typing import Counter, TypedDict
from nltk.stem import PorterStemmer
from collections import defaultdict
from pathlib import Path

class InvertedIndex:
    index: defaultdict[str, set[int]] = defaultdict(set)
    docmap: defaultdict[int, Movie] = defaultdict()
    term_frequencies: defaultdict[int, Counter[str]] = defaultdict(Counter)
    stop_words: list[str] = []

    def __add_document(self: InvertedIndex, doc_id: int, term: str, stop_words: list[str]) -> None:
        tokenized = process_text(term, stop_words)
        for token in tokenized:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokenized)

    def get_document(self: InvertedIndex, term: str) -> list[int]:
        doc_ids = self.index[term.lower()]

        return sorted(doc_ids)

    def get_tf(self: InvertedIndex, doc_id: int, term: str) -> int:
        tokenized = process_text(term, self.stop_words)
        if len(tokenized) > 1:
            raise ValueError("too many terms")

        return self.term_frequencies[doc_id][tokenized[0]]
    
    def get_idf(self: InvertedIndex, term: str) -> float:
        tokenized = process_text(term, self.stop_words)
        if len(tokenized) > 1:
            raise ValueError("too many terms")

        token = tokenized[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_if(self: InvertedIndex, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def build(self: InvertedIndex) -> None:
        with open('data/movies.json', 'r') as file:
            data = json.load(file)

        stop_words = read_stopwords()
        self.stop_words = stop_words
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

        term_cache_file = f"{cache_dir}/term_frequencies.pkl"
        with open(term_cache_file, 'wb') as term_file:
            pickle.dump(self.term_frequencies, term_file)

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

        term_cache_file = Path("cache/term_frequencies.pkl")
        if not term_cache_file.exists():
            raise FileNotFoundError(f"{term_cache_file} was not found")

        with open(term_cache_file, 'rb') as f:
            self.term_frequencies = pickle.load(f)


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

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a term in a document id")
    tf_parser.add_argument("doc_id", type=int, help="Id of document")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequence for a term")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the tf-idf for a term in a document id")
    tfidf_parser.add_argument("doc_id", type=int, help="Id of document")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for result in results:
                print(f"{result['title']} - {result['id']}")
            pass
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            docs = sorted(inverted_index.index['merida'])
            print(f"First document for token 'merida' = {docs[0]}")
            pass
        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            print(f"{inverted_index.get_tf(args.doc_id, args.term)}")
            pass
        case "idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequence of '{args.term}': {idf:.2f}")
            pass
        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf_idf = inverted_index.get_tf_if(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

