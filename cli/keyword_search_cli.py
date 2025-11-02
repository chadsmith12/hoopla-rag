#!/usr/bin/env python3

import argparse
import json
import string
from typing import TypedDict

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
    return remove_stop_words(tokenized, stop_words)

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    search_parser = subparsers.add_parser("search", help="Search movies BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_results = search(args.query, 5)
            for index, result in enumerate(search_results):
                print(f"{index + 1}: {result['title']}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

