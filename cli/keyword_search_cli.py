#!/usr/bin/env python3

import argparse
import json
from typing import TypedDict

class Movie(TypedDict):
    id: int
    title: str
    description: str


def search(query: str, num_results: int) -> list[Movie]:
    with open('data/movies.json', 'r') as file:
        data = json.load(file)

    results: list[Movie] = []
    movies: list[Movie] = data['movies']
    
    for movie in movies:
        if query.lower() in movie['title'].lower():
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

