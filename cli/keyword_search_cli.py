#!/usr/bin/env python3

import argparse
import json

def search():
    with open('data/movies.json', 'r') as file:
        data = json.load(file)

    movies = data['movies']

    print(movies)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    search_parser = subparsers.add_parser("search", help="Search movies BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search()
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

