#!/usr/bin/env python3

import argparse
from lib.keyword_search import InvertedIndex, Movie, process_text, read_stopwords, BM25_K1, BM25_B

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

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

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
        case "bm25idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            bm25_idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
            pass
        case "bm25tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            bm25_tf = inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

