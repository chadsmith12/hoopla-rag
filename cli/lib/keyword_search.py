import json
import math
from operator import index
import pickle
import string
from typing import Counter, TypedDict
from nltk.stem import PorterStemmer
from collections import defaultdict
from pathlib import Path

BM25_K1: float = 1.5
BM25_B: float = 0.75

class Movie(TypedDict):
    id: int
    title: str
    description: str

class InvertedIndex:
    index: defaultdict[str, set[int]] = defaultdict(set)
    docmap: defaultdict[int, Movie] = defaultdict()
    term_frequencies: defaultdict[int, Counter[str]] = defaultdict(Counter)
    doc_lenths: defaultdict[int, int] = defaultdict(int)
    stop_words: list[str] = []

    def __add_document(self: InvertedIndex, doc_id: int, term: str, stop_words: list[str]) -> None:
        tokenized = process_text(term, stop_words)
        for token in tokenized:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokenized)
        self.doc_lenths[doc_id] = len(tokenized)

    def __get_avg_doc_length(self: InvertedIndex) -> float:
        if len(self.doc_lenths) == 0:
            return 0.0
        
        total_doc_length = 0.0
        for doc in self.doc_lenths:
            total_doc_length += self.doc_lenths[doc]

        return total_doc_length / len(self.doc_lenths)

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

    def get_bm25_idf(self: InvertedIndex, term: str) -> float:
        tokenized = process_text(term, self.stop_words)
        if len(tokenized) > 1:
            raise ValueError("too many terms")

        token = tokenized[0]
        number_docs = len(self.docmap)
        doc_count = len(self.index[token])
        numerator = number_docs - doc_count + 0.5
        denominator = doc_count + 0.5

        return math.log((numerator / denominator) + 1)

    def get_bm25_tf(self: InvertedIndex, doc_id: int, term: str, k1: float, b: float = BM25_B) -> float:
        length_norm = 1 - b + b * (self.doc_lenths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self: InvertedIndex, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term, BM25_K1) * self.get_bm25_idf(term)

    def bm25_search(self: InvertedIndex, query: str, limit: int = 5) -> list[tuple[Movie, float]]:
        tokenized = process_text(query, self.stop_words)
        scores: defaultdict[int, float] = defaultdict(float)
        
        for doc_id in self.docmap:
            doc_score = 0.0
            for token in tokenized:
                doc_score += self.bm25(doc_id, token)
            scores[doc_id] = doc_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[tuple[Movie, float]] = []
        for doc_id, score in sorted_docs[:limit]:
            movie = self.docmap[doc_id]
            results.append((movie, score))

        return results
    
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

        doc_length_cache_file = f"{cache_dir}/doc_lengths.pkl"
        with open(doc_length_cache_file, 'wb') as f:
            pickle.dump(self.doc_lenths, f)

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

        doc_lengths_cache_file = Path("cache/doc_lengths.pkl")
        if not doc_lengths_cache_file.exists():
            raise FileNotFoundError(f"{doc_lengths_cache_file} was not found")

        with open(doc_lengths_cache_file, 'rb') as f:
            self.doc_lenths = pickle.load(f)


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
