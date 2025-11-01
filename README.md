# RAG Course

Course from [boot.dev](https://boot.dev) going over learning RAG.

## Text Processing

When doing keyword search exact search isn't what the user is looking for. We can go through a text processing pipeline:

1: Case sensitivity - convert all characters to lowercase
2: Punctuation - Remove the punctuation. We don't care about periods or commas 
3: Tokenization - Break the text into individual words
4: Stop Words - Remove common [stop words](https://en.wikipedia.org/wiki/Stop_word)
5: Steming - Keep only the [stem](https://en.wikipedia.org/wiki/Word_stem) of words
