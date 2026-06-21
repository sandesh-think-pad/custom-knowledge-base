from .loader import DocumentLoader
from .parser import NodeParser
from .indexer import WeaviateIndexer


def main():
    loader = DocumentLoader()
    documents = loader.load()

    parser = NodeParser()
    nodes = parser.parse(documents)

    print("Connecting to Weaviate...")
    with WeaviateIndexer() as indexer:
        indexer.create_schema()
        indexer.index(nodes)


if __name__ == "__main__":
    main()
