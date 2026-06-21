from llama_index.core.node_parser import SentenceWindowNodeParser


class NodeParser:
    """Splits documents into overlapping sentence-window nodes for context-aware retrieval."""

    def __init__(self, window_size: int = 3):
        self._parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    def parse(self, documents) -> list:
        nodes = self._parser.get_nodes_from_documents(documents)
        print(f"Parsed {len(nodes)} nodes.")
        return nodes
