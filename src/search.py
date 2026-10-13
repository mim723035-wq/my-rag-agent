import re
import argparse
import json
import logging
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
jieba.setLogLevel(logging.WARNING)


def tokenize(text):
    """中文保留jieba分词；纯英文按单词和数字切分。"""
    text = text.casefold()

    if re.search(r"[\u4e00-\u9fff]", text):
        return [
            word
            for word in jieba.lcut(text)
            if any(char.isalnum() for char in word)
        ]

    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)


class SearchTool:
    def __init__(self, documents_path):
        with Path(documents_path).open("r", encoding="utf-8") as file:
            self.documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if not self.documents:
            raise ValueError("文档库不能为空")

        # 第一版每篇文档都很短，直接将标题和正文一起索引。
        tokenized_documents = [
            tokenize(doc["title"] + "\n" + doc["text"])
            for doc in self.documents
        ]

        if not any(tokenized_documents):
            raise ValueError("文档库没有可索引的词")

        self.document_token_sets = [
            set(tokens) for tokens in tokenized_documents
        ]
        self.index = BM25Okapi(
            tokenized_documents,
            k1=1.5,
            b=0.75,
        )

    def search(self, query, top_k=3):
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        # 排除完全没有词语交集的文档。
        # 不能简单把 BM25 分数当作“相关概率”。
        candidates = [
            index
            for index, tokens in enumerate(self.document_token_sets)
            if query_token_set & tokens
        ]

        # 分数相同时按文档 ID 排序，保持结果稳定。
        ranked_indices = sorted(
            candidates,
            key=lambda index: (
                -float(scores[index]),
                self.documents[index]["id"],
            ),
        )

        results = []
        for index in ranked_indices[:top_k]:
            doc = self.documents[index]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(float(scores[index]), 4),
            })

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    tool = SearchTool(ROOT / "data" / "documents.jsonl")
    results = tool.search(args.query, args.top_k)

    print(json.dumps({
        "query": args.query,
        "query_tokens": tokenize(args.query),
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()