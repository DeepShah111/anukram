# Hybrid retrieval: bge-m3 dense (ChromaDB) + BM25 sparse, fused with weighted RRF and balanced per case.

import os
import pickle
import hashlib
import tempfile
import shutil

from langchain_chroma import Chroma
from src.jina_embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import (
    logger, VECTOR_DB_DIR, TOP_K_VECTORS, RRF_K, MAX_CHUNKS_PER_CASE,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, setup_environment,
)


class CustomHybridRetriever:
    # Fuses dense + sparse results with weighted RRF, then balances chunks per case.
    def __init__(self, dense_retriever, sparse_retriever, dense_weight=0.5, sparse_weight=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _accumulate(self, docs, weight, scores, doc_map):
        # Add each document's weighted RRF contribution: weight * 1/(rank + K).
        for rank, doc in enumerate(docs):
            chunk_id = doc.metadata.get(
                "chunk_id", hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
            )
            doc_map[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (rank + RRF_K))

    def _balance_by_case(self, ranked, doc_map):
        # Take top results but cap how many chunks come from any single case.
        case_counts, balanced = {}, []
        for chunk_id, _ in ranked:
            if len(balanced) >= TOP_K_VECTORS:
                break
            doc = doc_map[chunk_id]
            case_id = doc.metadata.get("case_id", "unknown")
            if case_counts.get(case_id, 0) < MAX_CHUNKS_PER_CASE:
                case_counts[case_id] = case_counts.get(case_id, 0) + 1
                balanced.append(doc)
        # Fill remaining slots if the cap left us short (e.g. only one case matched).
        if len(balanced) < TOP_K_VECTORS:
            seen = {d.metadata.get("chunk_id") for d in balanced}
            for chunk_id, _ in ranked:
                if len(balanced) >= TOP_K_VECTORS:
                    break
                if chunk_id not in seen:
                    balanced.append(doc_map[chunk_id])
                    seen.add(chunk_id)
        return balanced

    def invoke(self, query):
        # Run both retrievers, fuse, and return balanced results.
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        scores, doc_map = {}, {}
        self._accumulate(dense_docs, self.dense_weight, scores, doc_map)
        self._accumulate(sparse_docs, self.sparse_weight, scores, doc_map)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return self._balance_by_case(ranked, doc_map)


class HybridRetrievalEngine:
    # Builds or loads the Chroma + BM25 indexes and returns a ready hybrid retriever.
    def __init__(self):
        self.bm25_path = os.path.join(VECTOR_DB_DIR, "bm25_index.pkl")
        self.bm25_hash_path = os.path.join(VECTOR_DB_DIR, "bm25_index.sha256")
        logger.info("Loading embedding model %s on %s", EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.retriever = None

    @staticmethod
    def _file_sha256(path):
        # Return the SHA-256 of a file's contents.
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _write_bm25(self, sparse_retriever):
        # Save BM25 atomically with an integrity hash so a half-written index can't corrupt a run.
        with tempfile.NamedTemporaryFile(dir=VECTOR_DB_DIR, suffix=".pkl", delete=False) as tmp:
            pickle.dump(sparse_retriever, tmp)
            tmp_path = tmp.name
        file_hash = self._file_sha256(tmp_path)
        shutil.move(tmp_path, self.bm25_path)
        with open(self.bm25_hash_path, "w") as hf:
            hf.write(file_hash)
        logger.info("BM25 index saved with integrity hash.")

    def _load_bm25(self):
        # Load BM25 and verify its integrity hash before use.
        with open(self.bm25_hash_path) as hf:
            expected = hf.read().strip()
        if self._file_sha256(self.bm25_path) != expected:
            raise RuntimeError("BM25 integrity check failed. Delete artifacts/vector_db and rebuild.")
        with open(self.bm25_path, "rb") as f:
            return pickle.load(f)

    def build(self, document_chunks=None, rebuild=False):
        # Load existing indexes when present, else build them from the given chunks.
        setup_environment()
        chroma_exists = os.path.isdir(VECTOR_DB_DIR) and any(
            f != "bm25_index.pkl" and f != "bm25_index.sha256" for f in os.listdir(VECTOR_DB_DIR)
        )
        bm25_exists = os.path.exists(self.bm25_path)

        if chroma_exists and bm25_exists and not rebuild:
            logger.info("Loading existing indexes from disk (warm start).")
            vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embedding_model)
            sparse_retriever = self._load_bm25()
        else:
            if not document_chunks:
                raise ValueError("No indexes on disk and no document_chunks provided to build them.")
            logger.info("Building dense (ChromaDB) index...")
            vector_store = Chroma.from_documents(
                documents=document_chunks, embedding=self.embedding_model, persist_directory=VECTOR_DB_DIR
            )
            logger.info("Building sparse (BM25) index...")
            sparse_retriever = BM25Retriever.from_documents(document_chunks)
            sparse_retriever.k = TOP_K_VECTORS
            self._write_bm25(sparse_retriever)

        dense_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_VECTORS})
        self.retriever = CustomHybridRetriever(dense_retriever, sparse_retriever)
        logger.info("Hybrid retrieval engine ready.")
        return self.retriever