"""Download Wikipedia passage corpus from HuggingFace and save as JSONL.

Uses the florin-hf/wiki_dump2018_nq_open dataset (21M passages from DPR
Wikipedia corpus). Supports downloading a subset to fit in limited RAM/disk.

Usage:
  # Download 500K passages (recommended for 16GB RAM machines):
  python scripts/download_wiki_corpus.py --max_docs 500000

  # Download full corpus (needs 32GB+ RAM and ~15GB disk):
  python scripts/download_wiki_corpus.py
"""
import argparse
import json
import os
import time

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Max passages to download. None = full 21M corpus.')
    parser.add_argument('--output_path', type=str,
                        default='data/passage_corpus/wiki_passages.jsonl')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode (saves RAM, slower).')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    tag = '%dk' % (args.max_docs // 1000) if args.max_docs else 'full'
    if args.max_docs:
        base, ext = os.path.splitext(args.output_path)
        args.output_path = '%s_%s%s' % (base, tag, ext)

    if os.path.exists(args.output_path):
        n_existing = sum(1 for _ in open(args.output_path))
        print('Output already exists: %s (%d lines)' % (args.output_path, n_existing))
        print('Delete it to re-download, or use a different --output_path.')
        return

    print('Loading dataset (streaming=%s) ...' % args.streaming)
    t0 = time.time()

    if args.streaming:
        ds = load_dataset('florin-hf/wiki_dump2018_nq_open', split='train',
                          streaming=True)
    else:
        if args.max_docs and args.max_docs <= 1_000_000:
            ds = load_dataset('florin-hf/wiki_dump2018_nq_open', split='train',
                              streaming=True)
        else:
            ds = load_dataset('florin-hf/wiki_dump2018_nq_open', split='train')

    n_written = 0
    with open(args.output_path, 'w') as f:
        for row in ds:
            if args.max_docs and n_written >= args.max_docs:
                break
            doc = {
                'doc_id': str(n_written),
                'title': row.get('title', ''),
                'text': row.get('text', ''),
            }
            f.write(json.dumps(doc) + '\n')
            n_written += 1
            if n_written % 50000 == 0:
                elapsed = time.time() - t0
                print('  %d passages written (%.1fs)' % (n_written, elapsed))

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print('Done: %d passages -> %s (%.1f MB, %.1fs)' % (
        n_written, args.output_path, size_mb, elapsed))


if __name__ == '__main__':
    main()
