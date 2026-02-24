import argparse
import json
import os

from datasets import load_dataset


def _iter_context_docs(example):
  titles = example['context']['title']
  sentences = example['context']['sentences']
  for title, sent_list in zip(titles, sentences):
    text = ' '.join(sent_list).strip()
    if text:
      yield title, text


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_path', type=str, default='data/passage_corpus/hotpotqa_passages.jsonl')
  parser.add_argument('--splits', nargs='+', default=['train', 'validation'],
                      help='HotpotQA splits to include (train/validation).')
  parser.add_argument('--max_examples_per_split', type=int, default=None,
                      help='Optional cap per split for quick sanity corpora.')
  args = parser.parse_args()

  out_dir = os.path.dirname(args.output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  dedupe = {}
  for split in args.splits:
    raw = load_dataset('hotpot_qa', 'distractor', split=split)
    for i, ex in enumerate(raw):
      if args.max_examples_per_split is not None and i >= args.max_examples_per_split:
        break
      for title, text in _iter_context_docs(ex):
        key = (title, text)
        if key not in dedupe:
          dedupe[key] = None

  with open(args.output_path, 'w') as handle:
    for idx, (title, text) in enumerate(dedupe.keys()):
      row = {
          'doc_id': 'hotpot_%d' % idx,
          'title': title,
          'text': text,
      }
      handle.write(json.dumps(row) + '\n')

  print('Wrote %d passages to %s' % (len(dedupe), args.output_path))


if __name__ == '__main__':
  main()
