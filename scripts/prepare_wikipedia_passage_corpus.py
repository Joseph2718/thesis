import argparse
import csv
import json
import os
import pathlib


def _print_format_help():
  print('No --input_corpus_path provided.')
  print('Expected input format:')
  print('  1) JSONL: each line with fields doc_id/id, optional title, and text/passage.')
  print('  2) TSV: doc_id\\ttext OR doc_id\\ttitle\\ttext')
  print('Example:')
  print('  python scripts/prepare_wikipedia_passage_corpus.py --input_corpus_path /path/to/wiki_passages.jsonl')


def _normalize_json_row(row, default_doc_id):
  doc_id = row.get('doc_id', row.get('id', row.get('pid', row.get('passage_id', default_doc_id))))
  title = row.get('title', row.get('wikipedia_title', ''))
  text = row.get('text', row.get('passage', row.get('contents', '')))
  if text is None:
    text = ''
  return str(doc_id), str(title or ''), str(text).strip()


def _iter_jsonl(path):
  with open(path, 'r') as handle:
    for i, line in enumerate(handle):
      if not line.strip():
        continue
      row = json.loads(line)
      doc_id, title, text = _normalize_json_row(row, default_doc_id='row_%d' % i)
      if text:
        yield doc_id, title, text


def _iter_tsv(path):
  with open(path, 'r') as handle:
    reader = csv.reader(handle, delimiter='\t')
    for i, row in enumerate(reader):
      if not row:
        continue
      if i == 0 and row[0].lower() in ('doc_id', 'id', 'pid', 'passage_id'):
        continue
      if len(row) == 1:
        continue
      if len(row) == 2:
        doc_id, text = row
        title = ''
      else:
        doc_id = row[0]
        title = row[1]
        text = '\t'.join(row[2:])
      text = text.strip()
      if text:
        yield str(doc_id), str(title), str(text)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_corpus_path', type=str, default=None,
                      help='Path to Wikipedia passage corpus in JSONL or TSV format.')
  parser.add_argument('--output_path', type=str, default='data/passage_corpus/wiki_passages.jsonl')
  parser.add_argument('--max_docs', type=int, default=None,
                      help='Optional cap for quick preprocessing sanity checks.')
  args = parser.parse_args()

  if not args.input_corpus_path:
    _print_format_help()
    return

  input_path = pathlib.Path(args.input_corpus_path)
  suffix = input_path.suffix.lower()
  if suffix == '.jsonl':
    iterator = _iter_jsonl(args.input_corpus_path)
  elif suffix in ('.tsv', '.txt'):
    iterator = _iter_tsv(args.input_corpus_path)
  else:
    raise ValueError('Unsupported corpus format: %s (expected .jsonl or .tsv)' % suffix)

  out_dir = os.path.dirname(args.output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  n_docs = 0
  with open(args.output_path, 'w') as handle:
    for doc_id, title, text in iterator:
      if args.max_docs is not None and n_docs >= args.max_docs:
        break
      out = {
          'doc_id': doc_id,
          'title': title,
          'text': text,
      }
      handle.write(json.dumps(out) + '\n')
      n_docs += 1

  print('Wrote %d normalized passages to %s' % (n_docs, args.output_path))


if __name__ == '__main__':
  main()
