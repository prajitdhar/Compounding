#!/usr/bin/env python3
"""Generate synthetic benchmark-shaped test inputs, NOT human judgments.

The seven familiar spelling/lemma pairs only exercise the lexical-substitution
specification. Every numeric rating and every other pair is fabricated.
"""
import argparse
import csv
from pathlib import Path

COLUMNS = ['modifier', 'head', 'avgModifier', 'stdevModifier', 'avgHead',
           'stdevHead', 'compositionality', 'stdevHeadModifier', 'is_adj',
           'compound', 'source', 'is_original']
ALTERNATIVES = {
    'reddy90': [('call', 'centre', 'call', 'center')],
    'cordeiro90': [('benign', 'tumour', 'benign', 'tumor'),
                  ('computer', 'program', 'computer', 'programme'),
                  ('grey', 'matter', 'gray', 'matter'),
                  ('labour', 'union', 'labor', 'union')],
    'cordeiro100': [('baby', 'blues', 'baby', 'blue'),
                   ('contact', 'lenses', 'contact', 'lens')],
}

def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for source, count, filename in [('reddy90', 90, 'reddy_90.txt'),
            ('cordeiro90', 90, 'cordeiro_90.txt'),
            ('cordeiro100', 100, 'cordeiro_100.txt')]:
        rows = []
        alternatives = ALTERNATIVES[source]
        for index in range(count):
            modifier, head = (alternatives[index][:2] if index < len(alternatives)
                else (f'synthetic{source}modifier{index}', f'synthetic{source}head{index}'))
            score = round(((index * 17 + count) % 49) / 10, 1)
            row = dict(zip(COLUMNS, [modifier, head, score, 0.1,
                round((score + 0.7) % 5, 1), 0.2, score, 0.3,
                index % 4 == 0 or (source == 'reddy90' and index in (2, 3)),
                f'{modifier}_{head}', source, True]))
            rows.append(row)
            if index < len(alternatives):
                modified = dict(row)
                modified['modifier'], modified['head'] = alternatives[index][2:]
                modified['compound'] = modified['modifier'] + '_' + modified['head']
                modified['is_original'] = False
                rows.append(modified)
        with (directory / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    generate(parser.parse_args().directory)
