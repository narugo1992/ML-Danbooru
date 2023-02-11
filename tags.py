import json
import os
import re

import pandas as pd
import spacy
from pybooru import Danbooru
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')


def load_classes_from_local():
    with open('class.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    cnt = len(data)
    assert cnt == max(map(int, data.keys())) + 1
    _, tags = zip(*sorted(map(lambda x: (int(x[0]), x[1]), data.items())))
    return tags


db = Danbooru(
    'danbooru',
    username=os.environ.get('DANBOORU_USERNAME', None),
    api_key=os.environ.get('DANBOORU_TOKEN', None),
)


def _get_danbooru_posts_count(tag: str) -> int:
    return db.count_posts(tag)['counts']['posts']


if __name__ == '__main__':
    df = pd.DataFrame({'tag': load_classes_from_local()})

    items = []
    counts = []
    for name in tqdm(df['tag']):
        text = name
        matching = re.match('^(?P<number>\\d+\\+?)(?P<tail>(boy|girl|koma|other)[\\s\\S]?)$', text)
        if matching:  # 1girl, 1boy, 2girls, 6+boys, etc
            text = matching.group('number') + '_' + matching.group('tail')

        sentence = text.replace('_', ' ')
        for token in nlp(sentence):
            if token.dep_ == 'ROOT':
                items.append((token.lemma_, token.pos_))
                break

        counts.append(_get_danbooru_posts_count(name))

    roots, poss = zip(*items)
    df['root'] = roots
    df['pos'] = poss
    df['count'] = counts

    dist_dir = os.environ.get('DIST_DIR', 'dist')
    os.makedirs(dist_dir, exist_ok=True)
    df.to_csv(os.path.join(dist_dir, 'tags.csv'), index=False)
