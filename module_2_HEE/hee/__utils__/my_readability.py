"""https://github.com/mholtzscher/spacy_readability"""
"""https://en.wikipedia.org/wiki/Readability"""

import spacy
from spacy_readability import Readability

# python -m spacy download en
nlp = spacy.load('en')
read = Readability()
nlp.add_pipe(read, last=True)


# doc = nlp("I am some really difficult text to read because I use obnoxiously large words.")

def readable_score(text):
    doc = nlp(text)
    """
    Flesch–Kincaid:
    https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    fk_score = doc._.flesch_kincaid_reading_ease
    fk_grade = doc._.flesch_kincaid_grade_level
    # print('Flesch–Kincaid','score',fk_score,'grade',fk_grade)

    """
    Dale-Chall
    """
    dc_score = doc._.dale_chall
    # print('Dale-Chall','score',dc_score)

    cli_grade = doc._.coleman_liau_index
    # print('coleman_liau_index','grade',cli_grade)

    """
    https://en.wikipedia.org/wiki/Automated_readability_index
    """
    ari_score = doc._.automated_readability_index
    # print('Automated_readability_index','score', ari_score)

    # print(doc._.forcast)

    """
    SMOG Grade
    """
    # print('smog',doc._.smog)

    return fk_score, fk_grade, dc_score, cli_grade, ari_score


if __name__ == '__main__':
    text = "From the creator of Maisy comes the delightful story of a young woodpecker as he journeys through his first day learning to do what woodpeckers do best – peck, peck, peck! After his daddy shows him how, he flies from place to place perfecting his peck, pecking more and more holes right through the pages, until it’s time for bed. The clever rhyming text is a real joy to read aloud and the bright, bold illustrations make this a great book for sharing and interacting with, especially as the holes little woodpecker makes just happen to be the perfect size for little fingers! All the things little woodpecker pecks through will help build vocabulary too and perhaps even inform future reading as he pecks right through a copy of Jane Eyre!"
    readable_score(text)