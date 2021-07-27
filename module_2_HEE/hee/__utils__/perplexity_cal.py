# https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
import collections, nltk
# we first tokenize the text corpus
corpus =" Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus"
tokens = nltk.word_tokenize(corpus)

#here you construct the unigram language model
def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model [f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model

def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity


if __name__=="__main__":
    testset1 = "Monty"
    testset2 = "abracadabra gobbledygook rubbish"

    model = unigram(tokens)
    print(perplexity(testset1, model))
    print(perplexity(testset2, model))
