"""
CORET CORET
offer(Ethiopian Airlines, climate-controlled services)
offer(Dokasch, Opticooler container)
equip(Opticooler, battery powered cooling compressors)
equip(Opticooler, battery powered cooling heaters)
is_type_of(pharmaceuticals, temperature sensitive goods)
is_type_of(insulin, pharmaceuticals)
is_type_of(vaccines, pharmaceuticals)
is_a(Dokasch, ULD Provider)
located(Ethiopian Airlines, Addis Ababa)
handle(Addis Ababa, temperature sensitive goods)
handle(Dokasch Opticooler, termperature-sensitive shipments)
offer(Ethiopian Airlines, Dokasch active container)

(Brussels Airlines Cargo, Air Logistics Group)
offer(Brussels Airlines Cargo, belly-hold space)

Extract terms --> Find Lexical relations in the text, analyze
Extract pairs of terms --> analyze if it is related terms
"""
import re
import nltk
import stanza


class Relator(object):

    def __init__(self, extracted_terms):
        self.extracted_terms = extracted_terms

    @staticmethod
    def find_occurrences(token, text):
        occurrences = [m.start() for m in re.finditer(token, text, re.IGNORECASE)]
        print([text[occurrence:occurrence+len(token)] for occurrence in occurrences])


if __name__ == "__main__":
    test_extracted_tokens = "dokasch|ethiopian airlines|ethiopian cargo|opticooler container|director|addis ababa hub|heaters|compressors|opticooler|customers|self|temperature control|logistics services|since february|temperature sensitive goods"
    test_token = "ethiopian airlines"
    test_text = """Egyptair Cargo targets growth with new freighters"""
    # relator = Relator(test_extracted_tokens)
    # Relator.find_occurrences(test_token, test_text)

    # parser = nltk.CoreNLPParser(url=f"http://localhost:9000")
    # annotated = annotate_sentence(test_text)
    # print([(word["word"], word["pos"])for sentence in annotated for word in sentence])
    #
    stanza_nlp = stanza.Pipeline("en", package="gum", processors="tokenize,pos,lemma,depparse", verbose=False)
    doc = stanza_nlp(test_text)
    print(doc)
    # print([(word.text, word.xpos) for sentence in doc.sentences for word in sentence.words])

    # from nltk.corpus import stopwords
    # from spacy.lang.en.stop_words import STOP_WORDS
    # print(stopwords.words("english"))
    # print(STOP_WORDS)
