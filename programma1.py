import sys
import nltk
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report



def leggi_file(filename): 

    # Funzione per leggere il contenuto di un file
    with open (filename, "r") as infile:
        contenuto = infile.read() 
    return contenuto 


def splitter_and_tokenizer(testo): 

     # Suddivide il testo in frasi 
    frasi= nltk.tokenize.sent_tokenize(testo) 
    tokens=[] # Lista per contenere tutti i token
    for frase in frasi: 
        tokens_frase= nltk.tokenize.word_tokenize(frase) # Tokenizza la frase
        tokens.extend(tokens_frase) # Estende la lista di token con i token della frase corrente
        
    return frasi, tokens


def PoS_tagger(tokens): 

     # Esegue il part-of-speech tagging
    tokens_POS= nltk.tag.pos_tag(tokens)

    return tokens_POS # Restituisce la lista di token con le relative parti del discorso

def medium_length_sentences(testo): 

    frasi, tokens= splitter_and_tokenizer(testo) # Ottiene la lista delle frasi e dei token dal testo utilizzando la funzione splitter_and_tokenizer
    lung_media_frasi = len(tokens)/len(frasi) # Calcola la lunghezza media delle frasi

    return lung_media_frasi

def medium_length_tokens(tokens): 
    numero_caratteri= 0
    for token in tokens:
        if re.match(r'\b[\w]+\b', token):  # Utilizza un'espressione regolare per verificare se il token contiene solo caratteri alfanumerici
            numero_caratteri+=len(token)
    # Calcola la lunghezza media dei token
    lung_media_tokens = numero_caratteri/len(tokens) 

    return lung_media_tokens

def numero_hapax(tokens): 
    lista_hapax = [] 

    # Itera attraverso ogni token unico nella lista
    for token in list(set(tokens)): 
        freq_token= tokens.count(token)  # Conta la frequenza del token nella lista originale

         # Aggiunge il token alla lista degli hapax se compare solo una volta
        if freq_token == 1: 
            lista_hapax.append(token) 
    return len(lista_hapax) 


def hapax_in_corpus(tokens):

    # Funzione per calcolare il numero di hapax in diverse porzioni di un corpus
    hapax_counts = [] 
    hapax_counts.append(numero_hapax(tokens[:500]))   # Porzione da 500 token
    hapax_counts.append(numero_hapax(tokens[:1000]))  # Porzione da 1000 token 
    hapax_counts.append(numero_hapax(tokens[:3000]))  # Porzione da 3000 token
    hapax_counts.append(numero_hapax(tokens))      # Intero corpus

    return hapax_counts

def calculate_V_and_TTR(tokens):

    # Funzione per calcolare la dimensione del vocabolario e la Type-Token Ratio per diverse porzioni di un corpus
    dimensione_vocabolario=[]
    valore_TTR=[]

    # Itera attraverso il corpus con incrementi di 200 token
    for i in range(200, len(tokens), 200): 
        sottoinsieme_tokens= tokens[:i] #Si crea un sottoinsieme di token prendendo i primi i token dal corpus

        # Calcola la dimensione del vocabolario per il sottoinsieme corrente
        vocabolario= (set(sottoinsieme_tokens))
        dimensione_vocabolario.append(len(vocabolario))

        # Calcola la TTR per il sottoinsieme corrente
        TTR = len(vocabolario) / i
        valore_TTR.append(TTR)
        print(f"Porzione {i} token - Vocabolario: {len(vocabolario)}, TTR: {TTR}")

    return dimensione_vocabolario, valore_TTR


def Vocabolario_lemmi(tokens):

    # Funzione per calcolare la dimensione del vocabolario dei lemmi
    lemmatizzatore = WordNetLemmatizer() 
    lemmi = [] # Lista per contenere i lemmi

    # Itera attraverso ogni token e la rispettiva parte del discorso
    for token, pos in PoS_tagger(tokens): 
        wordnet_pos = Wordnet_postagging(pos)   # Converte la parte del discorso in formato WordNet
        lemmatized = lemmatizzatore.lemmatize(token, pos=wordnet_pos)    # Esegue la lemmatizzazione del token utilizzando la parte del discorso
        lemmi.append(lemmatized)  # Aggiunge il lemma alla lista

    vocabolario = list(set(lemmi)) # Crea il vocabolario
    
    return len(vocabolario)


def Wordnet_postagging(pos_tag):

    # Mappa la parte del discorso NLTK alla parte del discorso WordNet corrispondente
    if pos_tag.startswith('N'): # Sostantivi
        return wordnet.NOUN
    elif pos_tag.startswith('V'): # Verbi
        return wordnet.VERB
    elif pos_tag.startswith('R'): # Avverbi
        return wordnet.ADV
    elif pos_tag.startswith('J'): # Aggettivi
        return wordnet.ADJ
    else:
        return wordnet.NOUN 

    
def get_trained_model():

    # Carica le recensioni di film dalla directory 'movie_reviews'
    moviedir = 'movie_reviews'
    movie_dataset = load_files(moviedir, shuffle=True) #loading all files

    # Suddivide il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(movie_dataset.data, 
                                                    movie_dataset.target, 
                                                    test_size = 0.20, 
                                                    random_state = 32)
    
    # Crea un vettorizzatore TF-IDF 
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, max_features=3000)

    # Inizializza un classificatore Multinomial Naive Bayes
    classifier = MultinomialNB()

    # Crea una pipeline che combina il vettorizzatore TF-IDF e il classificatore
    pipeline = Pipeline([
    ('tfidf-vectorizer', vectorizer),
    ('multinomialNB', classifier)
    ])

    # Addestra il modello utilizzando il set di addestramento
    trained_model = pipeline.fit(X_train, y_train)
    
    return trained_model


def polarity (testo, trained_model):

    # Sentences splitting
    sentences = nltk.tokenize.sent_tokenize(testo) 

    # Predice la polarità di ciascuna frase utilizzando il modello addestrato
    predicted = trained_model.predict(sentences)

    # Conta il numero di frasi classificate come "NEG" e "POS"
    num_neg = sum(predicted == 0)
    num_pos = sum(predicted == 1)
    
    return num_neg, num_pos
    


def main(filename1, filename2):
    trained_pipeline= get_trained_model()

    corpus1= leggi_file(filename1)
    corpus2=leggi_file(filename2)

    frasi1, tokens1= splitter_and_tokenizer(corpus1)
    frasi2, tokens2= splitter_and_tokenizer(corpus2)

    print ("\n")
    print ("OUPUT PROGRAMMA 1")
    print ("\n")

    print("1. Confronto tra numero di token e numero di frasi")
    print("Il primo corpus ha", len(frasi1), "frasi, e", len(tokens1), "token.")
    print("Il secondo corpus ha", len(frasi2), "frasi, e", len(tokens2), "token.")
    print ("\n")

    tokens_PoS1= PoS_tagger(tokens1)
    tokens_PoS2= PoS_tagger(tokens2)

    lung_media_frasi1= medium_length_sentences(corpus1)
    lung_media_frasi2= medium_length_sentences(corpus2)
    lung_media_tokens1= medium_length_tokens(tokens1)
    lung_media_tokens2= medium_length_tokens(tokens2)

    print("2. Confronto tra lunghezza media delle frasi in token e lunghezza media dei token a eccezione della punteggiatura, in caratteri")
    print("Nel primo corpus le frasi hanno in media", lung_media_frasi1, "token, i token hanno in media", lung_media_tokens1, "caratteri")
    print("Nel primo corpus le frasi hanno in media", lung_media_frasi2, "token, i token hanno in media", lung_media_tokens2 ,"caratteri")
    print ("\n")

    risultati1 = hapax_in_corpus(tokens1)
    risultati2= hapax_in_corpus(tokens2)
    print("3. Confronto il numero di hapax ogni 500, 1000 e 3000 token e nell'intero corpus.")
    print("Nel primo corpus si hanno")
    print(f"Hapax tra i primi 500 token: {risultati1[0]}")
    print(f"Hapax tra i primi 1000 token: {risultati1[1]}")
    print(f"Hapax tra i primi 3000 token: {risultati1[2]}")
    print(f"Hapax nell'intero corpus: {risultati1[3]}")
    print("Nel secondo corpus si hanno")
    print(f"Hapax tra i primi 500 token: {risultati2[0]}")
    print(f"Hapax tra i primi 1000 token: {risultati2[1]}")
    print(f"Hapax tra i primi 3000 token: {risultati2[2]}")
    print(f"Hapax nell'intero corpus: {risultati2[3]}")
    print ("\n")
    
    print("4. Confronto la dimensione del vocabolario e la ricchezza lessicale (TTR) calcolata per porzioni incrementali di 200 token fino ad arrivare a tutto il resto")
    dimensione_vocabolario1, TTR1= calculate_V_and_TTR(tokens1)
    dimensione_vocabolario2, TTR2= calculate_V_and_TTR(tokens2)
    print ("\n")
    
    vocabolario_lemmi1= Vocabolario_lemmi(tokens1)
    vocabolario_lemmi2= Vocabolario_lemmi(tokens2)
    print("5. Confronto la dimensione del vocabolario dei lemmi")
    print("Nel primo corpus la dimensione del vocabolario dei lemmi è", (vocabolario_lemmi1))
    print("Nel secondo corpus la dimensione del vocabolario dei lemmi è", (vocabolario_lemmi2))
    print ("\n")

    num_neg, num_pos= polarity(corpus1, trained_pipeline)
    num_neg2, num_pos2= polarity(corpus2, trained_pipeline)
    print("6. Confronto la distribuzione di frasi con polarità negativa e positiva")
    print("Nel primo corpus ci sono", num_neg, "frasi NEG e", num_pos, "POS. ")
    print("Nel secondo corpus ci sono", num_neg2, "frasi NEG e", num_pos2, "POS.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


