import nltk
import sys
import math
from nltk import FreqDist



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

def stampa_lista_tuple(lista):
    # Funzione per stampare una lista di tuple con numeri progressivi
    # Il ciclo for itera attraverso la lista di tuple utilizzando gli indici generati da enumerate

    for i, tupla in enumerate(lista, start=1): # La funzione enumerate restituisce coppie (indice, elemento) dalla lista. L'argomento start=1 indica che l'indice inizia da 1 invece che da 0.
        print(f"{i}. {tupla[0]}: {tupla[1]}")

def get_frequent_PoS_classes(tokens_PoS, PoS_classes):
    tokens_classificati = []

    for token, PoS in tokens_PoS:   # Itera attraverso ogni coppia di token e parte del discorso
        if PoS.startswith(PoS_classes): # Se la parte del discorso inizia con una delle classi specificate
            tokens_classificati.append(token) # Aggiunge il token alla lista dei token classificati
    freq_dist = nltk.FreqDist(tokens_classificati)     # Calcola la frequenza dei token classificati utilizzando FreqDist
    tokens_frequenti = freq_dist.most_common(50) # Ottiene i 50 token più frequenti

    return tokens_frequenti


def top_frequent_ngrams(tokens, n):
    # Funzione per ottenere i top N n-grammi più frequenti
    ngrammi_frequenti = [] # Lista per contenere i risultati
    if n == 1:
        n_grammi = list(tokens)
    elif n == 2: # Quando n è 2 la funzione restituisce una lista di bigrammi 
        n_grammi = list(nltk.bigrams(tokens))
    elif n==3: # Trigrammi
        n_grammi = list(nltk.trigrams(tokens))
    else: # N-grammi
        n_grammi = list(nltk.ngrams(tokens,n))
        
    freq_dist = FreqDist(n_grammi)     # Calcola la distribuzione di frequenza degli n-grammi
    top_ngrammi = freq_dist.most_common(20)
    ngrammi_frequenti.append((n, top_ngrammi)) # Aggiunge i risultati alla lista dei n-grammi frequenti

    return ngrammi_frequenti

def top_frequent_PoS_ngrams(tokens, n):
    # Funzione per ottenere i top N n-grammi di parti del discorso più frequenti

    PoS_tags = []
    for token, PoS in nltk.pos_tag(tokens):  # Ottiene le parti del discorso per ogni token 
        PoS_tags.append(PoS)
    if n == 1:
        ngrammi_PoS_frequenti = FreqDist(PoS_tags).most_common(20)
    elif n == 2: # Bigrammi
        ngrammi_PoS = list(nltk.bigrams(PoS_tags))
        ngrammi_PoS_frequenti = FreqDist(ngrammi_PoS).most_common(20)
    elif n == 3: # Trigrammi
        ngrammi_PoS = list(nltk.trigrams(PoS_tags))
        ngrammi_PoS_frequenti = FreqDist(ngrammi_PoS).most_common(20)
    else: # N-grammi 
        ngrammi_PoS = list(nltk.ngrams(PoS_tags, n))
        ngrammi_PoS_frequenti = FreqDist(ngrammi_PoS).most_common(20)

    return ngrammi_PoS_frequenti

def get_J_N_bigrams(tokens):
    # Funzione per ottenere bigrammi composti da un aggettivo seguito da un sostantivo

    bigrammi_PoS = list(nltk.bigrams(nltk.pos_tag(tokens))) # Ottiene bigrammi di token e relative parti del discorso
    bigrammi_J_N = [] # Lista per contenere i bigrammi di aggettivo e sostantivo

    for bigramma in bigrammi_PoS:
        tokens1, PoS1 = bigramma[0]
        tokens2, PoS2 = bigramma[1]
        # Se il primo elemento del bigramma è un aggettivo e il secondo è un sostantivo aggiunge il bigramma alla lista
        if PoS1.startswith("J") and PoS2.startswith("N"):
            bigrammi_J_N.append((tokens1, tokens2))

    return bigrammi_J_N


def get_top_10_J_N_bigrams(tokens):
    # Funzione per ottenere i top 10 bigrammi di aggettivo e sostantivo più frequenti

    bigrammi_aggettivo_sostantivo = get_J_N_bigrams(tokens)  # Ottiene i bigrammi aggettivo-sostantivo
    freq_dist = FreqDist(bigrammi_aggettivo_sostantivo)     # Calcola la distribuzione di frequenza dei bigrammi
    top_10 = freq_dist.most_common(10)

    return top_10


def probabilita_condizionata_max(bigrammi, tokens):
    # Funzione per calcolare la probabilità condizionata massima di bigrammi

    prob_cond_bigrammi = {} # Dizionario per contenere le probabilità condizionate dei bigrammi
    for bigramma in bigrammi:
        frequenza_bigramma= bigrammi.count(bigramma) # Calcola la frequenza del bigramma contando quante volte appare nella lista dei bigrammi
        frequenza_token= tokens.count(bigramma[0])   # Calcola la frequenza del primo token del bigramma contando quante volte appare nella lista dei token
        probabilita_condizionata = frequenza_bigramma/frequenza_token  # Calcola la probabilità condizionata del bigramma
        prob_cond_bigrammi.update({bigramma: probabilita_condizionata})  # Aggiunge il bigramma e la sua probabilità condizionata al dizionario

    bigrammi_probabilita_decrescente = sorted(prob_cond_bigrammi.items(), key=lambda x: x[1], reverse=True) # Ordina in maniera decrescente
    
    return bigrammi_probabilita_decrescente[:10] # Restituisce i primi 10 bigrammi con la probabilità condizionata massima


def probabilita_congiunta_max(bigrammi, tokens):
    # Funzione per calcolare la probabilità congiunta massima di bigrammi

    prob_congiunta_bigrammi = {}

    for bigramma in bigrammi: 
        frequenza_bigrammi = bigrammi.count(bigramma)  
        frequenza_primo_token = tokens.count(bigramma[0]) 
        prob_condizionata =frequenza_bigrammi/frequenza_primo_token # Calcola la probabilità condizionata del bigramma

        frequenza_relativa =frequenza_primo_token/len(tokens) # Calcola la frequenza relativa
        prob_congiunta = prob_condizionata*frequenza_relativa  # Calcola la probabilità congiunta
        prob_congiunta_bigrammi.update({bigramma: prob_congiunta}) # Aggiunge il bigramma e la sua probabilità congiunta al dizionario
    bigrammi_ordine_decrescente = sorted(prob_congiunta_bigrammi.items(), key=lambda x: x[1], reverse=True)

    return bigrammi_ordine_decrescente[:10]        


def MI_max(bigrammi, tokens):
    # Funzione per ottenere i top 10 bigrammi con Mutual Information massima

    bigrammi_frequenza = nltk.FreqDist(bigrammi).most_common()  # Calcola la distribuzione di frequenza dei bigrammi e ottiene i bigrammi più comuni
    diz_bigrammi_MI = {}
    
    for bigramma, frequenza in bigrammi_frequenza:
        # Calcola la frequenza del primo e secondo token del bigramma nella lista dei token
        frequenza_token1 = tokens.count(bigramma[0])
        frequenza_token2 = tokens.count(bigramma[1])
        MI_bigrammi = math.log ((frequenza * len(tokens)/(frequenza_token1 * frequenza_token2)), 2)   # Calcola il valore di Mutual Information del bigramma
        diz_bigrammi_MI.update({bigramma: MI_bigrammi}) # Aggiunge il bigramma e il suo valore di Mutual Information al dizionario
    
    bigrammi_MI_decrescente = sorted(diz_bigrammi_MI.items(), key=lambda x: x[1], reverse=True)

    return bigrammi_MI_decrescente[:10]


def LMI_max(bigrammi,tokens):
    # Funzione per ottenere i top 10 bigrammi con Local Mutual Information massima
    diz_lmi = {}
    lunghezza_bigrammi = len(bigrammi)

    for bigramma in bigrammi:
        # Calcola la frequenza del bigramma e dei token singoli

        frequenza_bigramma = bigrammi.count(bigramma)
        frequenza_token1 = tokens.count(bigramma[0])
        frequenza_token2 = tokens.count(bigramma[1])

        # Calcola le probabilità del bigramma e dei token individuali
        probabilita_bigramma = frequenza_bigramma / lunghezza_bigrammi
        probabilita_token1 = frequenza_token1 / lunghezza_bigrammi
        probabilita_token2 = frequenza_token2 / lunghezza_bigrammi

        # Calcola la probabilità condizionata
        prob_condizionata = probabilita_bigramma / (probabilita_token1 * probabilita_token2)

        # Calcola la LMI e aggiunge il valore e il rispettivo bigramma al dizioanrio
        lmi_value = frequenza_bigramma * math.log(prob_condizionata, 2)
        diz_lmi.update({bigramma:lmi_value})

    bigrammi_lmi_decrescente = sorted(diz_lmi.items(), key=lambda x: x[1], reverse=True)

    return bigrammi_lmi_decrescente[:10]


def get_tokenized_sentences(frasi):
    # Funzione per ottenere frasi tokenizzate a partire da una lista di frasi

    frasi_tokenizzate = {}
    for frase in frasi:
        # Tokenizza ogni frase e aggiunge al dizionario
        tokens_frase = nltk.tokenize.word_tokenize(frase)
        frasi_tokenizzate[frase] = tokens_frase

    return frasi_tokenizzate

def get_hapax(tokens):
    # Funzione per ottenere hapax all'interno di una lista di tokens

    hapax = []
    for token in set(tokens):
        if tokens.count(token) == 1: # Verifica se il token appare una sola volta nella lista dei tokens
            hapax.append(token)

    return hapax

def get_sentences1020(corpus):
    # Funzione per ottenere frasi con lunghezza tra 10 e 20 token e almeno metà dei token non sono hapax

    frasi, tokens = splitter_and_tokenizer(corpus)
    frasi_tokenizzate = get_tokenized_sentences(frasi)
    hapax = get_hapax(tokens)
    frasi_selezionate = []

    for frase, tokens_frase in frasi_tokenizzate.items():  # Itera attraverso ogni frase e i relativi tokens nel dizionario
        lunghezza_frase = len(tokens_frase)
        if 10 <= lunghezza_frase <= 20:   # Verifica se la lunghezza della frase è tra 10 e 20 token
            conteggio_non_hapax = sum(1 for token in tokens_frase if token not in hapax) # Conta il numero di tokens non hapax nella frase
            if conteggio_non_hapax >= lunghezza_frase / 2: # Verifica se almeno metà dei tokens non sono hapax
                frasi_selezionate.append(frase) # Aggiunge le frasi filtrate

    return frasi_selezionate


def max_min_media_freq(frasi, tokens):
    # Funzione per calcolare la frase con la massima e minima media di distribuzione di frequenza

    frasi_tokenizzate = get_tokenized_sentences(frasi)  # Ottiene frasi tokenizzate a partire dalla lista di frasi
    # Inizializza i valori massimi e minimi
    max_val, min_val = 0, float("inf") 
    frase_max, frase_min = None, None

    for frase, tokens_frase in frasi_tokenizzate.items(): # Itera attraverso ogni frase e i relativi tokens nel dizionario
        somma = sum(tokens.count(token) for token in tokens_frase) # Calcola la somma delle frequenze dei tokens nella frase
        media_distrFreq_frase = round(somma / len(tokens_frase), 3) # Calcola la media della distribuzione di frequenza nella frase
        
        # Aggiorna i valori massimi e minimi se necessario
        if media_distrFreq_frase > max_val:
            max_val, frase_max = media_distrFreq_frase, frase
        
        if media_distrFreq_frase < min_val:
            min_val, frase_min = media_distrFreq_frase, frase

    return frase_max, max_val, frase_min, min_val


def markov2_max(frasi, tokens):
    # Funzione per ottenere la frase con la massima probabilità secondo un modello Markov di ordine 2

    frasi_tokenizzate = get_tokenized_sentences(frasi)
    
    # Ottiene bigrammi e trigrammi dal corpus completo
    bigrammi_corpus = list(nltk.bigrams(tokens))
    trigrammi_corpus = list(nltk.trigrams(tokens))
    max_prob = 0 # Inizializza la massima probabilità
    best_sentence = "" # Inizializza la frase con la massima probabilità

    for frase, tokens_frase in frasi_tokenizzate.items():
        bigrammi_frase = list(nltk.bigrams(tokens_frase))
        trigrammi_frase = list(nltk.trigrams(tokens_frase))

        # Calcola la probabilità condizionata del primo bigramma
        prob_parola1 = tokens.count(tokens_frase[0]) / len(tokens)
        prob_parola2 = bigrammi_corpus.count(bigrammi_frase[0]) / tokens.count(tokens_frase[0])
        prodotto = prob_parola1 * prob_parola2

        for trigramma in trigrammi_frase: # Itera attraverso ogni trigramma nella frase
            freq_trigramma = trigrammi_corpus.count(trigramma)
            freq_bigramma = bigrammi_corpus.count(bigrammi_frase[-1]) if bigrammi_frase else 0
            
            # Calcola la probabilità condizionata della parola corrente
            prob_condizionata_parola = freq_trigramma / freq_bigramma if freq_bigramma != 0 else 0
            prodotto *= prob_condizionata_parola    # Moltiplica la probabilità condizionata al prodotto

        if prodotto > max_prob: # Aggiorna la massima probabilità e la frase con la massima probabilità se necessario
            max_prob = prodotto
            best_sentence = frase
            
    return best_sentence, max_prob

def get_NE(tokens_PoS):
    # Funzione per ottenere le Named Entities

    NE_tree = nltk.ne_chunk(tokens_PoS) # Utilizza la funzione ne_chunk di nltk per identificare le Named Entities
    NE = []

    for nodo in NE_tree: # Itera attraverso ogni nodo nell'albero delle Named Entities
        if hasattr(nodo, 'label'): # Verifica se il nodo ha un'etichetta (è una Named Entity)
            entity_type = nodo.label() # Ottiene il tipo di Named Entity
            tokens = []
            for token, PoS in nodo.leaves():
                tokens.append(token)  # Aggiunge il token alla lista
            # Unisce i token per formare l'entità
            entity = " ".join(tokens)
            NE.append((entity, entity_type))
    return NE

def get_frequent_NE(NE,classe): 
    # La funzione restituisce gli elementi più frequenti (con relativa frequenza) per una classe di NE

    freqdist_NE = FreqDist(NE[classe]).most_common(15)
    return freqdist_NE


def main(filename):

    corpus= leggi_file(filename)

    frasi1, tokens1= splitter_and_tokenizer(corpus)

    tokens_PoS1= PoS_tagger(tokens1)
    print ("\n")
    print ("OUPUT PROGRAMMA 2")
    print ("\n")

    print("1. I TOP 50 Sostantivi più frequenti (con relativa frequenza, ordinata per frequenza decrescente):")
    print ("\n")
    stampa_lista_tuple(get_frequent_PoS_classes(tokens_PoS1,"N"))
    print ("\n")

    print("I TOP 50 Avverbi più frequenti (con relativa frequenza, ordinata per frequenza decrescente):")
    print ("\n")
    stampa_lista_tuple(get_frequent_PoS_classes(tokens_PoS1,"R"))
    print ("\n")

    print("I TOP 50 Aggettivi più frequenti (con relativa frequenza, ordinata per frequenza decrescente):")
    print ("\n")
    stampa_lista_tuple(get_frequent_PoS_classes(tokens_PoS1,"J"))
    print ("\n")

    print("2. I TOP 20 n-grammi (n=[1,2,3,4,5]):")
    print ("\n")
    
    risultati1 = []
    for n in [1, 2, 3, 4, 5]:
        risultati1.extend(top_frequent_ngrams(tokens1, n))

    for n, top_ngrammi in risultati1:
        print(f"\nTop {n}-grammi più frequenti:", top_ngrammi)
    print ("\n")

    print("3. I TOP 20 n-grammi di PoS più frequenti (con relativa frequenza, e ordinati per frequenza decrescente):")
    print ("\n")
    risultati2 = []
    for n in [1, 2, 3]:
        risultati2.extend(top_frequent_PoS_ngrams(tokens1, n))

    for n, ngrammi_PoS in risultati2:
        print(f"\nTop {n}-grammi più frequenti:", ngrammi_PoS)
    print ("\n")

    top_10_bigrammi = get_top_10_J_N_bigrams(tokens1)
    print("4a. I TOP 10 bigrammi aggettivo-sostantivo ordinati per frequenza")
    print ("\n")
    for bigramma, frequenza in top_10_bigrammi:
        print(f"{bigramma}: Frequenza={frequenza}")
    
    print ("\n")
    risultato_prob_condizionata_max = probabilita_condizionata_max(get_J_N_bigrams(tokens1), tokens1)
    print("4b.  I TOP 10 bigrammi aggettivo-sostantivo ordinati per probabilità condizionata massima")
    print ("\n")
    for bigramma, probabilita in risultato_prob_condizionata_max:
        print(f"{bigramma}: Probabilità condizionata massima={probabilita}")
    print ("\n")

    risultato_prob_congiunta = probabilita_congiunta_max(get_J_N_bigrams(tokens1), tokens1)
    print("4c.  I TOP 10 bigrammi aggettivo-sostantivo ordinati per probabilità congiunta massima")
    print ("\n")
    for bigramma, probabilita in risultato_prob_congiunta:
        print(f"{bigramma}: Probabilità congiunta massima={probabilita}")
    print ("\n")

    risultato_MI= MI_max(get_J_N_bigrams(tokens1), tokens1)
    print("4d.  I TOP 10 bigrammi aggettivo-sostantivo ordinati per Mutual Information")
    print ("\n")
    for bigramma, MI in risultato_MI:
        print(f"{bigramma}: Mutual Information massima={MI}")
    print ("\n")
    
    risultato_LMI= LMI_max(get_J_N_bigrams(tokens1), tokens1)
    print("4e.  I TOP 10 bigrammi aggettivo-sostantivo ordinati per Local Mutual Information")
    print ("\n")
    for bigramma, LMI in risultato_LMI:
        print(f"{bigramma}: Local Mutual Information massima={LMI}")
    print ("\n")

    set_top_10_MI = set(risultato_MI)
    set_top_10_LMI = set(risultato_LMI)
    elementi_comuni = set_top_10_MI & set_top_10_LMI
    numero_elementi_comuni = len(elementi_comuni)
    print(f"4f. Numero di elementi comuni tra top 10 MI e LMI: {numero_elementi_comuni}")
    print ("\n")

    frase_max1,max1,frase_min1,min1 = max_min_media_freq(get_sentences1020(corpus),tokens1)
    print(f"5a. La frase con la media della distribuzione di frequenza dei token più alta, ovvero {max1}, è:\n\t{frase_max1}")
    print ("\n")
    print(f"5b. La frase con la media della distribuzione di frequenza dei token più bassa, ovvero {min1}, è:\n\t{frase_min1}")
    print ("\n")

    frase_scelta, max2 = markov2_max(get_sentences1020(corpus),tokens1)
    print(f"5c. La frase con la probabilità più alta secondo un modello di Markov di ordine 2 costruito a partire dal corpus in input è {frase_scelta}, con probabilità {max2} ")

    named_entities = get_NE(tokens_PoS1)
    NE_dict = {}
    for entity, entity_type in named_entities:
        if entity_type in NE_dict.keys():
            NE_dict[entity_type].append(entity)
        else:
            NE_dict[entity_type] = [entity]
    print ("\n")

    print("6. Frequent Named Entities:")
    for entity_type in NE_dict.keys():
        frequent_entities = get_frequent_NE(NE_dict, entity_type)
        print(f"\n{entity_type}: {frequent_entities}")
   

if __name__ == "__main__":
    main(sys.argv[1])

