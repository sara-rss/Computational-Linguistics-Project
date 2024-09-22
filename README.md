# Computational Linguistics Project

## Programma 1

Il codice sviluppato deve prendere in input i due corpora, effettuare le operazioni di annotazione linguistica richieste (sentence splitting, tokenizzazione, PoS tagging, lemmatizzazione), e produrre un confronto dei corpora rispetto a:

1. Numero di frasi e token;
2. Lunghezza media delle frasi in token e lunghezza media dei token, a eccezione della punteggiatura, in caratteri;
3. Numero di Hapax tra i primi 500, 1000, 3000 token, e nell’intero corpus;
4. Dimensione del vocabolario e ricchezza lessicale (Type-Token Ratio, TTR), calcolata per porzioni incrementali di 200 token fino ad arrivare a tutto il testo;
5. Numero di lemmi distinti (i.e., la dimensione del vocabolario dei lemmi);
6. Distribuzione di frasi con polarità positiva e negativa. 

## Programma 2

Il codice sviluppato deve prendere in input un corpus, effettuare le operazioni di annotazione richieste (sentence splitting, tokenizzazione, PoS tagging), ed estrarre le seguenti informazioni:

1. I top-50 Sostantivi, Avverbi e Aggettivi più frequenti (con relativa frequenza, ordinata per frequenza decrescente);
2. I top-20 n-grammi più frequenti (con relativa frequenza, e ordinati per frequenza decrescente);
3. I top 20 n-grammi di PoS più frequenti;
4. I top-10 bigrammi composti da Aggettivo e Sostantivo, ordinati per:
  a. frequenza decrescente, con relativa frequenza
  b. probabilità condizionata massima, e relativo valore di probabilità
  c. probabilità congiunta massima, e relativo valore di probabilità
  d. MI (Mutual Information) massima, e relativo valore di MI
  e. LMI (Local Mutual Information) massima, e relativo valore di MI
  f. Calcolare e stampare il numero di elementi comuni ai top-10 per MI e per LMI
6. Considerate le frasi con una lunghezza compresa tra 10 e 20 token, in cui almeno la metà (considerare la parte intera della divisione per due come valore) dei token occorre almeno 2 volte nel corpus (i.e., non è un hapax), si identifichino:
  a. La frase con la media della distribuzione di frequenza dei token più alta
  b. La frase con la media della distribuzione di frequenza dei token più bassa
  c. La frase con probabilità più alta secondo un modello di Markov di ordine 2 costruito a partire dal corpus di input
7. Estratte le Entità Nominate del testo, identificare per ciascuna classe di NE i 15 elementi più frequenti, ordinati per frequenza decrescente e con relativa frequenza.
