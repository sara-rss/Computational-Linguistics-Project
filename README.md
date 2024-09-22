# Computational-Linguistics-Project

Linguistica Computazionale, Università di Pisa, a.a. 2022/2023

Obiettivo
Realizzazione di due programmi scritti in Python su Jupyter Notebook che utilizzino i moduli di NLTK per analizzare linguisticamente due corpora di testo inglese, confrontarli sulla base di alcuni indici statistici, ed estrarre da essi informazioni.

Programma 1
Il codice sviluppato deve prendere in input i due corpora, effettuare le operazioni di annotazione linguistica richieste (sentence splitting, tokenizzazione, PoS tagging, lemmatizzazione), e produrre un confronto dei corpora rispetto a:

Numero di frasi e token;
Lunghezza media delle frasi in token e lunghezza media dei token, a eccezione della punteggiatura, in caratteri;
Numero di Hapax tra i primi 500, 1000, 3000 token, e nell’intero corpus;
Dimensione del vocabolario e ricchezza lessicale (Type-Token Ratio, TTR), calcolata per porzioni incrementali di 200 token (i.e., i primi 200, i primi 400, i primi 600, ...);
Numero di lemmi distinti (i.e., la dimensione del vocabolario dei lemmi).
Programma 2
Il codice sviluppato deve prendere in input un corpus, effettuare le operazioni di annotazione richieste (sentence splitting, tokenizzazione, PoS tagging), ed estrarre le seguenti informazioni:

La sequenza ordinata per frequenza decrescente, con relativa frequenza, di:
10 PoS, bigrammi di PoS, e trigrammi di PoS più frequenti
20 Sostantivi, Avverbi, e Aggettivi più frequenti
Estratti i bigrammi composti da Aggettivo e Sostantivo mostare:
I 20 più frequenti, con relativa frequenza
I 20 con probabilità condizionata massima, e relativo valore di probabilità c. I 20 con forza associativa (Pointwise Mutual Information, PMI) massima, e relativa PMI
Considerate le frasi con una lunghezza compresa tra 10 e 20 token, in cui almeno la metà (considerare la parte intera della divisione per due come valore) dei token occorre almeno 2 volte nel corpus (i.e., non è un hapax), si identifichino:
La frase con la media della distribuzione di frequenza dei token più alta
La frase con la media della distribuzione di frequenza dei token più bassa c. La frase con probabilità più alta secondo un modello di Markov di ordine 2 costruito a partire dal corpus di input
Estratte le Entità Nominate del testo, identificare per ciascuna classe di NE i 15 elementi più frequenti, ordinati per frequenza decrescente e con relativa frequenza.
