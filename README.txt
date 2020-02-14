Struttura del programma:
File pdfflow.py:
class mkPDF:
    loading delle griglie da file e chiamata dell'interpolatore
    membri:
        fname: string, nome file da cui fare loading dei dati
        num_members: numero di membri nella cartella (TODO)     
        subgrid: lista di oggetti subgrid_pdf
    metodi:
        __init__: input (fname, dirname):
                    fname nella forma '<set_name>/<member_number>'
                    dirname (opzionale), cartella dove sono salvati i PDFsets, default: './local/share/LHAPDF/'
        _xfxQ2: wrapper della tf.function per il metodo _xfxQ2_fn
        _xfxQ2_fn: input: arrays (a_x, a_Q2): chiama l'interpolatore/estrapolatore per fittare i punti che vengono forniti
                deve fare il masking per assegnare ciascun punto all'interno della griglia corretta in base alla scala di energia corrispondente
        xfxQ2: wrapper più esterno, ritorna a_x, a_Q2, f: f è dict se il flavor PID non è specificato, altrimenti è un array

        xfxQ: input: arrays (a_x, a_Q): quadra il vettore a_Q e chiama la funzione xfxQ2 (todo)


File subgrid.py:
class subgrid:
    ciascuna istanza di questa classe è una subgrid all'interno del filed della pdf
    membri:
        x: tensor, vettore ordinata dei knot della subgrid sull'asse x
        Q: tensor, vettore dei knot della subgrid su asse Q
        Q2: tensor, vettore ordinata dei knot della subgrid sull'asse Q2
        flav: array, vettore corrispondente al flavor scheme usato (deve essere uguale per tutte le subgrids)
        logx: tensor, logaritmo elementwise di x
        logQ2: tensor, logaritmo elementwise di Q2
        x_Min: valore minimo su asse x
        x_Max: valore massimo su asse x
        Q2_min: valore minimo su asse Q2
        Q2_max: valore massimo su asse Q2
        values: tensor, vettore dei valori della subgrid letti da file
    metodi:
        print_summary: stampa un riassunto della subgrid
        interpolate: (la main function) dati due array (a_x, a_Q2) calcola per ciascun elemento il valore della pdf interpolato in quel punto
        get_value: dati due array di knots, calcola il valore della griglia in ciascun punto
        remove_edge_stripes: divide i punti di input in due sottoinsiemi, quelli vicini al bordo (interpolazione bilineare) e quelli lontani dal bordo (interpolazione bicubica)
        two/four neighbour knots: prepara il terreno per l'interpolazione lineare/cubica ritornando per ciascun punto, i necessari valori vicini dei knots e della pdf
        df_dx: calcola le necessarie derivate della funzione nei punti della griglia vicini a ciascun punto di input

        extrapolate: dati due array di dati extra Physical Range, calcola il valore della pdf estrapolato in quel punto (TODO)

def linear_interpolation: implementazione dell'interpolazione lineare monodimensionale
def cubic_interpolation: implementazione dell'interpolazione cubica monodimensionale

Algorithm:
    L'algoritmo prevede una prima fase in cui si caricano le pdf da file
    Una seconda fase in cui si ottengono le pdf interpolate in ciascun punto richiesto:
        (a_x, a_Q2) --> xf(a_x, a_Q2)


Obiettivi:

    - Scrivere funzioni che siano high level tanto quanto lo sono quelle di lhapdf, possibilmente con gli stessi nomi e le stesse signatures
    - Implementare una funzione plot che generi una figura prendendo in input i vettori di drawn points nel piano (X,Q2)
    - Valutare le performance in termini di tempo generando un numero elevato di punti: vedere come varia la dipendenza dal tempo di questi algoritmi al variare del numero di drawn points
    - Per il momento posso solo caricare un file singolo di una pdf, non tutto un set ---> implementare anche questa funzione come per lhapdf


Note:
    Discrepanze di implementazione nell'algoritmo:
    - non sono riuscito a capire quale subgrid passa lhapdf alla funzione LogBicubicInterpolator::_interpolateXQ2 implementata nel file LHAPDF-6.2.3/src/LogBicubicInterpolator.cc
    - sempre nella stessa funzione vengono eseguiti degli if per fare lo switch tra bilinear e bicubic interpolation. pdfflow esegue invece la bilinear nella striscia più esterna di ogni subgrid (funzione remove_edge_stripes in subgrid.py), mentre nella restante parte interna ho eseguito la bicubic. La bilinear infatti richiede 4 knots (2x2, uno prima e uno dopo su ogni asse) attorno al query point, mentre la bicubic ne richiede 16 (4x4, due prima e due dopo su ogni asse). Così facendo si può evitare il comando if nella funzione df_dx, che invece viene chiamato ripetutamente dall'algoritmo di lhapdf.