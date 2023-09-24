"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 2
"""

"""Q 1.1"""
def rev_complement(seq):
    """
    Q:  find the reverse complement of the sequence.
        note, assume seq contains all capital letters and are restricted
        to A, T, G, and C.
    Ex:
    >>> find_complement(seq="ATGTACTAGCTA")
    TAGCTAGTACAT

    Input
    - seq: the sequence to find the complement for

    Output
    - comp: the complement
    """
    seq = seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c")
    seq = seq.upper()
    seq = seq[::-1]
    return seq


"""Q 1.2"""
def gc_content(seq):
    """
    Q:  find the GC% of the sequence.
    Ex:
    >>> gc_content("ATCGACTCGAGTCGTACGTTCACG")
    0.5416666666666666

    Input
    - seq: sequence

    Output
    - gc = GC% (a float between 0 and 1)
    """
    return ((seq.count('C') + seq.count('G')) / len(seq) * 100)/100

"""Q 1.3"""
def find_motif_freq(motif, seq):
    """
    Q:  given a target motif, find its frequency in a dna strand.
    Ex:
    >>> find_motif_freq("AA", "AAAAAAAAA")
    8
    >>> find_motif_freq("ATC", "ACTGACTATCGTCAGTCGATCTAATCCTG")
    3

    Input
    - motif: target substring
    - seq: sequence to search in

    Output
    - freq: frequency of the given motif
    """
    count = 0
    for i in range(len(seq)):
        if seq[i:].startswith(motif):
            count = count + 1
    return(count)

"""Q 1.4"""
def find_binding_site(bind, seq):
    """
    Q:  given a sequence, find the first position of the binding site.
        note, the binding site is the reverse complement of bind.
        hint: you can call the rev_complement() method earlier.
        hint: return -1 if the binding site is NOT found in seq
    Ex:
    >>> find_binding_site("ATGC", "ACTCGACTCAGCATCATACGGACTC")
    10

    Inputs
    - bind: the short binding-sequence that will bind to the sequence seq
    - seq: sequence to be bind to

    Outputs
    - pos: position (0 indexed)
    """
    rev = rev_complement(bind)
    return seq.find(rev)


if __name__ == '__main__':
    # print(rev_complement("ATGTACTAGCTA"))
    # print(gc_content("ATCGACTCGAGTCGTACGTTCACG"))
    # print(find_motif_freq("ATC", "ACTGACTATCGTCAGTCGATCTAATCCTG"))
    # print(find_binding_site("ATGC", "ACTCGACTCAGCATCATACGGACTC"))
    pass
