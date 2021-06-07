"""Convert NCBI Gene Symbols to Entrez IDs"""

import argparse

##Copied from the answer to this issue https://github.com/tanghaibao/goatools/issues/144

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert Gene symbols to Entrez IDS")
    parser.add_argument("--gene_info", type=str, default='Homo_sapiens.gene_info', required=True,
                        help="Path to gene info file downloaded from NCBI")
    parser.add_argument("--study", type=str, default='study_symbols.txt', required=True,
                        help="Path to file containing study symbols")
    parser.add_argument("--pop", type=str, default='pop_symbols',
                        help="Path to file containing population symbols")
    parser.add_argument("--study_out", type=str, default='study_geneids.txt',
                        help="Output file to save the study gene ids")
    parser.add_argument("--pop_out", type=str, default='pop_geneids.txt',
                        help="Output file to save the pop gene ids")
    return parser.parse_args()


def main():
    """Convert NCBI Gene Symbols to Entrez IDs"""
    args = parse_arguments()
    fin_info = args.gene_info
    fin_stu = args.study
    fin_pop = args.pop
    fout_stu = args.study_out
    fout_pop = args.pop_out

    # 1. Read the gene files
    symbols_stu = read_symbols(fin_stu)
    symbols_pop = read_symbols(fin_pop)
    symbol2geneid = read_symbol2geneid(fin_info)

    # 2. Convert the lists of gene Symbols to Entrez GeneIDs
    geneids_stu = convert_symbol_to_geneid(symbols_stu, symbol2geneid)
    geneids_pop = convert_symbol_to_geneid(symbols_pop, symbol2geneid)
    print('  {N:6,} GeneIDs in study'.format(N=len(geneids_stu)))
    print('  {N:6,} GeneIDs in population'.format(N=len(geneids_pop)))

    # 3. Write geneids
    write_geneids(fout_stu, geneids_stu)
    write_geneids(fout_pop, geneids_pop)


def convert_symbol_to_geneid(symbols, symbol2geneid):
    """Convert gene symbols to geneids"""
    syms1 = set(symbol2geneid).intersection(symbols)
    # Uncomment to see which symbols were not found
    # print('NOT FOUND:', symbols.difference(syms1))
    return set(symbol2geneid[sym] for sym in syms1)

def write_geneids(fout, geneids):
    """Read a list of gene symbols and return a list"""
    with open(fout, 'w') as prt:
        for geneid in sorted(geneids):
            prt.write('{GeneID}\n'.format(GeneID=geneid))
        print('  {N:6,} WROTE: {FOUT}'.format(N=len(geneids), FOUT=fout))

def read_symbols(fin):
    """Read a list of gene symbols and return a list"""
    with open(fin) as ifstrm:
        symbols = set(line.strip() for line in ifstrm)
        print('  {N:6,} READ: {FIN}'.format(N=len(symbols), FIN=fin))
        return symbols

def read_symbol2geneid(fin):
    """Get gene Symbol-to-GeneID from reading gene_info file"""
    symbol2geneid = {}
    with open(fin) as ifstrm:
        for line in ifstrm:
            vals = line.split('\t')
            if vals[0] == '9606':
                symbol = vals[2]
                geneid = int(vals[1])
                synonyms = vals[4]
                # Add gene Symbol
                if symbol != '-':
                    symbol2geneid[symbol] = geneid
                # Add gene Symbol synonyms
                if synonyms != '-':
                    for syn in synonyms.split('|'):
                        if syn not in symbol2geneid:
                            symbol2geneid[syn] = geneid
                        else:
                            pass
                            # print('Synonym is a gene Symbol: {S} {SYM}'.format(S=syn, SYM=symbol))
        print('  {N:6,} READ: {FIN}'.format(N=len(symbol2geneid), FIN=fin))
    return symbol2geneid


if __name__ == '__main__':
    main()
