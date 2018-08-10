grammar_string_zinc_new="""smiles -> branched_C
branched_C -> 'C' branch branch bond
branched_C -> 'C' '(' double_bond ')' bond
branched_C -> 'C' triple_bond
double_bond -> '=' 'O'
double_bond -> '=' 'S'
double_bond -> '=' 'N' bond
double_bond -> '=' 'C' branch bond
double_bond -> '=' 'C' double_bond
triple_bond -> '#' 'N'
triple_bond -> '#' 'C' bond
bond -> 'h'
bond -> nonH_bond
branch -> 'h'
branch -> '(' nonH_bond ')'
nonH_bond -> branched_C
nonH_bond -> 'F'
nonH_bond -> 'Cl'
nonH_bond -> 'Br'
nonH_bond -> 'I'
nonH_bond -> 'O' bond
nonH_bond -> 'N' double_bond
nonH_bond -> 'N' branch bond
nonH_bond -> 'S' bond
nonH_bond -> 'S' '(' '=' 'O' ')'  '(' '=' 'O' ')' bond
"""

plain_template = "plain_aromatic_ring_* -> 'c' '*' 'c' branch 'c' branch 'c' branch 'c' branch 'c' '*' branch\n\
                    nonH_bond -> plain_aromatic_ring_*\n"


for i in range(1,10):
    grammar_string_zinc_new += plain_template.replace('*', str(i))

grammar_string_zinc_new += "Nothing -> None\n"


grammar_string_zinc_new_ = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
aromatic_organic -> 's'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None"""