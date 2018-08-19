import copy

pre_grammar_string_zinc_new="""smiles ->  C
smiles -> nonH_bond
smiles -> N
smiles -> O
smiles -> aromatic_ring
C -> 'C' branch branch bond
C -> '[' 'C' chirality 'H' ']' branch bond
chirality -> '@'
chirality -> '@' '@'
C -> 'C' '(' double_bond ')' bond
C -> 'C' triple_bond
N -> 'N' branch bond
N -> 'N' double_bond
N -> 'N' triple_bond
O -> 'O' bond
O -> 'O' double_bond
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
nonH_bond -> C
nonH_bond -> 'F'
nonH_bond -> 'Cl'
nonH_bond -> 'Br'
nonH_bond -> 'I'
nonH_bond -> 'O' bond
nonH_bond -> '[' 'O' '-' ']'
nonH_bond -> 'N' double_bond
nonH_bond -> 'N' branch bond
nonH_bond -> '[' 'N' 'H' '+' ']' branch bond
nonH_bond -> 'S' bond
nonH_bond -> 'S' '(' '=' 'O' ')'  '(' '=' 'O' ')' bond
nonH_bond -> '[' 'N' 'H' '3' '+' ']'
nonH_bond -> aromatic_ring
nonH_bond -> double_aromatic_ring
aromatic_ring -> aromatic_starting_c aromatic_atom aromatic_atom aromatic_atom aromatic_atom final_aromatic_atom_1
aromatic_ring -> aromatic_starting_c aromatic_atom aromatic_atom aromatic_atom aromatic_atom final_aromatic_atom_1
aromatic_ring -> aromatic_starting_c aromatic_os aromatic_atom aromatic_atom final_aromatic_atom_1
aromatic_ring -> aromatic_starting_c aromatic_atom aromatic_os aromatic_atom final_aromatic_atom_1
aromatic_ring -> aromatic_starting_c aromatic_atom aromatic_atom aromatic_os final_aromatic_atom_1
aromatic_ring -> aromatic_starting_c aromatic_atom aromatic_atom aromatic_atom final_aromatic_os
aromatic_starting_c -> '-' 'c' num1 
aromatic_starting_c -> 'c' num1
double_aromatic_ring -> 'c' num2 aromatic_atom aromatic_atom aromatic_atom 'c' num1 'n' num2 aromatic_atom aromatic_atom final_aromatic_atom_1
aromatic_atom -> 'n' 
aromatic_atom -> 'c' branch 
aromatic_os -> 'o'
aromatic_os -> 's'
aromatic_os -> 'n' branch
aromatic_os -> '[' 'n' 'H' ']'
final_aromatic_atom_1 -> 'n' num1
final_aromatic_atom_1 -> 'c' num1 bond
final_aromatic_os -> 'o' num1
final_aromatic_os -> 's' num1
final_aromatic_os -> 'n' num1 bond
nonH_bond -> aliphatic_ring
aliphatic_ring -> 'N' num1 cycle_bond
aliphatic_ring -> 'C' num1 cycle_double_bond
aliphatic_ring -> 'C' num1 branch cycle_bond
cycle_bond -> 'N' branch cycle_bond
cycle_bond -> 'N' cycle_double_bond
cycle_bond -> 'C' branch branch cycle_bond
cycle_bond -> 'C' '(' double_bond ')' cycle_bond
cycle_bond -> 'C' branch cycle_double_bond
cycle_bond -> 'O' cycle_bond
cycle_bond -> 'S' cycle_bond
cycle_double_bond -> 'N' cycle_bond
cycle_double_bond -> 'C' branch cycle_bond
cycle_bond -> 'N' num1 bond
cycle_bond -> 'C' num1 branch bond
cycle_bond -> 'C' num1 double_bond
cycle_bond -> 'O' num1
cycle_bond -> 'S' num1
"""

# nonH_bond -> plain_aromatic_ring
# add rules for generating ring numerals
for i in range(1,10):
    pre_grammar_string_zinc_new += "num1 -> '" + str(i) + "'\n"
    pre_grammar_string_zinc_new += "num2 -> '" + str(i) + "'\n"

for i in range(10,15):#50):
    pre_grammar_string_zinc_new += "num1 -> '%" + str(i) + "'\n"
    pre_grammar_string_zinc_new += "num2 -> '%" + str(i) + "'\n"

pre_grammar_string_zinc_new += "Nothing -> None\n"

def purge_implicit_H(x: str):
    old_str = x.split('\n')
    new_str = []
    while True:
        for s in old_str:
            s = s.replace('_bond', '_BOND')
            if str(s[:4]) != 'bond' and str(s[:6]) != 'branch':
                if 'branch' in s:
                    new_str.append(s.replace('branch', '', 1))
                    new_str.append(s.replace('branch', "'(' nonH_bond ')'", 1))
                elif 'bond' in s:
                    new_str.append(s.replace('bond', '', 1))
                    new_str.append(s.replace('bond', 'nonH_bond', 1))
                else:
                    new_str.append(s)
        if len(old_str) == len(new_str):
            break
        else:
            old_str = new_str
            new_str = []

    new_str = ''.join([s.replace('_BOND', '_bond') + '\n' for s in new_str])
    print(new_str)
    return new_str

grammar_string_zinc_new__ = purge_implicit_H(pre_grammar_string_zinc_new)