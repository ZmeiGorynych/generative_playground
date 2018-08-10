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
nonH_bond -> aromatic_ring
nonH_bond -> double_aromatic_ring
aromatic_ring -> 'c' num1 aromatic_atom aromatic_atom aromatic_atom aromatic_atom final_aromatic_atom_1
double_aromatic_ring -> 'c' num2 aromatic_atom aromatic_atom aromatic_atom 'c' num1 'n' num2 aromatic_atom aromatic_atom final_aromatic_atom_1
aromatic_atom -> 'n' 
aromatic_atom -> 'c' branch 
final_aromatic_atom_1 -> 'n' num1
final_aromatic_atom_1 -> 'c' num1 bond
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
    grammar_string_zinc_new += "num1 -> '" + str(i) + "'\n"
    grammar_string_zinc_new += "num2 -> '" + str(i) + "'\n"

for i in range(10,50):
    grammar_string_zinc_new += "num1 -> '%" + str(i) + "'\n"
    grammar_string_zinc_new += "num2 -> '%" + str(i) + "'\n"

grammar_string_zinc_new += "Nothing -> None\n"
