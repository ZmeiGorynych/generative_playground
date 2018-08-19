import copy

pre_grammar_string_zinc_new = """
smiles -> valence_1 bond
smiles -> valence_2 double_bond
smiles -> valence_3 triple_bond
bond -> 'h'
bond -> nonH_bond
branch -> 'h'
branch -> '(' nonH_bond ')'
nonH_bond -> valence_1
nonH_bond -> valence_2 bond
nonH_bond -> valence_3 double_bond
nonH_bond -> valence_4 triple_bond
double_bond -> valence_2
double_bond -> valence_3 bond
double_bond -> valence_4 double_bond
triple_bond -> valence_3 
triple_bond -> valence_4 bond
valence_4 -> 'C'
valence_4 -> '[' 'C' '@' ']'
valence_4 -> '[' 'C' '@' '@' ']'
valence_4 -> '[' 'N' '+' ']'
valence_3 -> '[' 'C' '@' 'H' ']'
valence_3 -> '[' 'C' '@' '@' 'H' ']'
valence_3 -> 'N'
valence_3 -> '[' 'N' 'H' '+' ']'
valence_3 -> valence_4 branch
valence_2 -> 'O'
valence_2 -> 'S'
valence_2 -> 'S' '(' '=' 'O' ')'  '(' '=' 'O' ')'
valence_2 -> valence_3 branch
valence_2 -> valence_4 '(' double_bond ')'
valence_1 -> 'F'
valence_1 -> 'Cl'
valence_1 -> 'Br'
valence_1 -> 'I'
valence_1 -> '[' 'O' '-' ']'
valence_1 -> '[' 'N' 'H' '3' '+' ']'
valence_1 -> valence_2  branch
valence_1 -> valence_3 '(' double_bond ')'
valence_1 -> valence_4 '(' triple_bond ')'
nonH_bond -> aliphatic_ring
aliphatic_ring -> valence_3_num1 cycle_bond
aliphatic_ring -> valence_4_num1 cycle_double_bond
cycle_bond -> valence_2 cycle_bond
cycle_bond -> valence_3 cycle_double_bond
cycle_double_bond -> valence_3 cycle_bond
cycle_bond -> valence_2_num1
cycle_double_bond -> valence_3_num1"""

def add_numbered_valence(grammar_str:str):
    my_str = grammar_str.split('\n')
    my_str_new = copy.copy(my_str)
    num_token = 'num1'
    num_valences = ['valence_2', 'valence_3', 'valence_4']
    insert_points = ["']'","'S'", "'O'", "'C'", "'N'"]
    for s in my_str:
        if str(s[:9]) in num_valences:
            for nv in num_valences:
                s = s.replace(nv, nv + '_' + num_token)

            s_lhs, s_rhs = s.split(' -> ')
            # now try inserting num_token after the first actual atom we find
            for ip in insert_points:
                s_rhs_new = s_rhs.replace(ip, ip + ' ' + num_token)
                if s_rhs != s_rhs_new:
                    break
                # if str(s_rhs[:7]) == "'S' '('":
                #     # special case
                #     s_rhs = s_rhs.replace("'S'","'S' " + num_token)
                # else:
                #     # insert a num1 token after the last terminal, if any
                #     insert_ind = s_rhs.rfind("'") + 1
                #     s_rhs = s_rhs[:insert_ind] + ' ' + num_token + (s_rhs[insert_ind:] if insert_ind < len(s_rhs) else '')

            my_str_new.append(s_lhs + ' -> ' + s_rhs_new)

    new_str = ''.join([s + '\n' for s in my_str_new])
    print(new_str)
    print('******************')
    return new_str

pre_grammar_string_zinc_new = add_numbered_valence(pre_grammar_string_zinc_new)

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

grammar_string_zinc_new = pre_grammar_string_zinc_new#purge_implicit_H(pre_grammar_string_zinc_new)