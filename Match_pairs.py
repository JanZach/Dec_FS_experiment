import pandas as pd
import numpy
import random
import math

Data = pd.read_csv('C:/Users/janmo/OneDrive/Dokumente/Goethe Uni/Doktor/Projekte/Decentralized Feature Selection 1/Otree/otree_test_output_3.csv')
Data = Data.loc[pd.isna(Data['participant._current_app_name'])==False, :]
Data['dec_fs_dictatorGame.1.player.kept_predicted'] = 80  # Only for test reasons, delete when instances are filled!

Data['group_id']    = 0
Data['Role']        = 0
Data['decision_made_by']        = "/"
Data['Payoff']      = 0

print(Data.head())
# Create list with unique participant IDs
lst_participant_codes = list(Data.loc[pd.isna(Data['participant._current_app_name'])==False , 'participant.code'])

#lst_participant_codes.append("test")

# randomly select entry of list without replacing
def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

#fill a list with pairs (as tuple); if list has uneven length, assign the remaining participant to variable

def rand_pairs(lst):
    pairs = []
    group_id = 1
    while len(lst) > 1:
        rand1 = pop_random(lst)
        rand2 = pop_random(lst)
        
        Data.loc[(Data['participant.code'] == rand1)|(Data['participant.code'] == rand2), "group_id"] = group_id
        group_id += 1
        
        pair = rand1, rand2
        pairs.append(pair)
        
    Data.loc[Data['participant.code'] == lst[0], "group_id"] = group_id
    
    return [pairs, lst]


def calc_payoffs(pairs, left_par = []):
    for pair in pairs:
        print(pair)
        dictator_ID = random.choices(pair)[0]
        for player in pair:
            if player == dictator_ID:
                Data.loc[Data['participant.code'] == player, 'Role'] = "Dictator"
            else:
                Data.loc[Data['participant.code'] == player, 'Role'] = "Recipient"
            
    if left_participant != []:
        print("left participant:", left_par)
        Data.loc[Data['participant.code'] == left_participant[0], 'Role'] = 'Dictator'
     
    # Calculate dictator payoff    
    Data.loc[Data['Role']=='Dictator', 'decision_made_by'] = Data.apply(lambda row: random.choices(["player", "machine"],
                                                                              weights=(100-row['dec_fs_dictatorGame.1.player.BDM'],
                                                                                       row['dec_fs_dictatorGame.1.player.BDM']))[0], axis=1)
    
    Data.loc[Data['decision_made_by']=='player', 'Payoff'] = Data['dec_fs_dictatorGame.1.player.kept'] + Data['dec_fs_dictatorGame.1.player.specialEndowment']
    Data.loc[Data['decision_made_by']=='machine', 'Payoff'] = Data['dec_fs_dictatorGame.1.player.kept_predicted'] + Data['dec_fs_dictatorGame.1.player.specialEndowment']                                                          
    
    # Calculate recipient payoff                                            
    n_groups = Data['group_id'].unique()[-1]
    for i in range(1, n_groups+1): 
        
        if len(Data.loc[Data['group_id']==i])==2:
        
            dictator_payoff         = Data.loc[(Data['Role']=='Dictator') & (Data['group_id']==i), 'Payoff'].iloc[0]
            dictator_specEndowment  = Data.loc[(Data['Role']=='Dictator') & (Data['group_id']==i), 'dec_fs_dictatorGame.1.player.specialEndowment'].iloc[0]
            recipient_specEndowment = Data.loc[(Data['Role']=='Recipient') & (Data['group_id']==i), 'dec_fs_dictatorGame.1.player.specialEndowment'].iloc[0]
            
            Data.loc[(Data['Role']=='Recipient') & (Data['group_id']==i), 'Payoff'] = 100 - (dictator_payoff - dictator_specEndowment) + recipient_specEndowment
    
          

# Execute functions
pairs, left_participant = rand_pairs(lst_participant_codes)
calc_payoffs(pairs, left_par = left_participant)













