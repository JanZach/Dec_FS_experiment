

from otree.api import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import shap
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier
import random
import statistics
import pickle


doc = """
One player decides how to divide a certain amount between himself and the other
player.
See: Kahneman, Daniel, Jack L. Knetsch, and Richard H. Thaler. "Fairness
and the assumptions of economics." Journal of business (1986):
S285-S300.
"""

# Initialize the base variables
class C(BaseConstants):
    NAME_IN_URL = 'dictator'
    PLAYERS_PER_GROUP = None     # No groups as we infer the data from each participant
    NUM_ROUNDS = 1
    INSTRUCTIONS_TEMPLATE = 'dictator_experiment_test/instructions.html'
    # Initial amount allocated to the dictator
    ENDOWMENT = cu(100)         # Endowment may be adapted
    #DICTATOR_ROLE = 'Dictator'
    #RECIPIENT_ROLE = 'Recipient'


class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    """
     kept = models.CurrencyField()
    """
class Player(BasePlayer):
    treatment = models.StringField()                          # Is participant in treatment 0 or 1 ?

    dictator_prediction_full = models.IntegerField()           # Prediction based on all questionnaire attributes
    dictator_prediction_dec_fs = models.IntegerField()         # Prediction  based on selected questionnaire attributes

    kept = models.CurrencyField(                               # How much monetary value does the dictator keep?
        doc="""Amount dictator decided to keep for himself""",
        choices=[[100, "100 Punkte"], [70, "70 Punkte"]],
        label="Ich möchte den folgenden Betrag für mich behalten:", blank=False
    )

    kept_predicted = models.CurrencyField()

    # Participants' descriptives
    # Alter
    age = models.IntegerField(label="Ihr Alter:", min=18, max=99, blank=False)
    # Geschlecht
    sex = models.IntegerField(choices=[[1, "Weiblich"], [2, "Männlich"]],
                              widget=widgets.RadioSelect, label="Ihr Geschlecht:",
                              blank=False)
    # German born
    germborn = models.IntegerField(choices=[[0, 'Nicht in Deutschland geboren'], [1, 'In Deutschland geboren']], label="Sind Sie in Deutschland geboren?",
                                    widget=widgets.RadioSelect)
    # Monatliches Brutto-Einkommen
    income = models.IntegerField(choices=[[0, '< 1000 €'],[1, '1001 - 1500 €'],[2, '1501 - 2000 €'],
                                          [3, '2001 - 2500 €'],[4, '2501 - 3000 €'],[5, '3001 - 3500 €'],
                                          [6, '3501 - 4000 €'],[7, '4001 - 4500 €'],[8, '4501 - 5000 €'],
                                          [9, '> 5000 €']], label="Ihr monatliches Bruttoeinkommen in Euro:", blank=False)
    # Ausbildungs- und Trainingsjahre (Schule + weiterführende Ausbildung wie Uni, Berufsschule etc.)
    education = models.IntegerField(label="Die Gesamtdauer Ihrer Ausbildung in Jahre (Beginn ab der 1. Schulklasse)", blank=False)

    ### Likert 10 ###

    # Zufriedenheit mit Freizeit (bgp0108)
    sat_leisureTime = models.IntegerField(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="Ihrer Freizeit?", widget=widgets.RadioSelectHorizontal, blank=True)
    # Zufriedenheit mit Haushaltseinkommen (erst mal weglassen) (bgp0105)
    # ...
    # Zufriedenheit mit Personal Income (bgp0106)
    sat_persIncome = models.IntegerField(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="Ihrem persönlichen Einkommen?", widget=widgets.RadioSelectHorizontal, blank=True)
    # Zufriedenheit mit Social Life (bgp0111)
    sat_socialLife = models.IntegerField(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="Ihren Freunden und Bekannten?", widget=widgets.RadioSelectHorizontal, blank=True)
    # Zufriedenheit mit Demokratie (bgp0112)
    sat_democracy = models.IntegerField(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="der Demokratie, wie sie in Deutschland derzeit besteht?", widget=widgets.RadioSelectHorizontal, blank=True)

    ### Likert other ###
    # Risikobereitschaft (bgp05)
    will_risk = models.IntegerField(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="Ihre Risikobereitschaft:",
                                    widget=widgets.RadioSelectHorizontal, blank=True)
    # Wichtigkeit Glaube (bgp0610)
    importance_religion = models.IntegerField(choices=[[1,"Sehr wichtig"], [2, "Wichtig"], [3, "Weniger wichtig"], [4, "Ganz unwichtig"]],
                                    label="Wichtigkeit der Religion:",
                                    widget=widgets.RadioSelectHorizontal, blank=True)
    # Überprüfung Kontostand (bgp0702)
    check_account = models.IntegerField(choices=[[1, "Täglich"], [2, "Mindestens 1 mal pro Woche"], [3, "Mindestens 1 mal pro Monat"], [4, "Seltener"], [5, "Nie"]],
                                              label="Häufigkeit der Überprüfung des Kontostands:",
                                              widget=widgets.RadioSelectHorizontal, blank=True)
    # Wie oft Alkohol (bgp115)
    alcohol = models.IntegerField(choices=[[1, "Täglich"], [2, "An vier bis sechs Tagen in der Woche"], [3, "An zwei bis drei Tagen in der Woche"],
                                           [4, "An zwei bis vier Tagen im Monat"], [5, "Einmal im Monat oder seltener"],
                                           [6, "Nie"]], label="Häufigkeit des Verzehrs alkoholischer Getränke",
                                              widget=widgets.RadioSelect, blank=True)

    # - - - - - - - - - - - - - - - - - - - - - -

    # Dummy variables
    age_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Alter", widget=widgets.RadioSelect)
    sex_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Geschlecht", widget=widgets.RadioSelect)
    germborn_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="In Deutschland geboren", widget=widgets.RadioSelect)
    income_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Monats-Bruttoeinkommen", widget=widgets.RadioSelect)
    education_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Ausbildungs-Jahre", widget=widgets.RadioSelect)
    sat_leisureTime_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Zufriedenheit Freizeit", widget=widgets.RadioSelect)
    sat_persIncome_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Zufriedenheit persönliches Einkommen", widget=widgets.RadioSelect)
    sat_socialLife_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'],[1, 'Zurückhalten']], label="Zufriedenheit mit Freunden und Bekannten", widget=widgets.RadioSelect)
    sat_democracy_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'], [1, 'Zurückhalten']],label="Zufriedenheit mit Demokratie",widget=widgets.RadioSelect)
    will_risk_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'], [1, 'Zurückhalten']],label="Risikobereitschaft", widget=widgets.RadioSelect)
    importance_religion_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'], [1, 'Zurückhalten']],label="Wichtigkeit der Religion / des Glaubens", widget=widgets.RadioSelect)
    check_account_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'], [1, 'Zurückhalten']],label="Prüfung des Kontos",widget=widgets.RadioSelect)
    alcohol_dummy = models.IntegerField(choices=[[0, 'Bereitstellen'], [1, 'Zurückhalten']],label="Verzehr alkoholischer Getränke", widget=widgets.RadioSelect)


    ### ---------- ML model beliefs
    # Weitere measures überlegen
    AI_competence   = models.IntegerField(choices=list(range(1,8)), label = "Die KI ist kompetent und effektiv bei der Vorhersage Ihrer Entscheidung",widget=widgets.RadioSelect)
    AI_prejudice    = models.IntegerField(choices=list(range(1, 8)), label = "Die KI gibt eine unvoreingenommene Bewertung ab",widget=widgets.RadioSelect)
    AI_acc          = models.IntegerField(min=0, max=100, label = "Auf einer Skala von 0 bis 100%: Für wie präzise halten Sie die Vorhersage der KI?")

    ### Willingness to pay
    BDM = models.IntegerField(
        min=20, max=80, label="Please adjust the probability that the model overwrites your decision"
    )

    # Special endowment for willingness-to-pay measure
    specialEndowment = models.CurrencyField()

# FUNCTIONS
def creating_session(subsession): # Assigns the experimental groups; itertools.cycle ensures that we have 50/50 distribution of treatment groups
    import itertools
    treatments = itertools.cycle(['baseline', 'treatment_unaware', 'treatment_aware'])
    x = 1
    for player in subsession.get_players():
        player.treatment=next(treatments)
        print('player', x, 'is in condition:', player.treatment)
        x += 1
    subsession.group_randomly()

"""
def set_payoffs(group: Group):
    dictator = group.get_player_by_role(C.DICTATOR_ROLE)
    recipient = group.get_player_by_role(C.RECIPIENT_ROLE)

    dictator.payoff = group.kept + dictator.specialEndowment
    recipient.payoff = 100-group.kept + recipient.specialEndowment
"""

ml_model = pickle.load(open('altruism_prediction_model_bins.sav','rb'))


def predict_fairness_full(player: Player):  # Predict the fairness/reciprocity of dictator based on all attributes

    # Initialize each dummy with 0; extent the number and names of dummies according to the used features

    player.age_dummy = 0
    player.sex_dummy = 0
    player.germborn_dummy = 0
    player.income_dummy = 0
    player.education_dummy = 0
    player.sat_leisureTime_dummy = 0
    player.sat_persIncome_dummy = 0
    player.sat_socialLife_dummy = 0
    player.sat_democracy_dummy = 0
    player.will_risk_dummy = 0
    player.importance_religion_dummy = 0
    player.check_account_dummy = 0
    player.alcohol_dummy = 0

    # Create the input for the ML model; consists of (1) questionnaire attr. and (2) dummies=0
    input_obs_dict = pd.DataFrame({"age": player.age,
                                   "education": player.education,
                                   "sex": player.sex,
                                   "germborn": player.germborn,
                                   "sat_leisureTime": player.sat_leisureTime,
                                   "sat_persIncome": player.sat_persIncome,
                                   "sat_democracy": player.sat_democracy,
                                   "will_risk": player.will_risk,
                                   "sat_socialLife": player.sat_socialLife,
                                   "importance_religion": player.importance_religion,
                                   "check_account": player.check_account,
                                   "alcohol": player.alcohol,
                                   "income": player.income,
                                   # ----- Dummies --------
                                   "age_dummy": player.age_dummy,
                                   "education_dummy": player.education_dummy,
                                   "sex_dummy": player.sex_dummy,
                                   "germborn_dummy": player.germborn_dummy,
                                   "sat_leisureTime_dummy": player.sat_leisureTime_dummy,
                                   "sat_persIncome_dummy": player.sat_persIncome_dummy,
                                   "sat_democracy_dummy": player.sat_democracy_dummy,
                                   "will_risk_dummy": player.will_risk_dummy,
                                   "sat_socialLife_dummy": player.sat_socialLife_dummy,
                                   "importance_religion_dummy": player.importance_religion_dummy,
                                   "check_account_dummy": player.check_account_dummy,
                                   "alcohol_dummy": player.alcohol_dummy,
                                   "income_dummy": player.income_dummy
                                   },
                                  index=[0])

    input_obs = pd.DataFrame(input_obs_dict)  # Convert dict to DataFrame (only 1 row since we look at each single participant)

    player.dictator_prediction_full = int(ml_model.predict(input_obs))  # Perform the prediction


    if player.dictator_prediction_full == 0:
        player.kept_predicted = 100
    elif player.dictator_prediction_full == 1:
        player.kept_predicted = 70


def predict_fairness_dec_fs(player: Player):  # Predict the fairness/reciprocity of dictator after dec. FS

    # Define the value of the features for the model; either the real feature value (disclose) or the training set
    # median (withhold)

    # https://stackoverflow.com/questions/51747961/how-to-use-2-index-variable-in-a-single-for-loop-in-python

    dummies = [player.age_dummy, player.sex_dummy, player.germborn_dummy, player.income_dummy, player.education_dummy,
               player.sat_leisureTime_dummy, player.sat_persIncome_dummy, player.sat_socialLife_dummy,
               player.sat_democracy_dummy,
               player.will_risk_dummy, player.importance_religion_dummy, player.check_account_dummy,
               player.alcohol_dummy]

    player_vars = [player.age, player.sex, player.germborn, player.income, player.education,
                   player.sat_leisureTime, player.sat_persIncome, player.sat_socialLife, player.sat_democracy,
                   player.will_risk, player.importance_religion, player.check_account, player.alcohol]

    participant = player.participant

    par_fields = ["age", "sex", "germborn", "income", "education",
                          "sat_leisureTime", "sat_persIncome", "sat_socialLife", "sat_democracy",
                          "will_risk", "importance_religion", "check_account", "alcohol"]

    dict_medians = {'age': 45.0, 'bgbilzeit': 11.5, 'labgro16': 2200.0,
                    'sex': 0.0, 'germborn': 1.0, 'bgp0108': 8.0,
                    'bgp0106': 7.0, 'bgp0112': 6.0, 'bgp05': 5.0,
                    'bgp0111': 8.0, 'bgp0610': 2.0, 'bgp0702': 3.0, 'bgp115': 5.0}

    for dummy, var, key, par_field in zip(dummies, player_vars, list(dict_medians.keys()), par_fields):

        if dummy == 1:
            participant.vars[par_field] = dict_medians[key]
        else:
            participant.vars[par_field] = var

    input_obs_dict = pd.DataFrame({"age": participant.age,
                                   "education": participant.education,
                                   "sex": participant.sex,
                                   "germborn": participant.germborn,
                                   "sat_leisureTime": participant.sat_leisureTime,
                                   "sat_persIncome": participant.sat_persIncome,
                                   "sat_democracy": participant.sat_democracy,
                                   "will_risk": participant.will_risk,
                                   "sat_socialLife": participant.sat_socialLife,
                                   "importance_religion": participant.importance_religion,
                                   "check_account": participant.check_account,
                                   "alcohol": participant.alcohol,
                                   "income": participant.income,
                                   # ----- Dummies --------
                                   "age_dummy": player.age_dummy,
                                   "education_dummy": player.education_dummy,
                                   "sex_dummy": player.sex_dummy,
                                   "germborn_dummy": player.germborn_dummy,
                                   "sat_leisureTime_dummy": player.sat_leisureTime_dummy,
                                   "sat_persIncome_dummy": player.sat_persIncome_dummy,
                                   "sat_democracy_dummy": player.sat_democracy_dummy,
                                   "will_risk_dummy": player.will_risk_dummy,
                                   "sat_socialLife_dummy": player.sat_socialLife_dummy,
                                   "importance_religion_dummy": player.importance_religion_dummy,
                                   "check_account_dummy": player.check_account_dummy,
                                   "alcohol_dummy": player.alcohol_dummy,
                                   "income_dummy": player.income_dummy
                                   },
                                  index=[0])

    input_obs = pd.DataFrame(input_obs_dict)
    player.dictator_prediction_dec_fs = int(ml_model.predict(input_obs))

    if player.dictator_prediction_dec_fs == 0:
        player.kept_predicted = 100
    elif player.dictator_prediction_dec_fs == 1:
        player.kept_predicted = 70

def calculate_specialEndowment(player: Player):
    player.specialEndowment = 12 - (abs(player.BDM - 50))/2.5

"""
def set_group_decision(group: Group):
    dictator = group.get_player_by_role(C.DICTATOR_ROLE)

    group.kept = random.choices([dictator.kept, dictator.kept_predicted],
                                weights=(1-dictator.BDM, dictator.BDM))[0]
"""


# PAGES
class Introduction(Page):
    pass

class questionnaire(Page):
    form_model = 'player'

    form_fields = ["age", "sex", "germborn", "income", "education",
                   "sat_leisureTime", "sat_persIncome", "sat_socialLife", "sat_democracy",
                   "will_risk", "importance_religion", "check_account", "alcohol"]

    @staticmethod
    def vars_for_template(player: Player):
        descriptives = ["age", "sex", "germborn", "income", "education"]
        Likert_vars_10 = ["sat_leisureTime", "sat_persIncome", "sat_socialLife", "sat_democracy"]
        # will_risk: 0 (gar nicht risikobereit) - 10 (sehr risikobereit)
        # importance_religion: 1 (Sehr wichtig), 2 (wichtig), 3 (weniger wichtig), 4 (ganz unwichtig)
        # check_account: [1] Täglich	[2] Mindestens 1 Mal pro Woche	[3] Mindestens 1 Mal pro Monat	[4] Seltener	[5] Nie
        # alcohol: [1] Täglich [2] An vier bis sechs Tagen in der Woche	[3] An zwei bis drei Tagen in der Woche	[4] An zwei bis vier Tagen im Monat	[5] Einmal im Monat oder seltener [6] Nie
        return dict(descriptives=descriptives,
                    Likert_vars_10=Likert_vars_10)

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        predict_fairness_full(player)


class Explanation_of_DC(Page):
    pass

class dec_fs(Page):
    form_model = 'player'
    form_fields = ["age_dummy", "sex_dummy", "germborn_dummy", "income_dummy", "education_dummy",
               "sat_leisureTime_dummy", "sat_persIncome_dummy", "sat_socialLife_dummy", "sat_democracy_dummy",
               "will_risk_dummy", "importance_religion_dummy", "check_account_dummy", "alcohol_dummy"] #todo: Bei neuen features anpassen!

    @staticmethod
    def is_displayed(player: Player):
        return (player.treatment == 'treatment_unaware') | (player.treatment == 'treatment_aware')

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        predict_fairness_dec_fs(player)


class Offer(Page):
    form_model = 'player'
    form_fields = ['kept']

class Results(Page):
    pass

class Introduction_of_algorithm(Page):
    pass

class Elicitation_of_model_beliefs(Page):
    form_model="player"
    form_fields= ["AI_competence", "AI_prejudice", "AI_acc"]

class BDM(Page):
    form_model="player"
    form_fields= ["BDM"]

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        calculate_specialEndowment(player)
        #participant = player.participant
        #participant.specialEndowment = player.specialEndowment

"""
class Final_payoffs_wait_page(WaitPage):

    @staticmethod
    def after_all_players_arrive(group: Group):
        set_group_decision(group)
        set_payoffs(group)

class Final_payoffs(Page):
    @staticmethod
    def vars_for_template(player: Player):
        return dict(payoff = player.payoff)
"""

page_sequence = [Introduction,
                 questionnaire,
                 Explanation_of_DC,
                 Offer,
                 Introduction_of_algorithm,
                 dec_fs,
                 Elicitation_of_model_beliefs,
                 BDM,
                 Results,
                 ]
