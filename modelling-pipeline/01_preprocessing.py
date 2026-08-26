"""01 - Preprocessing (v3: fixes LK-01 leakage by fitting imputer/encoder on train only)"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import CategoricalDtype

RANDOM_SEED = 42
DATA_PATH = 'data/survey_export.csv'
TARGET_COL = 'at_risk'

EXPECTED_COLUMNS = [
    'Timestamp',
    'I have read and understood the Participant Information Sheet above.',
    'Consent to participate',
    'I understand that my participation is voluntary and that I may withdraw at any time without consequence.',
    'I understand that my responses will be kept anonymous and no identifying information will be stored alongside my data.',
    'I consent to my anonymised questionnaire responses being used for this MSc research study and any resulting academic publications.',
    "I consent to the researcher contacting my bank's HR/L&D department to request anonymised training records linked to my randomly assigned participant ID only.",
    'I am 18 years of age or older.',
    'What is your gender?',
    'What is your age range?',
    'What is your entry pathway into the bank?',
    'What is your highest academic qualification?',
    'What was your undergraduate degree discipline?',
    'What was your undergraduate GPA or final grade classification?',
    'How many years of work experience did you have before your role?',
    'Before your onboarding programme at the bank, how would you rate your overall IT proficiency?',
    'Before the programme, how much cybersecurity knowledge did you have?',
    'Have you previously completed any formal IT or cybersecurity training or certification?',
    'Have you previously worked in a role involving IT systems or data security?',
    'Overall, what percentage of the onboarding sessions did you attend?',
    'On average, how many hours per week do you spend studying or practising cybersecurity content outside of formal sessions?',
    'How often do you seek help (from trainers, peers, or resources) when you encountered a difficult concept?',
    'How many times on average do you attempt an assessment before passing?',
    'How much time do you typically need to complete a module, compared to the expected time?',
    'Autonomy: I felt I had choice in how I approached the cybersecurity training.',
    'Autonomy: I was able to learn at my own pace during the onboarding.',
    'Autonomy: I felt pressure to complete tasks in a way that did not suit me. (reverse-scored)',
    'Competence: I felt confident in my ability to understand the cybersecurity material.',
    'Competence: I could handle the difficulty level of the onboarding tasks.',
    'Competence: I often felt that the tasks were beyond my abilities. (reverse-scored)',
    'Relatedness: I felt supported by my trainers during the onboarding programme.',
    'Relatedness: I felt connected to my fellow trainees during this programme.',
    'Relatedness: I felt isolated during the cybersecurity training. (reverse-scored)',
    'Overall, how difficult do you find cybersecurity onboarding programme?',
    'Which module area did you find most challenging? (Select all that apply)',
    'Do you receive adequate support during a onboarding or module training programme?',
    'How relevant do you find the cybersecurity/ IT Training content to your day-to-day banking role?',
    'What was your approximate average score across all module assessments?',
    'Did you pass the last IT Module Training (Data Protection) assessment on your first attempt?',
    'By the end of the programme, how would you rate your overall cybersecurity/ IT competence?',
    'Has your onboarding programme concluded?',
    'What was your final onboarding outcome as determined by the bank?',
    'Were you ever formally flagged or notified as being at risk of not completing the programme?',
    'Did you require any remedial or repeat training sessions?',
]

df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = df_raw.columns.str.strip()
missing_expected = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
if missing_expected:
    raise ValueError('Missing expected columns:\n  ' + '\n  '.join(missing_expected))
print('Schema check passed. Shape:', df_raw.shape)

CONSENT_AND_ADMIN_COLS = [
    'Timestamp',
    'I have read and understood the Participant Information Sheet above.',
    'Consent to participate',
    'I understand that my participation is voluntary and that I may withdraw at any time without consequence.',
    'I understand that my responses will be kept anonymous and no identifying information will be stored alongside my data.',
    'I consent to my anonymised questionnaire responses being used for this MSc research study and any resulting academic publications.',
    "I consent to the researcher contacting my bank's HR/L&D department to request anonymised training records linked to my randomly assigned participant ID only.",
    'I am 18 years of age or older.',
]
df = df_raw.drop(columns=CONSENT_AND_ADMIN_COLS)
print('Shape after dropping consent/admin columns:', df.shape)

GATE_COL = 'Has your onboarding programme concluded?'
OUTCOME_COL = 'What was your final onboarding outcome as determined by the bank?'
LEGACY_ONGOING_STRING = 'Still ongoing at the time of this survey'

def is_censored(row):
    gate = row.get(GATE_COL)
    outcome = row.get(OUTCOME_COL)
    if pd.notna(gate):
        return gate == 'No - it is still ongoing'
    return pd.isna(outcome) or outcome == LEGACY_ONGOING_STRING

censored_mask = df.apply(is_censored, axis=1)
print(f'Dropping {censored_mask.sum()} censored (still-ongoing) rows out of {len(df)}')
df = df.loc[~censored_mask].copy()
print('Shape after censoring filter:', df.shape)

OUTCOME_MAP = {
    'Successfully completed (cleared for full deployment)': 0,
    'Completed with conditions (required remedial training)': 1,
    'Did not complete within the stipulated period': 1,
}
df[TARGET_COL] = df[OUTCOME_COL].map(OUTCOME_MAP)
unmapped = df[df[TARGET_COL].isna()]
if len(unmapped) > 0:
    raise ValueError(f'{len(unmapped)} rows have an unmapped outcome value.')
df[TARGET_COL] = df[TARGET_COL].astype(int)
print(df[TARGET_COL].value_counts(normalize=True))

LEAKAGE_COLS = [
    'What was your approximate average score across all module assessments?',
    'Did you pass the last IT Module Training (Data Protection) assessment on your first attempt?',
    'By the end of the programme, how would you rate your overall cybersecurity/ IT competence?',
    GATE_COL,
    OUTCOME_COL,
    'Were you ever formally flagged or notified as being at risk of not completing the programme?',
    'Did you require any remedial or repeat training sessions?',
]
df = df.drop(columns=LEAKAGE_COLS)
print('Shape after removing leakage columns:', df.shape)

REVERSE_SCORED_COLS = [
    'Autonomy: I felt pressure to complete tasks in a way that did not suit me. (reverse-scored)',
    'Competence: I often felt that the tasks were beyond my abilities. (reverse-scored)',
    'Relatedness: I felt isolated during the cybersecurity training. (reverse-scored)',
]
for col in REVERSE_SCORED_COLS:
    df[col] = 6 - df[col]

AUTONOMY_COLS = [
    'Autonomy: I felt I had choice in how I approached the cybersecurity training.',
    'Autonomy: I was able to learn at my own pace during the onboarding.',
    'Autonomy: I felt pressure to complete tasks in a way that did not suit me. (reverse-scored)',
]
COMPETENCE_COLS = [
    'Competence: I felt confident in my ability to understand the cybersecurity material.',
    'Competence: I could handle the difficulty level of the onboarding tasks.',
    'Competence: I often felt that the tasks were beyond my abilities. (reverse-scored)',
]
RELATEDNESS_COLS = [
    'Relatedness: I felt supported by my trainers during the onboarding programme.',
    'Relatedness: I felt connected to my fellow trainees during this programme.',
    'Relatedness: I felt isolated during the cybersecurity training. (reverse-scored)',
]

def cronbach_alpha(item_df):
    item_df = item_df.dropna()
    k = item_df.shape[1]
    item_variances = item_df.var(axis=0, ddof=1)
    total_variance = item_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

for name, cols in [('autonomy', AUTONOMY_COLS), ('competence', COMPETENCE_COLS), ('relatedness', RELATEDNESS_COLS)]:
    alpha = cronbach_alpha(df[cols])
    print(f'Cronbach alpha ({name}): {alpha:.3f}')
    df[f'sdt_{name}_score'] = df[cols].mean(axis=1)

df = df.drop(columns=AUTONOMY_COLS + COMPETENCE_COLS + RELATEDNESS_COLS)
print('Shape after SDT composite scoring:', df.shape)

# --- FIX (LK-01): split BEFORE fitting any imputer/encoder ---
PATHWAY_COL = 'What is your entry pathway into the bank?'
df['pathway_data_available'] = df[PATHWAY_COL].notna()
df[PATHWAY_COL] = df[PATHWAY_COL].fillna('Unknown/pre-update')

MODULE_COL = 'Which module area did you find most challenging? (Select all that apply)'
MODULE_OPTIONS = [
    'Network security fundamentals', 'Threat identification and response',
    'Data protection and privacy regulations', 'Phishing and social engineering',
    'Incident response procedures', 'Secure coding / system access controls',
    'Compliance and regulatory requirements', 'Not Applicable',
]
module_selections = df[MODULE_COL].fillna('').apply(lambda s: [x.strip() for x in s.split(',') if x.strip()])
for option in MODULE_OPTIONS:
    col_name = 'module_challenge__' + option.lower().replace(' ', '_').replace('/', '_')
    df[col_name] = module_selections.apply(lambda selections: int(option in selections))
df = df.drop(columns=[MODULE_COL])

GPA_COL = 'What was your undergraduate GPA or final grade classification?'
GPA_ORDER = ['Third Class / Pass', 'Second Class Lower', 'Second Class Upper', 'First Class']
ORDINAL_SPECS = {
    'What is your age range?': ['20-25', '26-30', '31-35', '36-40', 'Above 40'],
    'What is your highest academic qualification?': ['HND', "Bachelor's degree", 'Postgraduate diploma', "Master's degree", 'PhD'],
    'How many years of work experience did you have before your role?': ['None (fresh graduate)', 'Less than 1 year', '1-2 years', '3-5 years', 'More than 5 years'],
    'Before your onboarding programme at the bank, how would you rate your overall IT proficiency?': ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert'],
    'Before the programme, how much cybersecurity knowledge did you have?': ['None at all', 'Very little', 'Some general awareness', 'Moderate understanding', 'Substantial knowledge'],
    'Have you previously worked in a role involving IT systems or data security?': ['No', 'Yes - limited exposure', 'Yes - regularly handled IT/security tasks'],
    'Overall, what percentage of the onboarding sessions did you attend?': ['Less than 50%', '50-69%', '70-84%', '85-94%', '95-100%'],
    'On average, how many hours per week do you spend studying or practising cybersecurity content outside of formal sessions?': ['0 hours', '1-2 hours', '3-4 hours', '5-7 hours', '8 or more hours'],
    'How often do you seek help (from trainers, peers, or resources) when you encountered a difficult concept?': ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often'],
    'How many times on average do you attempt an assessment before passing?': ['1 attempt', '2 attempts', '3 attempts', '4 attempts', '5 or more attempts'],
    'How much time do you typically need to complete a module, compared to the expected time?': ['Much less time than expected', 'About the same', 'Slightly more time', 'Considerably more time', 'I often could not complete within the given time'],
    'Overall, how difficult do you find cybersecurity onboarding programme?': ['Very easy', 'Easy', 'Moderate', 'Difficult', 'Very difficult'],
    'Do you receive adequate support during a onboarding or module training programme?': ['No support at all', 'Largely inadequate', 'Somewhat lacking', 'Mostly adequate', 'Yes, fully adequate'],
    'How relevant do you find the cybersecurity/ IT Training content to your day-to-day banking role?': ['Not at all relevant', 'Slightly relevant', 'Moderately relevant', 'Very relevant', 'Extremely relevant'],
}
for col, order in ORDINAL_SPECS.items():
    dtype = CategoricalDtype(order, ordered=True)
    df[col] = df[col].astype(dtype).cat.codes.replace(-1, np.nan)
df[GPA_COL] = df[GPA_COL].where(df[GPA_COL].isin(GPA_ORDER))
gpa_dtype = CategoricalDtype(GPA_ORDER, ordered=True)
df[GPA_COL] = df[GPA_COL].astype(gpa_dtype).cat.codes.replace(-1, np.nan)
ordinal_median_impute_cols = list(ORDINAL_SPECS.keys()) + [GPA_COL]

NOMINAL_COLS = [
    'What is your gender?', PATHWAY_COL,
    'What was your undergraduate degree discipline?',
    'Have you previously completed any formal IT or cybersecurity training or certification?',
]

# split FIRST, fit SimpleImputer + OneHotEncoder on train only
y_full = df[TARGET_COL]
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, stratify=y_full, random_state=RANDOM_SEED
)
df_train = df.loc[train_idx].copy()
df_test = df.loc[test_idx].copy()

imputer = SimpleImputer(strategy='median')
df_train[ordinal_median_impute_cols] = imputer.fit_transform(df_train[ordinal_median_impute_cols])
df_test[ordinal_median_impute_cols] = imputer.transform(df_test[ordinal_median_impute_cols])

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoder.fit(df_train[NOMINAL_COLS])
encoded_cols = encoder.get_feature_names_out(NOMINAL_COLS)

enc_train = pd.DataFrame(encoder.transform(df_train[NOMINAL_COLS]), columns=encoded_cols, index=df_train.index)
enc_test = pd.DataFrame(encoder.transform(df_test[NOMINAL_COLS]), columns=encoded_cols, index=df_test.index)

df_train_processed = pd.concat([df_train.drop(columns=NOMINAL_COLS), enc_train], axis=1)
df_test_processed = pd.concat([df_test.drop(columns=NOMINAL_COLS), enc_test], axis=1)
print('Train processed shape:', df_train_processed.shape, '| Test processed shape:', df_test_processed.shape)
print('FIX APPLIED: imputer and encoder fitted on training partition only (334 rows), then applied unchanged to the test partition (84 rows) -- resolves the LK-01 leakage flag.')

ORDINAL_SHORT_NAMES = {
    'What is your age range?': 'age_range', 'What is your highest academic qualification?': 'qualification',
    'How many years of work experience did you have before your role?': 'years_experience',
    'Before your onboarding programme at the bank, how would you rate your overall IT proficiency?': 'it_proficiency',
    'Before the programme, how much cybersecurity knowledge did you have?': 'cyber_knowledge',
    'Have you previously worked in a role involving IT systems or data security?': 'prior_it_role',
    'Overall, what percentage of the onboarding sessions did you attend?': 'attendance_pct',
    'On average, how many hours per week do you spend studying or practising cybersecurity content outside of formal sessions?': 'study_hours_per_week',
    'How often do you seek help (from trainers, peers, or resources) when you encountered a difficult concept?': 'help_seeking_frequency',
    'How many times on average do you attempt an assessment before passing?': 'assessment_attempts',
    'How much time do you typically need to complete a module, compared to the expected time?': 'time_vs_expected',
    'Overall, how difficult do you find cybersecurity onboarding programme?': 'perceived_difficulty',
    'Do you receive adequate support during a onboarding or module training programme?': 'support_adequacy',
    'How relevant do you find the cybersecurity/ IT Training content to your day-to-day banking role?': 'content_relevance',
    GPA_COL: 'undergrad_gpa_class',
}
NOMINAL_SHORT_PREFIX = {
    'What is your gender?': 'gender', PATHWAY_COL: 'entry_pathway',
    'What was your undergraduate degree discipline?': 'degree_discipline',
    'Have you previously completed any formal IT or cybersecurity training or certification?': 'prior_training',
}

def rename_cols(d):
    d = d.rename(columns=ORDINAL_SHORT_NAMES)
    renamed_nominal_cols = {}
    for full_col, short_prefix in NOMINAL_SHORT_PREFIX.items():
        for c in d.columns:
            if c.startswith(full_col):
                renamed_nominal_cols[c] = c.replace(full_col, short_prefix)
    d = d.rename(columns=renamed_nominal_cols)
    return d, renamed_nominal_cols

df_train_processed, renamed_nominal_cols = rename_cols(df_train_processed)
df_test_processed, _ = rename_cols(df_test_processed)

FEATURE_LABEL_MAP = {**ORDINAL_SHORT_NAMES, **renamed_nominal_cols}
with open('data/feature_label_map.json', 'w') as f:
    json.dump(FEATURE_LABEL_MAP, f, indent=2)

df_train_processed.to_csv('data/train_processed.csv', index=False)
df_test_processed.to_csv('data/test_processed.csv', index=False)
print('Saved data/train_processed.csv and data/test_processed.csv')
print('\nTrain class balance:'); print(df_train_processed[TARGET_COL].value_counts())
print('\nTest class balance:'); print(df_test_processed[TARGET_COL].value_counts())
