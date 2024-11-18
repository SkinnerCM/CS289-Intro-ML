'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

def freq_plus_feature(text, freq):
    return text.count('+')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_03_feature(text, freq):
    return float(freq['03'])
def freq_now_feature(text, freq):
    return float(freq['now'])
def freq_tr_feature(text, freq):
    return float(freq['tr'])
def freq_l_feature(text, freq):
    return float(freq['l'])
def freq_02_feature(text, freq):
    return float(freq['02'])
def freq_professional_feature(text, freq):
    return float(freq['professional'])
def freq_enron_feature(text, freq):
    return float(freq['enron'])
def freq_nbsp_feature(text, freq):
    return float(freq['nbsp'])
def freq_pills_feature(text, freq):
    return float(freq['pills'])

def freq_d_feature(text, freq):
    return float(freq['d'])
def freq_effective_feature(text, freq):
    return float(freq['effective'])
def freq_pt_feature(text, freq):
    return float(freq['pt'])
def freq_hou_feature(text, freq):
    return float(freq['hou'])
def freq_2004_feature(text, freq):
    return float(freq['2004'])
def freq_international_feature(text, freq):
    return float(freq['international'])
def freq_04_feature(text, freq):
    return float(freq['04'])
def freq_100_feature(text, freq):
    return float(freq['100'])
def freq_mmbtu_feature(text, freq):
    return float(freq['mmbtu'])
def freq_future_feature(text, freq):
    return float(freq['future'])
def freq_subject_feature(text, freq):
    return float(freq['subject'])
def freq_u_feature(text, freq):
    return float(freq['u'])
def freq_day_feature(text, freq):
    return float(freq['day'])
def freq_20_feature(text, freq):
    return float(freq['20'])
def freq_more_feature(text, freq):
    return float(freq['more'])
def freq_change_feature(text, freq):
    return float(freq['change'])
def freq_free_feature(text, freq):
    return float(freq['free'])
def freq_today_feature(text, freq):
    return float(freq['today'])
def freq_11_feature(text, freq):
    return float(freq['11'])
def freq_statements_feature(text, freq):
    return float(freq['statements'])
def freq_software_feature(text, freq):
    return float(freq['software'])
def freq_without_feature(text, freq):
    return float(freq['without'])
def freq_over_feature(text, freq):
    return float(freq['over'])
def freq_office_feature(text, freq):
    return float(freq['office'])

def freq_daily_feature(text, freq):
    return float(freq['daily'])
def freq_o_feature(text, freq):
    return float(freq['o'])
def freq_texas_feature(text, freq):
    return float(freq['texas'])
def freq_69_feature(text, freq):
    return float(freq['69'])
def freq_april_feature(text, freq):
    return float(freq['april'])
def freq_products_feature(text, freq):
    return float(freq['products'])
def freq_cc_feature(text, freq):
    return float(freq['cc'])
def freq_robert_feature(text, freq):
    return float(freq['robert'])
def freq_attached_feature(text, freq):
    return float(freq['attached'])
def freq_30_feature(text, freq):
    return float(freq['30'])
def freq_j_feature(text, freq):
    return float(freq['j'])
def freq_am_feature(text, freq):
    return float(freq['am'])
def freq_call_feature(text, freq):
    return float(freq['call'])
def freq_july_feature(text, freq):
    return float(freq['july'])
def freq_25_feature(text, freq):
    return float(freq['25'])
def freq_email_feature(text, freq):
    return float(freq['email'])
def freq_sent_feature(text, freq):
    return float(freq['sent'])
def freq_nom_feature(text, freq):
    return float(freq['nom'])

def freq_12_feature(text, freq):
    return float(freq['12'])
def freq_mary_feature(text, freq):
    return float(freq['mary'])
def freq_sitara_feature(text, freq):
    return float(freq['sitara'])
def freq_corp_feature(text, freq):
    return float(freq['corp'])
def freq_gary_feature(text, freq):
    return float(freq['gary'])
def freq_stock_feature(text, freq):
    return float(freq['stock'])
def freq_bob_feature(text, freq):
    return float(freq['bob'])
def freq_like_feature(text, freq):
    return float(freq['like'])
def freq_08_feature(text, freq):
    return float(freq['08'])
def freq_meter_feature(text, freq):
    return float(freq['meter'])
def freq_x_feature(text, freq):
    return float(freq['x'])
def freq_following_feature(text, freq):
    return float(freq['following'])
def freq_click_feature(text, freq):
    return float(freq['click'])
def freq_looking_feature(text, freq):
    return float(freq['looking'])
def freq_28_feature(text, freq):
    return float(freq['28'])
def freq_06_feature(text, freq):
    return float(freq['06'])
def freq_million_feature(text, freq):
    return float(freq['million'])
def freq_deal_feature(text, freq):
    return float(freq['deal'])
def freq_within_feature(text, freq):
    return float(freq['within'])
def freq_center_feature(text, freq):
    return float(freq['center'])
def freq_computron_feature(text, freq):
    return float(freq['computron'])
def freq_deals_feature(text, freq):
    return float(freq['deals'])
def freq_march_feature(text, freq):
    return float(freq['march'])
def freq_width_feature(text, freq):
    return float(freq['width'])
def freq_order_feature(text, freq):
    return float(freq['order'])
def freq_file_feature(text, freq):
    return float(freq['file'])
def freq_how_feature(text, freq):
    return float(freq['how'])
def freq_r_feature(text, freq):
    return float(freq['r'])
def freq_here_feature(text, freq):
    return float(freq['here'])
def freq_95_feature(text, freq):
    return float(freq['95'])

def freq_save_feature(text, freq):
    return float(freq['save'])
def freq_want_feature(text, freq):
    return float(freq['want'])
def freq_td_feature(text, freq):
    return float(freq['td'])
def freq_back_feature(text, freq):
    return float(freq['back'])
def freq_63_feature(text, freq):
    return float(freq['63'])
def freq_securities_feature(text, freq):
    return float(freq['securities'])
def freq_flow_feature(text, freq):
    return float(freq['flow'])
def freq_per_feature(text, freq):
    return float(freq['per'])
def freq_farmer_feature(text, freq):
    return float(freq['farmer'])
def freq_month_feature(text, freq):
    return float(freq['month'])
def freq_melissa_feature(text, freq):
    return float(freq['melissa'])
def freq_online_feature(text, freq):
    return float(freq['online'])
def freq_business_feature(text, freq):
    return float(freq['business'])
def freq_60_feature(text, freq):
    return float(freq['60'])
def freq_know_feature(text, freq):
    return float(freq['know'])
def freq_original_feature(text, freq):
    return float(freq['original'])
def freq_houston_feature(text, freq):
    return float(freq['houston'])
def freq_height_feature(text, freq):
    return float(freq['height'])
def freq_inc_feature(text, freq):
    return float(freq['inc'])
def freq_pm_feature(text, freq):
    return float(freq['pm'])

def freq_daren_feature(text, freq):
    return float(freq['daren'])
def freq_investment_feature(text, freq):
    return float(freq['investment'])
def freq_713_feature(text, freq):
    return float(freq['713'])
def freq_07_feature(text, freq):
    return float(freq['07'])
def freq_nomination_feature(text, freq):
    return float(freq['nomination'])
def freq_most_feature(text, freq):
    return float(freq['most'])
def freq_volumes_feature(text, freq):
    return float(freq['volumes'])
def freq_when_feature(text, freq):
    return float(freq['when'])
def freq_internet_feature(text, freq):
    return float(freq['internet'])
def freq_thanks_feature(text, freq):
    return float(freq['thanks'])
def freq_2000_feature(text, freq):
    return float(freq['2000'])
def freq_questions_feature(text, freq):
    return float(freq['questions'])
def freq_05_feature(text, freq):
    return float(freq['05'])
def freq_make_feature(text, freq):
    return float(freq['make'])
def freq_font_feature(text, freq):
    return float(freq['font'])
def freq_link_feature(text, freq):
    return float(freq['link'])
def freq_pec_feature(text, freq):
    return float(freq['pec'])
def freq_use_feature(text, freq):
    return float(freq['use'])
def freq_money_feature(text, freq):
    return float(freq['money'])
def freq_some_feature(text, freq):
    return float(freq['some'])
def freq_should_feature(text, freq):
    return float(freq['should'])
def freq_ect_feature(text, freq):
    return float(freq['ect'])
def freq_gas_feature(text, freq):
    return float(freq['gas'])
def freq_line_feature(text, freq):
    return float(freq['line'])
def freq_16_feature(text, freq):
    return float(freq['16'])
def freq_windows_feature(text, freq):
    return float(freq['windows'])
def freq_production_feature(text, freq):
    return float(freq['production'])
def freq_09_feature(text, freq):
    return float(freq['09'])
def freq_01_feature(text, freq):
    return float(freq['01'])
def freq_report_feature(text, freq):
    return float(freq['report'])
def freq_market_feature(text, freq):
    return float(freq['market'])
def freq_microsoft_feature(text, freq):
    return float(freq['microsoft'])
def freq_50_feature(text, freq):
    return float(freq['50'])
def freq_size_feature(text, freq):
    return float(freq['size'])
def freq_forward_feature(text, freq):
    return float(freq['forward'])
def freq_2001_feature(text, freq):
    return float(freq['2001'])
def freq_best_feature(text, freq):
    return float(freq['best'])
def freq_align_feature(text, freq):
    return float(freq['align'])
def freq_only_feature(text, freq):
    return float(freq['only'])
def freq_forwarded_feature(text, freq):
    return float(freq['forwarded'])
def freq_contract_feature(text, freq):
    return float(freq['contract'])
def freq_31_feature(text, freq):
    return float(freq['31'])
def freq_ticket_feature(text, freq):
    return float(freq['ticket'])
def freq_its_feature(text, freq):
    return float(freq['its'])
def freq_g_feature(text, freq):
    return float(freq['g'])
def freq_xp_feature(text, freq):
    return float(freq['xp'])
def freq_let_feature(text, freq):
    return float(freq['let'])
def freq_go_feature(text, freq):
    return float(freq['go'])
def freq_xls_feature(text, freq):
    return float(freq['xls'])
def freq_other_feature(text, freq):
    return float(freq['other'])
def freq_www_feature(text, freq):
    return float(freq['www'])
def freq_21_feature(text, freq):
    return float(freq['21'])
def freq_tenaska_feature(text, freq):
    return float(freq['tenaska'])
def freq_contact_feature(text, freq):
    return float(freq['contact'])

def freq_news_feature(text, freq):
    return float(freq['news'])
def freq_v_feature(text, freq):
    return float(freq['v'])
def freq_ena_feature(text, freq):
    return float(freq['ena'])
def freq_hpl_feature(text, freq):
    return float(freq['hpl'])
def freq_style_feature(text, freq):
    return float(freq['style'])
def freq_nd_feature(text, freq):
    return float(freq['nd'])
def freq_energy_feature(text, freq):
    return float(freq['energy'])
def freq_volume_feature(text, freq):
    return float(freq['volume'])
def freq_than_feature(text, freq):
    return float(freq['than'])
def freq_b_feature(text, freq):
    return float(freq['b'])
def freq_http_feature(text, freq):
    return float(freq['http'])
def freq_prices_feature(text, freq):
    return float(freq['prices'])
def freq_adobe_feature(text, freq):
    return float(freq['adobe'])

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    feature.append(freq_03_feature(text, freq))
    feature.append(freq_now_feature(text, freq))
    feature.append(freq_tr_feature(text, freq))
    feature.append(freq_l_feature(text, freq))
    feature.append(freq_02_feature(text, freq))
    feature.append(freq_professional_feature(text, freq))
    feature.append(freq_enron_feature(text, freq))
    feature.append(freq_nbsp_feature(text, freq))
    feature.append(freq_pills_feature(text, freq))
    feature.append(freq_plus_feature(text, freq))
    feature.append(freq_d_feature(text, freq))
    feature.append(freq_effective_feature(text, freq))
    feature.append(freq_pt_feature(text, freq))
    feature.append(freq_hou_feature(text, freq))
    feature.append(freq_2004_feature(text, freq))
    feature.append(freq_international_feature(text, freq))
    feature.append(freq_04_feature(text, freq))
    feature.append(freq_100_feature(text, freq))
    feature.append(freq_mmbtu_feature(text, freq))
    feature.append(freq_future_feature(text, freq))
    feature.append(freq_subject_feature(text, freq))
    feature.append(freq_u_feature(text, freq))
    feature.append(freq_day_feature(text, freq))
    feature.append(freq_20_feature(text, freq))
    feature.append(freq_more_feature(text, freq))
    feature.append(freq_change_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_today_feature(text, freq))
    feature.append(freq_11_feature(text, freq))
    feature.append(freq_statements_feature(text, freq))
    feature.append(freq_software_feature(text, freq))
    feature.append(freq_without_feature(text, freq))
    feature.append(freq_over_feature(text, freq))
    feature.append(freq_office_feature(text, freq))

    feature.append(freq_daily_feature(text, freq))
    feature.append(freq_o_feature(text, freq))
    feature.append(freq_texas_feature(text, freq))
    feature.append(freq_69_feature(text, freq))
    feature.append(freq_april_feature(text, freq))
    feature.append(freq_products_feature(text, freq))
    feature.append(freq_cc_feature(text, freq))
    feature.append(freq_robert_feature(text, freq))
    feature.append(freq_attached_feature(text, freq))
    feature.append(freq_30_feature(text, freq))
    feature.append(freq_j_feature(text, freq))
    feature.append(freq_am_feature(text, freq))
    feature.append(freq_call_feature(text, freq))
    feature.append(freq_july_feature(text, freq))
    feature.append(freq_25_feature(text, freq))
    feature.append(freq_email_feature(text, freq))
    feature.append(freq_sent_feature(text, freq))
    feature.append(freq_nom_feature(text, freq))

    feature.append(freq_12_feature(text, freq))
    feature.append(freq_mary_feature(text, freq))
    feature.append(freq_sitara_feature(text, freq))
    feature.append(freq_corp_feature(text, freq))
    feature.append(freq_gary_feature(text, freq))
    feature.append(freq_stock_feature(text, freq))
    feature.append(freq_bob_feature(text, freq))
    feature.append(freq_like_feature(text, freq))
    feature.append(freq_08_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_x_feature(text, freq))
    feature.append(freq_following_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_looking_feature(text, freq))
    feature.append(freq_28_feature(text, freq))
    feature.append(freq_06_feature(text, freq))
    feature.append(freq_million_feature(text, freq))
    feature.append(freq_deal_feature(text, freq))
    feature.append(freq_within_feature(text, freq))
    feature.append(freq_center_feature(text, freq))
    feature.append(freq_computron_feature(text, freq))
    feature.append(freq_deals_feature(text, freq))
    feature.append(freq_march_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_order_feature(text, freq))
    feature.append(freq_file_feature(text, freq))
    feature.append(freq_how_feature(text, freq))
    feature.append(freq_r_feature(text, freq))
    feature.append(freq_here_feature(text, freq))
    feature.append(freq_95_feature(text, freq))

    feature.append(freq_save_feature(text, freq))
    feature.append(freq_want_feature(text, freq))
    feature.append(freq_td_feature(text, freq))
    feature.append(freq_back_feature(text, freq))
    feature.append(freq_63_feature(text, freq))
    feature.append(freq_securities_feature(text, freq))
    feature.append(freq_flow_feature(text, freq))
    feature.append(freq_per_feature(text, freq))
    feature.append(freq_farmer_feature(text, freq))
    feature.append(freq_month_feature(text, freq))
    feature.append(freq_melissa_feature(text, freq))
    feature.append(freq_online_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_60_feature(text, freq))
    feature.append(freq_know_feature(text, freq))
    feature.append(freq_original_feature(text, freq))
    feature.append(freq_houston_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_inc_feature(text, freq))
    feature.append(freq_pm_feature(text, freq))

    feature.append(freq_daren_feature(text, freq))
    feature.append(freq_investment_feature(text, freq))
    feature.append(freq_713_feature(text, freq))
    feature.append(freq_07_feature(text, freq))
    feature.append(freq_nomination_feature(text, freq))
    feature.append(freq_most_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_when_feature(text, freq))
    feature.append(freq_internet_feature(text, freq))
    feature.append(freq_thanks_feature(text, freq))
    feature.append(freq_2000_feature(text, freq))
    feature.append(freq_questions_feature(text, freq))
    feature.append(freq_05_feature(text, freq))
    feature.append(freq_make_feature(text, freq))
    feature.append(freq_font_feature(text, freq))
    feature.append(freq_link_feature(text, freq))
    feature.append(freq_pec_feature(text, freq))
    feature.append(freq_use_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_some_feature(text, freq))
    feature.append(freq_should_feature(text, freq))
    feature.append(freq_ect_feature(text, freq))
    feature.append(freq_gas_feature(text, freq))
    feature.append(freq_line_feature(text, freq))
    feature.append(freq_16_feature(text, freq))
    feature.append(freq_windows_feature(text, freq))
    feature.append(freq_production_feature(text, freq))
    feature.append(freq_09_feature(text, freq))
    feature.append(freq_01_feature(text, freq))
    feature.append(freq_report_feature(text, freq))
    feature.append(freq_market_feature(text, freq))
    feature.append(freq_microsoft_feature(text, freq))
    feature.append(freq_50_feature(text, freq))
    feature.append(freq_size_feature(text, freq))
    feature.append(freq_forward_feature(text, freq))
    feature.append(freq_2001_feature(text, freq))
    feature.append(freq_best_feature(text, freq))
    feature.append(freq_align_feature(text, freq))
    feature.append(freq_only_feature(text, freq))
    feature.append(freq_forwarded_feature(text, freq))
    feature.append(freq_contract_feature(text, freq))
    feature.append(freq_31_feature(text, freq))
    feature.append(freq_ticket_feature(text, freq))
    feature.append(freq_its_feature(text, freq))
    feature.append(freq_g_feature(text, freq))
    feature.append(freq_xp_feature(text, freq))
    feature.append(freq_let_feature(text, freq))
    feature.append(freq_go_feature(text, freq))
    feature.append(freq_xls_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_www_feature(text, freq))
    feature.append(freq_21_feature(text, freq))
    feature.append(freq_tenaska_feature(text, freq))
    feature.append(freq_contact_feature(text, freq))

    feature.append(freq_news_feature(text, freq))
    feature.append(freq_v_feature(text, freq))
    feature.append(freq_ena_feature(text, freq))
    feature.append(freq_hpl_feature(text, freq))
    feature.append(freq_style_feature(text, freq))
    feature.append(freq_nd_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_volume_feature(text, freq))
    feature.append(freq_than_feature(text, freq))
    feature.append(freq_b_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_prices_feature(text, freq))
    feature.append(freq_adobe_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
