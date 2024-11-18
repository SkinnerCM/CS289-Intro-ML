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



# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_www(text, freq):
    return float(freq['www'])

def freq_com(text, freq):
    return float(freq['com'])

def freq_bless(text, freq):
    return float(freq['bless'])

def freq_http_space(text, freq):
    return float(freq['http '])

def freq_ambien(text, freq):
    return float(freq['ambien'])

def freq_penis(text, freq):
    return float(freq['penis'])

def freq_enlarge(text, freq):
    return float(freq['enlarge'])

def freq_sex(text, freq):
    return float(freq['sex'])

def freq_http(text, freq):
    return float(freq['http'])

def freq_questions(text, freq):
    return float(freq['questions'])

def freq_we(text, freq):
    return float(freq['we'])


def freq_thanks(text, freq):
    return float(freq['thanks'])

def freq_i(text, freq):
    return float(freq['i'])

def freq_congratulations(text, freq):
    return float(freq['congratulations'])

def freq_benin(text, freq):
    return float(freq['benin'])

def freq_won(text, freq):
    return float(freq['won'])

def freq_winner(text, freq):
    return float(freq['winner'])

def freq_sperm(text, freq):
    return float(freq['sperm'])

def freq_payment(text, freq):
    return float(freq['payment'])

def freq_company(text, freq):
    return float(freq['company'])

def freq_god(text, freq):
    return float(freq['god'])

def freq_girls(text, freq):
    return float(freq['girls'])

def freq_naked(text, freq):
    return float(freq['naked'])

def freq_live(text, freq):
    return float(freq['live'])

def freq_enron(text, freq):
    return float(freq['enron'])

def freq_colon(text,freq):
    return text.count(':')

def freq_fslash(text,freq):
    return text.count('/')

def freq_space(text,freq):
    return text.count(' ')

def freq_star(text, freq):
    return text.count('*')

def freq_dash(text, freq):
    return text.count('-')

def freq_plus(text, freq):
    return text.count('+')

def freq_atsymbol(text, freq):
    return text.count('@')


def freq_mmbtu(text, freq): 
    return float(freq['mmbtu'])
def freq_font(text, freq): 
    return float(freq['font'])
def freq_without(text, freq): 
    return float(freq['without'])
def freq_best(text, freq): 
    return float(freq['best'])
def freq_only(text, freq): 
    return float(freq['only'])
def freq_through(text, freq): 
    return float(freq['through'])
def freq_thanks(text, freq): 
    return float(freq['thanks'])
def freq_net(text, freq): 
    return float(freq['net'])
def freq_p(text, freq): 
    return float(freq['p'])
def freq_which(text, freq): 
    return float(freq['which'])
def freq_7(text, freq): 
    return float(freq['7'])
def freq_l(text, freq): 
    return float(freq['l'])
def freq_million(text, freq): 
    return float(freq['million'])
def freq_free(text, freq): 
    return float(freq['free'])
def freq_forward(text, freq): 
    return float(freq['forward'])
def freq_high(text, freq): 
    return float(freq['high'])
def freq_www(text, freq): 
    return float(freq['www'])
def freq_cc(text, freq): 
    return float(freq['cc'])
def freq_let(text, freq): 
    return float(freq['let'])
def freq_its(text, freq): 
    return float(freq['its'])
def freq_looking(text, freq): 
    return float(freq['looking'])
def freq_here(text, freq): 
    return float(freq['here'])
def freq_percent(text, freq): 
    return float(freq['%'])
def freq_farmer(text, freq): 
    return float(freq['farmer'])
def freq_stock(text, freq): 
    return float(freq['stock'])
def freq_0(text, freq): 
    return float(freq['0'])
def freq_over(text, freq): 
    return float(freq['over'])
def freq_ect(text, freq): 
    return float(freq['ect'])
def freq_mail(text, freq): 
    return float(freq['mail'])
def freq_my(text, freq): 
    return float(freq['my'])
def freq_report(text, freq): 
    return float(freq['report'])
def freq_online(text, freq): 
    return float(freq['online'])
def freq_prices(text, freq): 
    return float(freq['prices'])
def freq_info(text, freq): 
    return float(freq['info'])
def freq_no(text, freq): 
    return float(freq['no'])
def freq_n(text, freq): 
    return float(freq['n'])
def freq_these(text, freq): 
    return float(freq['these'])

def freq_news(text, freq): 
    return float(freq['news'])
def freq_more(text, freq): 
    return float(freq['more'])
def freq_link(text, freq): 
    return float(freq['link'])
def freq_http(text, freq): 
    return float(freq['http'])
def freq_computron(text, freq): 
    return float(freq['computron'])
def freq_products(text, freq): 
    return float(freq['products'])
def freq_rbracket(text, freq): 
    return float(freq[']'])
def freq_out(text, freq): 
    return float(freq['out'])
def freq_new(text, freq): 
    return float(freq['new'])
def freq_about(text, freq): 
    return float(freq['about'])
def freq_professional(text, freq): 
    return float(freq['professional'])
def freq_should(text, freq): 
    return float(freq['should'])
def freq_into(text, freq): 
    return float(freq['into'])
def freq_pills(text, freq): 
    return float(freq['pills'])
def freq_u(text, freq): 
    return float(freq['u'])
def freq_am(text, freq): 
    return float(freq['am'])
def freq_within(text, freq): 
    return float(freq['within'])
def freq_also(text, freq): 
    return float(freq['also'])
def freq_20(text, freq): 
    return float(freq['20'])
def freq_statements(text, freq): 
    return float(freq['statements'])
def freq_software(text, freq): 
    return float(freq['software'])
def freq_inc(text, freq): 
    return float(freq['inc'])
def freq_100(text, freq): 
    return float(freq['100'])
def freq_c(text, freq): 
    return float(freq['c'])
def freq_lbracket(text, freq): 
    return float(freq['['])

def freq_td(text, freq): 
    return float(freq['td'])
def freq_other(text, freq): 
    return float(freq['other'])
def freq_just(text, freq): 
    return float(freq['just'])
def freq_width(text, freq): 
    return float(freq['width'])
def freq_x(text, freq): 
    return float(freq['x'])
def freq_today(text, freq): 
    return float(freq['today'])

def freq_attached(text, freq): 
    return float(freq['attached'])
def freq_go(text, freq): 
    return float(freq['go'])
def freq_size(text, freq): 
    return float(freq['size'])
def freq_9(text, freq): 
    return float(freq['9'])
def freq_daren(text, freq): 
    return float(freq['daren'])
def freq_nbsp(text, freq): 
    return float(freq['nbsp'])
def freq_what(text, freq): 
    return float(freq['what'])
def freq_corp(text, freq): 
    return float(freq['corp'])
def freq_business(text, freq): 
    return float(freq['business'])
def freq_most(text, freq): 
    return float(freq['most'])
def freq_how(text, freq): 
    return float(freq['how'])
def freq_deal(text, freq): 
    return float(freq['deal'])
def freq_b(text, freq): 
    return float(freq['b'])
def freq_time(text, freq): 
    return float(freq['time'])
def freq_2001(text, freq): 
    return float(freq['2001'])
def freq_viagra(text, freq): 
    return float(freq['viagra'])
def freq_one(text, freq): 
    return float(freq['one'])
def freq_rangle(text, freq): 
    return float(freq['>'])

def freq_market(text, freq): 
    return float(freq['market'])
def freq_know(text, freq): 
    return float(freq['know'])
def freq_11(text, freq): 
    return float(freq['11'])
def freq_message(text, freq): 
    return float(freq['message'])
def freq_br(text, freq): 
    return float(freq['br'])
def freq_01(text, freq): 
    return float(freq['01'])
def freq_international(text, freq): 
    return float(freq['international'])
def freq_many(text, freq): 
    return float(freq['many'])
def freq_hpl(text, freq): 
    return float(freq['hpl'])
def freq_securities(text, freq): 
    return float(freq['securities'])
def freq_account(text, freq): 
    return float(freq['account'])
def freq_but(text, freq): 
    return float(freq['but'])
def freq_money(text, freq): 
    return float(freq['money'])
def freq_use(text, freq): 
    return float(freq['use'])
def freq_99(text, freq): 
    return float(freq['99'])
def freq_up(text, freq): 
    return float(freq['up'])
def freq_future(text, freq): 
    return float(freq['future'])
def freq_v(text, freq): 
    return float(freq['v'])
def freq_save(text, freq): 
    return float(freq['save'])
def freq_o(text, freq): 
    return float(freq['o'])
def freq_2004(text, freq): 
    return float(freq['2004'])
def freq_make(text, freq): 
    return float(freq['make'])
def freq_hou(text, freq): 
    return float(freq['hou'])
def freq_now(text, freq): 
    return float(freq['now'])
def freq_windows(text, freq): 
    return float(freq['windows'])
def freq_information(text, freq): 
    return float(freq['information'])
def freq_there(text, freq): 
    return float(freq['there'])
def freq_their(text, freq): 
    return float(freq['their'])
def freq_8(text, freq): 
    return float(freq['8'])
def freq_02(text, freq): 
    return float(freq['02'])
def freq_2000(text, freq): 
    return float(freq['2000'])
def freq_meter(text, freq): 
    return float(freq['meter'])
def freq_pipe(text, freq): 
    return float(freq['|'])
def freq_j(text, freq): 
    return float(freq['j'])
def freq_us(text, freq): 
    return float(freq['us'])
def freq_60(text, freq): 
    return float(freq['60'])
def freq_would(text, freq): 
    return float(freq['would'])
def freq_office(text, freq): 
    return float(freq['office'])
def freq_so(text, freq): 
    return float(freq['so'])
def freq_may(text, freq): 
    return float(freq['may'])
def freq_pm(text, freq): 
    return float(freq['pm'])
def freq_than(text, freq): 
    return float(freq['than'])
def freq_forwarded(text, freq): 
    return float(freq['forwarded'])
def freq_order(text, freq): 
    return float(freq['order'])
def freq_an(text, freq): 
    return float(freq['an'])
def freq_price(text, freq): 
    return float(freq['price'])
def freq_email(text, freq): 
    return float(freq['email'])
def freq_6(text, freq): 
    return float(freq['6'])
def freq_world(text, freq): 
    return float(freq['world'])
def freq_03(text, freq): 
    return float(freq['03'])
def freq_companies(text, freq): 
    return float(freq['companies'])
def freq_investment(text, freq): 
    return float(freq['investment'])
def freq_m(text, freq): 
    return float(freq['m'])
def freq_microsoft(text, freq): 
    return float(freq['microsoft'])
def freq_enron(text, freq): 
    return float(freq['enron'])
def freq_been(text, freq): 
    return float(freq['been'])
def freq_t(text, freq): 
    return float(freq['t'])
def freq_company(text, freq): 
    return float(freq['company'])
def freq_contact(text, freq): 
    return float(freq['contact'])
def freq_r(text, freq): 
    return float(freq['r'])
def freq_like(text, freq): 
    return float(freq['like'])
def freq_50(text, freq): 
    return float(freq['50'])
def freq_12(text, freq): 
    return float(freq['12'])
def freq_such(text, freq): 
    return float(freq['such'])
def freq_get(text, freq): 
    return float(freq['get'])
def freq_they(text, freq): 
    return float(freq['they'])
def freq_click(text, freq): 
    return float(freq['click'])
def freq_do(text, freq): 
    return float(freq['do'])
def freq_want(text, freq): 
    return float(freq['want'])
def freq_subject(text, freq): 
    return float(freq['subject'])
def freq_before(text, freq): 
    return float(freq['before'])



# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pipe(text,freq))
    feature.append(freq_rangle(text,freq))
    feature.append(freq_lbracket(text,freq))
    feature.append(freq_rbracket(text,freq))
    feature.append(freq_mmbtu(text,freq))
    feature.append(freq_font(text,freq))
    feature.append(freq_without(text,freq))
    feature.append(freq_best(text,freq))
    feature.append(freq_only(text,freq))
    feature.append(freq_through(text,freq))
    feature.append(freq_thanks(text,freq))
    feature.append(freq_net(text,freq))
    feature.append(freq_p(text,freq))
    feature.append(freq_which(text,freq))
    feature.append(freq_7(text,freq))
    feature.append(freq_l(text,freq))
    feature.append(freq_million(text,freq))
    feature.append(freq_free(text,freq))
    feature.append(freq_forward(text,freq))
    feature.append(freq_high(text,freq))
    feature.append(freq_www(text,freq))
    feature.append(freq_cc(text,freq))
    feature.append(freq_let(text,freq))
    feature.append(freq_its(text,freq))
    feature.append(freq_looking(text,freq))
    feature.append(freq_here(text,freq))
    feature.append(freq_percent(text,freq))
    feature.append(freq_farmer(text,freq))
    feature.append(freq_stock(text,freq))
    feature.append(freq_0(text,freq))
    feature.append(freq_over(text,freq))
    feature.append(freq_ect(text,freq))
    feature.append(freq_mail(text,freq))
    feature.append(freq_my(text,freq))
    feature.append(freq_report(text,freq))
    feature.append(freq_online(text,freq))
    feature.append(freq_prices(text,freq))
    feature.append(freq_info(text,freq))
    feature.append(freq_no(text,freq))
    feature.append(freq_n(text,freq))
    feature.append(freq_these(text,freq))
    
    feature.append(freq_news(text,freq))
    feature.append(freq_more(text,freq))
    feature.append(freq_link(text,freq))
    feature.append(freq_http(text,freq))
    feature.append(freq_computron(text,freq))
    feature.append(freq_products(text,freq))
    feature.append(freq_out(text,freq))
    feature.append(freq_new(text,freq))
    feature.append(freq_about(text,freq))
    feature.append(freq_professional(text,freq))
    feature.append(freq_should(text,freq))
    feature.append(freq_into(text,freq))
    feature.append(freq_pills(text,freq))
    feature.append(freq_u(text,freq))
    feature.append(freq_am(text,freq))
    feature.append(freq_within(text,freq))
    feature.append(freq_also(text,freq))
    feature.append(freq_20(text,freq))
    feature.append(freq_statements(text,freq))
    feature.append(freq_software(text,freq))
    feature.append(freq_inc(text,freq))
    feature.append(freq_100(text,freq))
    feature.append(freq_c(text,freq))
    feature.append(freq_td(text,freq))
    feature.append(freq_other(text,freq))
    feature.append(freq_just(text,freq))
    feature.append(freq_width(text,freq))
    feature.append(freq_x(text,freq))
    feature.append(freq_today(text,freq))
    
    feature.append(freq_attached(text,freq))
    feature.append(freq_go(text,freq))
    feature.append(freq_size(text,freq))
    feature.append(freq_9(text,freq))
    feature.append(freq_daren(text,freq))
    feature.append(freq_nbsp(text,freq))
    feature.append(freq_what(text,freq))
    feature.append(freq_corp(text,freq))
    feature.append(freq_business(text,freq))
    feature.append(freq_most(text,freq))
    feature.append(freq_how(text,freq))
    feature.append(freq_deal(text,freq))
    feature.append(freq_b(text,freq))
    feature.append(freq_time(text,freq))
    feature.append(freq_2001(text,freq))
    feature.append(freq_viagra(text,freq))
    feature.append(freq_one(text,freq))
    
    feature.append(freq_market(text,freq))
    feature.append(freq_know(text,freq))
    feature.append(freq_11(text,freq))
    feature.append(freq_message(text,freq))
    feature.append(freq_br(text,freq))
    feature.append(freq_01(text,freq))
    feature.append(freq_international(text,freq))
    feature.append(freq_many(text,freq))
    feature.append(freq_hpl(text,freq))
    feature.append(freq_securities(text,freq))
    feature.append(freq_account(text,freq))
    feature.append(freq_but(text,freq))
    feature.append(freq_money(text,freq))
    feature.append(freq_use(text,freq))
    feature.append(freq_99(text,freq))
    feature.append(freq_up(text,freq))
    feature.append(freq_future(text,freq))
    feature.append(freq_v(text,freq))
    feature.append(freq_save(text,freq))
    feature.append(freq_o(text,freq))
    feature.append(freq_2004(text,freq))
    feature.append(freq_make(text,freq))
    feature.append(freq_hou(text,freq))
    feature.append(freq_now(text,freq))
    feature.append(freq_windows(text,freq))
    feature.append(freq_information(text,freq))
    feature.append(freq_there(text,freq))
    feature.append(freq_their(text,freq))
    feature.append(freq_8(text,freq))
    feature.append(freq_02(text,freq))
    feature.append(freq_2000(text,freq))
    feature.append(freq_meter(text,freq))
    
    feature.append(freq_j(text,freq))
    feature.append(freq_us(text,freq))
    feature.append(freq_60(text,freq))
    feature.append(freq_would(text,freq))
    feature.append(freq_office(text,freq))
    feature.append(freq_so(text,freq))
    feature.append(freq_may(text,freq))
    feature.append(freq_pm(text,freq))
    feature.append(freq_than(text,freq))
    feature.append(freq_forwarded(text,freq))
    feature.append(freq_order(text,freq))
    feature.append(freq_an(text,freq))
    feature.append(freq_price(text,freq))
    feature.append(freq_email(text,freq))
    feature.append(freq_6(text,freq))
    feature.append(freq_world(text,freq))
    feature.append(freq_03(text,freq))
    feature.append(freq_companies(text,freq))
    feature.append(freq_investment(text,freq))
    feature.append(freq_m(text,freq))
    feature.append(freq_microsoft(text,freq))
    feature.append(freq_enron(text,freq))
    feature.append(freq_been(text,freq))
    feature.append(freq_t(text,freq))
    feature.append(freq_company(text,freq))
    feature.append(freq_contact(text,freq))
    feature.append(freq_r(text,freq))
    feature.append(freq_like(text,freq))
    feature.append(freq_50(text,freq))
    feature.append(freq_12(text,freq))
    feature.append(freq_such(text,freq))
    feature.append(freq_get(text,freq))
    feature.append(freq_they(text,freq))
    feature.append(freq_click(text,freq))
    feature.append(freq_do(text,freq))
    feature.append(freq_want(text,freq))
    feature.append(freq_subject(text,freq))
    feature.append(freq_before(text,freq))

    feature.append(freq_company(text,freq))
    feature.append(freq_star(text,freq))
    feature.append(freq_enron(text,freq))
    feature.append(freq_atsymbol(text,freq))
    feature.append(freq_plus(text,freq))
    feature.append(freq_god(text,freq))
    feature.append(freq_benin(text,freq))
    feature.append(freq_payment(text,freq))
    feature.append(freq_dash(text,freq))
    feature.append(freq_questions(text,freq))
    feature.append(freq_sperm(text,freq))
    feature.append(freq_won(text,freq))
    feature.append(freq_winner(text,freq))
    feature.append(freq_congratulations(text,freq))
    feature.append(freq_i(text,freq))
    feature.append(freq_we(text,freq))
    feature.append(freq_thanks(text,freq))
    feature.append(freq_bless(text,freq))
    feature.append(freq_colon(text,freq))
    feature.append(freq_http_space(text,freq))
    feature.append(freq_http(text,freq))
    feature.append(freq_ambien(text,freq))
    feature.append(freq_www(text,freq))
    feature.append(freq_com(text,freq))
    feature.append(freq_fslash(text,freq))
    feature.append(freq_penis(text,freq))
    feature.append(freq_enlarge(text,freq))
    feature.append(freq_sex(text,freq))
    feature.append(freq_space(text,freq))


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
