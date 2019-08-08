def plot_cleansing(movies):
    movies['PlotCorrected'] = movies['Plot']
    movies['PlotCorrected'] = movies['PlotCorrected'].str.strip()
    movies['PlotCorrected'] = movies['PlotCorrected'].str.lower()
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('`', '\'')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('’', '\'')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace("what's", 'what is')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace("can't", 'can not')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace("n't", ' not')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace("i'm", 'i am')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'ve', 'have')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'ll', 'will')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'d', 'would')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'re', 'are')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'s', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\'', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\\r', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\\', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('/', ' ')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(',', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('.', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('!', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('?', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('"', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(':', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(';', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('(', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(')', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('—', ' ')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\\n', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\[\d+\]', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\d+–\d+', '') # diffrent '–' don't remove!
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\d+-\d+', '') # diffrent '-' don't remove!


    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\$\d+', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'£\d+', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'₤\d+', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'\d+%', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace(r'#\d+', '')

    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('&', ' ')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('“', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('”', '')

    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('ˈ', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('‘', '')

    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('…', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('{', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('}', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('-', ' ') # diffrent '–' don't remove!
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('–', ' ') # diffrent '–' don't remove!
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\[', '')
    movies['PlotCorrected'] = movies['PlotCorrected'].str.replace('\]', '')
    return movies
