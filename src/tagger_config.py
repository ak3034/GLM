tags = ['.', 'CONJ', 'NUM', 'X', 'DET', 'ADP', 'ADJ', 'VERB', 'NOUN', 'PRT', 'PRON', 'ADV']

terminals = ["*","STOP"]

#A baseline model for the history generator.
dict = {'limited': ['VERB', 'ADJ'], 'four': ['NUM'], 'Until': ['ADP'], 'Manufacturers': ['NOUN'], 'Western': ['ADJ', 'NOUN'], 'under': ['ADP'], 'risk': ['NOUN'], 'rise': ['VERB', 'NOUN'], 'every': ['DET'], 'Continental': ['NOUN'], 'Thomas': ['NOUN'], 'ASSETS': ['NOUN'], 'MONEY': ['NOUN'], 'companies': ['NOUN'], 'Nasdaq': ['NOUN'], 'Paul': ['NOUN'], 'Hess': ['NOUN'], 'Dozen': ['NOUN'], 'leaders': ['NOUN'], 'estimates': ['VERB', 'NOUN'], 'budget': ['NOUN'], 'second': ['ADJ'], 'machines': ['NOUN'], 'even': ['ADV', 'VERB', 'ADJ'], 'contributed': ['VERB'], 'spokesman': ['NOUN'], 'above': ['ADJ', 'ADP'], 'AT&T': ['NOUN'], 'net': ['ADJ', 'NOUN'], 'ever': ['ADV'], 'reporter': ['NOUN'], 'never': ['ADV'], 'here': ['ADV'], 'reported': ['VERB'], 'active': ['ADJ'], '100': ['NUM'], 'Merrill': ['NOUN'], 'auction': ['NOUN'], 'employees': ['NOUN'], 'climbed': ['VERB'], 'reports': ['VERB', 'NOUN'], 'controversy': ['NOUN'], 'credit': ['NOUN'], 'Market': ['NOUN'], 'military': ['NOUN', 'ADJ'], 'settled': ['VERB'], 'criticism': ['NOUN'], 'Group': ['NOUN'], 'Three': ['NUM'], 'highly': ['ADV'], 'brought': ['VERB'], 'visible': ['ADJ'], 'total': ['NOUN', 'ADJ'], 'unit': ['NOUN'], 'swings': ['NOUN'], 'Star': ['NOUN'], 'would': ['VERB'], 'Express': ['NOUN'], 'June': ['NOUN'], 'music': ['NOUN'], 'until': ['ADP'], 'PaineWebber': ['NOUN'], 'phone': ['NOUN'], 'hold': ['VERB', 'NOUN'], 'must': ['VERB'], 'Salomon': ['NOUN'], '1990': ['NUM'], 'Financial': ['ADJ', 'NOUN'], 'room': ['NOUN'], 'rights': ['NOUN'], 'work': ['VERB', 'NOUN'], 'my': ['PRON'], 'example': ['NOUN'], 'Short-term': ['ADJ'], 'estate': ['NOUN'], 'give': ['VERB'], 'cited': ['VERB'], 'Capital': ['NOUN'], 'caution': ['NOUN'], 'want': ['VERB'], 'Digital': ['NOUN'], 'totaled': ['VERB'], 'guarantee': ['VERB', 'NOUN'], 'end': ['VERB', 'NOUN'], 'recovery': ['NOUN'], 'Journal': ['NOUN'], 'damage': ['NOUN'], 'far': ['ADV'], 'answer': ['VERB', 'NOUN'], 'widespread': ['ADJ'], 'Stock': ['NOUN'], 'A': ['NOUN', 'DET'], 'after': ['ADV', 'ADP'], 'wrong': ['ADJ'], 'president': ['NOUN'], 'law': ['NOUN'], 'All': ['DET'], 'effective': ['ADJ', 'NOUN'], 'Japanese': ['NOUN', 'ADJ'], 'Telerate': ['NOUN'], 'Lloyd': ['NOUN'], 'Another': ['DET'], 'Trust': ['NOUN'], 'order': ['VERB', 'NOUN'], 'operations': ['NOUN'], 'office': ['NOUN'], 'over': ['ADV', 'PRT', 'ADJ', 'ADP'], 'vary': ['VERB'], 'expects': ['VERB'], 'before': ['ADV', 'ADP'], 'His': ['PRON'], 'personal': ['ADJ'], ',': ['.'], 'Here': ['ADV'], 'better': ['ADV', 'ADJ'], 'production': ['NOUN'], 'Judge': ['NOUN'], 'weeks': ['NOUN'], 'then': ['ADV'], 'them': ['PRON'], 'weakness': ['NOUN'], 'school': ['NOUN'], 'they': ['PRON'], 'bank': ['NOUN'], '30-year': ['ADJ'], 'Indeed': ['ADV'], 'India': ['NOUN'], 'each': ['DET'], 'went': ['VERB'], 'side': ['NOUN'], 'bond': ['NOUN'], 'financial': ['ADJ'], 'series': ['NOUN'], 'used': ['VERB'], 'trading': ['VERB', 'NOUN'], 'network': ['NOUN'], 'William': ['NOUN'], 'got': ['VERB'], 'advancers': ['NOUN'], 'University': ['NOUN'], 'estimate': ['NOUN'], 'created': ['VERB'], 'September': ['NOUN'], 'National': ['NOUN'], 'days': ['NOUN'], 'Funds': ['NOUN'], 'already': ['ADV'], 'features': ['VERB', 'NOUN'], 'economists': ['NOUN'], 'another': ['DET'], 'electronic': ['ADJ'], 'Congress': ['NOUN'], 'top': ['NOUN', 'VERB', 'ADJ'], 'needed': ['VERB'], 'rates': ['NOUN'], 'too': ['ADV'], 'percentage': ['NOUN'], 'took': ['VERB'], 'target': ['NOUN'], 'showed': ['VERB'], 'Only': ['ADV'], 'spending': ['NOUN'], 'project': ['NOUN'], 'matter': ['VERB', 'NOUN'], 'acquisition': ['NOUN'], 'Chicago': ['NOUN'], 'talking': ['VERB'], 'seem': ['VERB'], 'relatively': ['ADV'], 'Then': ['ADV'], '-': ['.'], 'They': ['PRON'], 'Bank': ['NOUN'], 'Jones': ['NOUN'], 'though': ['ADV', 'ADP'], 'plenty': ['ADV', 'ADJ', 'NOUN'], 'Bond': ['NOUN'], 'professor': ['NOUN'], 'points': ['NOUN'], 'outnumbered': ['VERB'], 'consumer': ['NOUN'], 'Its': ['PRON'], 'came': ['VERB'], 'Union': ['NOUN'], 'Dec.': ['NOUN'], 'Big': ['ADJ', 'NOUN'], 'Republicans': ['NOUN'], 'do': ['VERB'], 'exports': ['NOUN'], 'wide': ['ADJ'], 'de': ['ADJ', 'NOUN', 'ADP'], '13': ['NUM'], 'new': ['ADJ'], 'report': ['VERB', 'NOUN'], 'Soviet': ['NOUN', 'ADJ'], 'volatility': ['NOUN'], 'countries': ['NOUN'], 'Over': ['ADP'], 'Washington': ['NOUN'], 'twice': ['ADV', 'ADJ'], 'bad': ['ADJ'], 'Advancing': ['VERB'], 'fair': ['ADJ', 'NOUN'], 'result': ['VERB', 'NOUN'], 'resigned': ['VERB'], 'best': ['ADV', 'ADJ'], 'subject': ['NOUN', 'ADJ'], 'said': ['VERB'], 'away': ['ADV'], 'Prices': ['NOUN'], 'approach': ['NOUN'], 'we': ['PRON'], 'men': ['NOUN'], 'terms': ['NOUN'], 'wo': ['VERB'], 'weak': ['ADJ'], 'however': ['ADV'], 'drew': ['VERB'], 'news': ['NOUN'], 'debt': ['NOUN'], 'received': ['VERB'], 'country': ['NOUN'], 'against': ['ADP'], 'and': ['CONJ'], 'year-end': ['ADJ', 'NOUN'], 'tough': ['ADJ'], 'Petroleum': ['NOUN'], 'Board': ['NOUN'], 'trust': ['VERB', 'NOUN'], 'holdings': ['NOUN'], 'three': ['NUM'], 'been': ['VERB'], '.': ['.'], 'much': ['ADV', 'ADJ'], 'interest': ['NOUN'], 'expected': ['VERB'], 'life': ['NOUN'], 'Tokyo': ['NOUN'], 'drugs': ['NOUN'], 'Poland': ['NOUN'], 'worked': ['VERB'], 'Tokyu': ['NOUN'], 'Co': ['NOUN'], 'Dollar': ['NOUN'], 'air': ['NOUN'], 'property': ['NOUN'], 'Tuesday': ['NOUN'], 'seven': ['NUM'], '30-share': ['NUM', 'ADJ'], 'On': ['ADP'], 'is': ['VERB'], 'it': ['PRON'], 'player': ['NOUN'], 'Bush': ['NOUN'], 'world-wide': ['ADV', 'ADJ'], 'in': ['ADV', 'PRT', 'ADP'], 'victims': ['NOUN'], 'if': ['ADP'], 'things': ['NOUN'], 'make': ['VERB'], 'split': ['VERB', 'NOUN'], 'President': ['NOUN'], 'several': ['ADJ'], 'analysts': ['NOUN'], 'Maybe': ['ADV'], 'hand': ['VERB', 'NOUN'], 'Angeles': ['NOUN'], 'Home': ['NOUN'], 'RJR': ['NOUN'], 'programs': ['NOUN'], 'FUNDS': ['NOUN'], 'practices': ['NOUN'], 'claims': ['NOUN'], 'the': ['DET'], 'TRUST': ['NOUN'], 'corporate': ['ADJ'], 'investments': ['NOUN'], 'left': ['VERB'], 'After': ['ADP'], 'just': ['ADV', 'ADJ'], 'yen': ['NOUN'], 'yet': ['ADV', 'CONJ'], 'previous': ['ADJ'], 'had': ['VERB'], 'board': ['NOUN'], 'easy': ['ADV', 'ADJ'], 'has': ['VERB'], 'James': ['NOUN'], 'possible': ['ADJ'], 'buy-out': ['ADJ', 'NOUN'], 'judge': ['NOUN'], 'advanced': ['VERB'], 'Also': ['ADV'], '50': ['NUM'], 'securities': ['NOUN'], 'officer': ['NOUN'], 'night': ['NOUN'], 'security': ['NOUN'], 'Pentagon': ['NOUN'], 'right': ['ADV', 'ADJ'], 'old': ['ADJ'], 'deal': ['VERB', 'NOUN'], 'people': ['NOUN'], 'election': ['NOUN'], 'specific': ['ADJ'], 'for': ['ADP'], 'denied': ['VERB'], 'Accepted': ['ADJ'], 'He': ['PRON'], 'bank-backed': ['ADJ'], 'marketing': ['VERB', 'NOUN'], 'manufacturing': ['VERB', 'NOUN'], 'First': ['ADV', 'X', 'ADJ', 'NOUN'], 'dollars': ['NOUN'], 'months': ['NOUN'], 'magazine': ['NOUN'], 'afternoon': ['NOUN'], 'efforts': ['NOUN'], 'Still': ['ADV'], 'slightly': ['ADV'], 'raised': ['VERB'], 'managers': ['NOUN'], 'Lehman': ['NOUN'], 'civil': ['ADJ'], 'down': ['ADV', 'PRT', 'ADP'], 'magazines': ['NOUN'], 'zero-coupon': ['NOUN', 'ADJ'], 'Ends': ['NOUN'], 'support': ['VERB', 'NOUN'], 'initial': ['ADJ'], 'fight': ['VERB', 'NOUN'], 'Amex': ['NOUN'], 'editor': ['NOUN'], 'way': ['ADV', 'NOUN'], 'call': ['VERB', 'NOUN'], 'was': ['VERB'], 'head': ['VERB', 'NOUN'], 'offering': ['VERB', 'NOUN'], 'manufacturers': ['NOUN'], 'becoming': ['VERB'], 'payable': ['ADJ'], 'bids': ['NOUN'], '3.5': ['NUM'], 'Estimated': ['VERB', 'ADJ'], 'later': ['ADV'], 'covers': ['VERB'], 'evidence': ['NOUN'], "''": ['.'], 'exist': ['VERB'], '20': ['NUM'], 'Francisco': ['NOUN'], 'no': ['ADV', 'DET'], 'stake': ['NOUN'], 'generally': ['ADV'], 'role': ['NOUN'], "'s": ['PRT', 'VERB'], 'models': ['NOUN'], 'surprise': ['NOUN'], 'football': ['NOUN'], 'utilities': ['NOUN'], 'fell': ['VERB'], "'m": ['VERB'], 'billion': ['NUM'], 'longer': ['ADV'], 'changed': ['VERB', 'ADJ'], 'daily': ['ADV', 'ADJ'], 'time': ['NOUN'], 'serious': ['ADJ'], 'decision': ['NOUN'], 'profits': ['NOUN'], 'chain': ['NOUN'], 'battle': ['NOUN'], 'certainly': ['ADV'], 'Sept.': ['NOUN'], 'Columbia': ['NOUN'], 'charge': ['VERB', 'NOUN'], 'marks': ['NOUN'], 'me': ['PRON'], 'word': ['NOUN'], 'trouble': ['NOUN'], 'Jersey': ['NOUN'], 'level': ['NOUN'], 'did': ['VERB'], 'leave': ['VERB'], 'team': ['NOUN'], 'depository': ['ADJ', 'NOUN'], 'speculation': ['NOUN'], 'George': ['NOUN'], 'cost': ['VERB', 'NOUN'], 'Campeau': ['NOUN'], 'adds': ['VERB'], 'shares': ['NOUN'], 'Ford': ['NOUN'], 'current': ['ADJ'], 'goes': ['VERB'], 'international': ['ADJ'], 'boost': ['VERB', 'NOUN'], 'collateral': ['NOUN'], 'transportation': ['NOUN'], 'water': ['NOUN'], 'groups': ['NOUN'], 'alone': ['ADV'], 'Times': ['NOUN'], 'My': ['PRON'], 'earthquake': ['NOUN'], 'change': ['VERB', 'NOUN'], 'Warner': ['NOUN'], 'healthy': ['ADJ'], 'trial': ['NOUN'], 'history': ['NOUN'], 'Justice': ['NOUN'], 'lending': ['NOUN', 'VERB', 'ADJ'], 'market': ['VERB', 'NOUN'], 'August': ['NOUN'], 'psyllium': ['NOUN'], 'third-quarter': ['NOUN', 'ADJ'], 'opposed': ['VERB'], 'francs': ['NOUN'], 'Those': ['DET'], '``': ['.'], 'October': ['NOUN'], 'These': ['DET'], 'goods': ['NOUN'], 'Time': ['NOUN'], 'cases': ['NOUN'], 'effort': ['NOUN'], 'car': ['NOUN'], 'abortion': ['NOUN'], 'Pacific': ['NOUN'], 'can': ['VERB'], 'following': ['ADJ', 'VERB', 'NOUN'], 'making': ['VERB', 'NOUN'], 'heart': ['ADV', 'NOUN'], 'figure': ['NOUN'], 'Federal': ['NOUN'], '1980s': ['NUM', 'NOUN'], 'dropped': ['VERB'], 'means': ['VERB', 'NOUN'], '1': ['NUM'], 'economy': ['NOUN'], 'huge': ['ADJ'], 'may': ['VERB'], 'Average': ['NOUN', 'ADJ'], 'produce': ['VERB'], 'date': ['NOUN'], 'such': ['DET', 'ADJ'], 'Corporate': ['NOUN', 'ADJ'], 'Pilson': ['NOUN'], 'Series': ['NOUN'], 'futures': ['NOUN'], 'so': ['ADV', 'ADP'], 'talk': ['VERB', 'NOUN'], 'Hong': ['NOUN'], 'years': ['NOUN'], 'course': ['NOUN'], 'White': ['ADJ', 'NOUN'], 'still': ['ADV'], 'stock-index': ['ADJ', 'NOUN'], 'group': ['NOUN'], 'farmer': ['NOUN'], 'policy': ['NOUN'], 'World': ['NOUN'], 'main': ['NOUN', 'ADJ'], 'nation': ['NOUN'], 'She': ['PRON'], 'half': ['NOUN', 'DET', 'ADJ'], 'not': ['ADV'], 'R.': ['NOUN'], 'now': ['ADV', 'ADJ'], 'name': ['NOUN'], 'Their': ['PRON'], 'drop': ['VERB', 'NOUN'], 'quarter': ['NOUN'], 'year': ['NOUN'], 'happen': ['VERB'], 'morning': ['NOUN'], 'Carolina': ['NOUN'], 'opened': ['VERB'], 'space': ['NOUN'], 'profit': ['VERB', 'NOUN'], 'increase': ['VERB', 'NOUN'], 'shows': ['VERB', 'NOUN'], 'earlier': ['ADV', 'ADJ'], 'cars': ['NOUN'], 'million': ['NUM'], 'quite': ['ADV', 'DET'], 'language': ['NOUN'], 'modest': ['ADJ'], '7\\/8': ['NUM'], 'thing': ['NOUN'], 'place': ['VERB', 'NOUN'], 'think': ['VERB'], 'first': ['ADV', 'ADJ'], 'Even': ['ADV', 'ADJ'], 'revenue': ['NOUN'], 'There': ['ADV', 'DET'], 'one': ['PRON', 'NUM', 'NOUN'], 'directly': ['ADV'], 'vote': ['VERB', 'NOUN'], 'Hollywood': ['NOUN'], 'open': ['PRT', 'VERB', 'ADJ'], 'size': ['NOUN'], 'city': ['NOUN'], 'little': ['ADV', 'ADJ'], 'Monday': ['NOUN'], 'returns': ['NOUN'], '2': ['NUM'], 'Such': ['ADJ'], 'that': ['ADV', 'DET', 'ADP'], 'surged': ['VERB'], 'than': ['ADP'], 'Inc.': ['NOUN'], '11': ['NUM'], '10': ['NUM'], 'television': ['NOUN'], '12': ['NUM'], '15': ['NUM'], 'third': ['NOUN', 'ADJ'], '17': ['NUM'], '16': ['NUM'], '19': ['NUM'], '18': ['NUM'], 'require': ['VERB'], 'future': ['ADJ', 'NOUN'], 'venture': ['NOUN'], 'were': ['VERB'], 'Litigation': ['NOUN'], 'Court': ['NOUN'], 'investors': ['NOUN'], 'remained': ['VERB'], 'turned': ['VERB'], 'plunge': ['NOUN'], 'investment-grade': ['ADJ'], 'sells': ['VERB'], 'Valley': ['NOUN'], 'saw': ['VERB', 'NOUN'], 'any': ['ADV', 'DET'], 'offer': ['VERB', 'NOUN'], 'note': ['VERB', 'NOUN'], 'equipment': ['NOUN'], 'potential': ['NOUN', 'ADJ'], 'take': ['VERB'], 'performance': ['NOUN'], 'An': ['DET'], 'price': ['NOUN'], 'especially': ['ADV'], 'average': ['NOUN', 'ADJ'], 'farmers': ['NOUN'], 'sale': ['NOUN'], 'professional': ['ADJ', 'NOUN'], 'senior': ['ADJ'], 'typically': ['ADV'], 'show': ['VERB', 'NOUN'], 'Systems': ['NOUN'], 'We': ['PRON'], 'slow': ['VERB', 'ADJ'], 'title': ['NOUN'], '3': ['NUM'], 'only': ['ADV', 'ADJ'], 'going': ['VERB'], 'black': ['NOUN', 'ADJ'], 'get': ['VERB'], 'contracts': ['NOUN'], 'nearly': ['ADV'], 'W.': ['NOUN'], 'yield': ['VERB', 'NOUN'], 'Kong': ['NOUN'], 'losers': ['NOUN'], 'concern': ['NOUN'], 'mortgage': ['NOUN'], 'That': ['DET', 'ADP'], 'federal': ['ADJ'], 'jumped': ['VERB'], 'forecast': ['VERB', 'NOUN'], 'enough': ['ADV', 'NOUN', 'ADJ'], 'bureau': ['NOUN'], 'between': ['ADP'], 'import': ['NOUN'], 'Mitsubishi': ['NOUN'], 'across': ['ADP'], 'notice': ['VERB', 'NOUN'], 'parent': ['NOUN'], 'U.S.': ['NOUN'], 'Securities': ['NOUN'], 'article': ['NOUN'], 'come': ['VERB'], 'many': ['ADV', 'DET', 'ADJ'], 'quiet': ['NOUN', 'ADJ'], 'contract': ['NOUN'], 'holders': ['NOUN'], 'traded': ['VERB'], 'among': ['ADP'], 'Mortgage': ['NOUN'], 'period': ['NOUN'], 'But': ['CONJ'], 'Lynch': ['NOUN'], 'hardly': ['ADV'], 'wants': ['VERB'], 'shopping': ['VERB', 'NOUN'], 'Analysts': ['NOUN'], 'quick': ['ADJ'], 'former': ['ADJ'], 'those': ['DET'], 'case': ['NOUN'], 'these': ['DET'], 'Reagan': ['NOUN'], 'cash': ['NOUN'], "n't": ['ADV'], 'editorial': ['ADJ', 'NOUN'], 'situation': ['NOUN'], 'telephone': ['NOUN'], 'according': ['VERB'], 'technology': ['NOUN'], 'different': ['ADJ'], 'pay': ['VERB', 'NOUN'], 'same': ['ADJ'], 'week': ['NOUN'], 'oil': ['NOUN'], 'I': ['PRON', 'NOUN'], 'IRS': ['NOUN'], 'director': ['NOUN'], 'totally': ['ADV'], 'largely': ['ADV'], 'charges': ['VERB', 'NOUN'], 'Is': ['VERB'], 'Motors': ['NOUN'], 'It': ['PRON'], 'May': ['NOUN'], 'without': ['ADP'], 'relief': ['NOUN'], 'In': ['ADP'], 'If': ['ADP'], 'summer': ['NOUN'], 'United': ['NOUN'], 'money': ['NOUN'], 'actions': ['NOUN'], 'DISCOUNT': ['ADJ', 'NOUN'], 'ounces': ['NOUN'], 'Markets': ['NOUN'], 'blow': ['NOUN'], 'announcement': ['NOUN'], 'death': ['NOUN'], 'Municipal': ['NOUN', 'ADJ'], 'rose': ['VERB'], 'seems': ['VERB'], 'improvement': ['NOUN'], '4': ['NUM'], 'real': ['ADV', 'ADJ'], 'around': ['ADV', 'PRT', 'ADP'], 'rules': ['NOUN'], 'Many': ['ADJ'], 'Sachs': ['NOUN'], 'early': ['ADV', 'ADJ'], 'inflation': ['NOUN'], 'world': ['NOUN'], 'serves': ['VERB'], "'ve": ['VERB'], 'London': ['NOUN'], 'either': ['ADV', 'DET'], 'output': ['NOUN'], 'West': ['ADJ', 'NOUN'], 'Since': ['ADP'], 'BRIEFS': ['NOUN'], 'competition': ['NOUN'], 'International': ['ADJ', 'NOUN'], 'Net': ['NOUN', 'ADJ'], 'legal': ['ADJ'], 'provides': ['VERB'], 'moderate': ['ADJ'], 'felt': ['VERB'], 'business': ['NOUN'], 'on': ['ADV', 'PRT', 'ADP'], 'Revenue': ['NOUN'], 'of': ['ADP'], 'industry': ['NOUN'], 'airline': ['NOUN'], 'mixed': ['VERB', 'ADJ'], 'or': ['CONJ'], 'No': ['ADV', 'X', 'DET'], 'instruments': ['NOUN'], 'parties': ['NOUN'], 'your': ['PRON'], 'her': ['PRON'], 'area': ['NOUN'], 'there': ['ADV', 'DET'], 'start': ['VERB', 'NOUN'], 'low': ['ADJ'], 'lot': ['ADV', 'NOUN'], 'ago': ['ADV', 'ADP'], 'complete': ['ADJ'], 'with': ['ADP'], 'House': ['NOUN'], 'gone': ['VERB'], 'certain': ['ADJ'], 'moved': ['VERB'], 'sales': ['NOUN'], 'Thursday': ['NOUN'], 'general': ['ADJ'], 'as': ['ADV', 'ADP'], 'at': ['ADP'], 'politics': ['NOUN'], 'moves': ['NOUN'], 'again': ['ADV'], '5': ['NUM'], 'you': ['PRON'], 'More': ['ADV', 'ADJ'], 'Jaguar': ['NOUN'], 'includes': ['VERB'], 'important': ['ADJ'], 'stocks': ['NOUN'], 'building': ['NOUN'], 'calls': ['VERB', 'NOUN'], 'Street': ['NOUN'], 'overseas': ['ADV', 'ADJ'], 'Canada': ['NOUN'], 'all': ['ADV', 'DET'], 'caused': ['VERB'], 'lack': ['VERB', 'NOUN'], 'dollar': ['NOUN'], '5\\/8': ['NUM'], 'month': ['NOUN'], 'deadline': ['NOUN'], 'follow': ['VERB'], 'Brown': ['NOUN'], 'to': ['PRT'], 'program': ['NOUN'], '14': ['NUM'], 'PRIME': ['NOUN', 'ADJ'], 'woman': ['NOUN'], 'very': ['ADV', 'ADJ'], 'resistance': ['NOUN'], 'fall': ['VERB', 'NOUN'], '`': ['.'], '--': ['.'], 'list': ['NOUN'], 'joined': ['VERB'], 'large': ['ADJ'], 'small': ['ADJ'], 'past': ['ADV', 'ADJ', 'NOUN', 'ADP'], 'Department': ['NOUN'], 'rate': ['NOUN'], 'further': ['ADV', 'ADJ'], 'East': ['ADJ', 'NOUN'], 'investment': ['NOUN'], 'Paribas': ['NOUN'], 'version': ['NOUN'], 'public': ['NOUN', 'ADJ'], 'movement': ['NOUN'], 'full': ['ADJ'], 'operating': ['VERB', 'NOUN'], 'strong': ['ADV', 'ADJ'], 'ahead': ['ADV'], 'losses': ['NOUN'], 'experience': ['VERB', 'NOUN'], 'advertising': ['VERB', 'NOUN'], 'action': ['NOUN'], 'options': ['NOUN'], 'via': ['ADP'], 'followed': ['VERB'], 'family': ['NOUN'], 'Europe': ['NOUN'], 'shareholders': ['NOUN'], 'takes': ['VERB'], 'two': ['NUM'], 'Corp': ['NOUN'], '6': ['NUM'], 'taken': ['VERB'], 'markets': ['VERB', 'NOUN'], 'more': ['ADV', 'ADJ'], 'Manville': ['NOUN'], 'flat': ['ADJ'], 'Intel': ['NOUN'], 'Sotheby': ['NOUN'], 'company': ['NOUN'], 'quarterly': ['ADJ', 'NOUN'], 'American': ['ADJ', 'NOUN'], 'So': ['ADV', 'ADP'], 'underwriter': ['NOUN'], 'include': ['VERB'], 'remain': ['VERB'], 'Your': ['PRON'], 'Inc': ['NOUN'], 'purchases': ['NOUN'], 'IBM': ['NOUN'], 'division': ['NOUN'], 'share': ['NOUN'], 'states': ['NOUN'], 'sense': ['NOUN'], 'sharp': ['ADJ'], '!': ['.'], 'information': ['NOUN'], 'needs': ['VERB', 'NOUN'], 'court': ['NOUN'], '5.5': ['NUM'], 'earnings': ['NOUN'], 'Hanover': ['NOUN'], 'plant': ['VERB', 'NOUN'], 'plans': ['VERB', 'NOUN'], 'Foreign': ['NOUN', 'ADJ'], 'reflect': ['VERB'], 'coming': ['VERB'], 'Dow': ['NOUN'], 'a': ['X', 'DET'], 'short': ['ADJ'], 'banks': ['NOUN'], 'turnover': ['NOUN'], 'help': ['VERB', 'NOUN'], 'soon': ['ADV'], 'Per-share': ['ADJ'], 'trade': ['VERB', 'NOUN'], 'held': ['VERB'], 'paper': ['NOUN'], 'through': ['ADV', 'ADP'], 'its': ['PRON'], 'Texas': ['NOUN'], '24': ['NUM'], '25': ['NUM'], '27': ['NUM'], 'March': ['NOUN'], '21': ['NUM'], '22': ['NUM'], '23': ['NUM'], '28': ['NUM'], '29': ['NUM'], 'actually': ['ADV'], 'late': ['ADV', 'ADJ'], 'systems': ['NOUN'], 'might': ['VERB', 'NOUN'], 'good': ['ADJ'], 'return': ['VERB', 'NOUN'], 'seeking': ['VERB'], 'food': ['NOUN'], 'always': ['ADV'], 'found': ['VERB'], 'heavy': ['ADJ', 'NOUN'], 'everyone': ['NOUN'], 'house': ['NOUN'], 'energy': ['NOUN'], 'idea': ['NOUN'], 'expect': ['VERB'], 'insurance': ['NOUN'], 'really': ['ADV'], 'since': ['ADP'], 'research': ['NOUN'], '7': ['NUM'], 'issue': ['VERB', 'NOUN'], 'houses': ['VERB', 'NOUN'], 'reason': ['NOUN'], 'base': ['NOUN', 'ADJ'], 'members': ['NOUN'], 'put': ['VERB', 'NOUN'], 'producers': ['NOUN'], 'owners': ['NOUN'], 'benefits': ['NOUN'], 'service': ['NOUN'], 'People': ['NOUN'], 'Boston': ['NOUN'], 'computers': ['NOUN'], 'pilots': ['NOUN'], 'major': ['ADJ'], 'slipped': ['VERB'], 'feel': ['VERB'], 'number': ['NOUN'], 'feet': ['NOUN'], 'done': ['VERB'], 'Earnings': ['NOUN'], 'story': ['NOUN'], 'leading': ['VERB'], 'least': ['ADJ'], 'statement': ['NOUN'], 'selling': ['ADJ', 'VERB', 'NOUN'], 'immediate': ['ADJ'], 'part': ['NOUN'], 'kind': ['NOUN'], 'Officials': ['NOUN'], 'toward': ['ADP'], 'Affairs': ['NOUN'], 'outstanding': ['ADJ'], 'imports': ['NOUN'], 'orders': ['NOUN'], 'sell': ['VERB', 'NOUN'], 'majority': ['NOUN'], 'costs': ['VERB', 'NOUN'], 'Germany': ['NOUN'], 'chairman': ['NOUN'], 'With': ['ADP'], 'added': ['VERB', 'ADJ'], 'Communications': ['NOUN'], 'plan': ['VERB', 'NOUN'], 'significant': ['ADJ'], '70': ['NUM'], 'services': ['NOUN'], 'The': ['NOUN', 'DET'], 'clear': ['ADV', 'VERB', 'ADJ'], 'sometimes': ['ADV'], 'institutions': ['NOUN'], 'sector': ['NOUN'], 'particularly': ['ADV'], 'gold': ['ADJ', 'NOUN'], 'Krenz': ['NOUN'], 'businesses': ['NOUN'], 'find': ['VERB'], 'impact': ['NOUN'], 'merger': ['NOUN'], 'writer': ['NOUN'], 'French': ['ADJ'], 'pretty': ['ADV', 'ADJ'], 'equity': ['NOUN'], '8': ['NUM'], 'his': ['PRON'], 'hit': ['VERB', 'NOUN'], 'gains': ['VERB', 'NOUN'], 'Notes': ['NOUN'], 'financing': ['VERB', 'NOUN'], 'Value': ['NOUN'], 'rest': ['VERB', 'NOUN'], 'closely': ['ADV'], 'during': ['ADP'], 'him': ['PRON'], 'banking': ['VERB', 'NOUN'], 'J.': ['NOUN'], 'common': ['ADJ'], 'activity': ['NOUN'], 'set': ['VERB', 'NOUN'], 'art': ['NOUN'], 'For': ['ADP'], 'Nov.': ['NOUN'], 'declines': ['NOUN'], 'see': ['VERB'], 'defense': ['NOUN'], 'are': ['VERB'], 'close': ['ADV', 'ADJ', 'VERB', 'NOUN'], 'declined': ['VERB'], 'currently': ['ADV'], 'probably': ['ADV'], 'available': ['ADJ'], 'recently': ['ADV'], 'sold': ['VERB'], 'dividend': ['NOUN'], 'AND': ['CONJ'], 'both': ['CONJ', 'DET'], 'last': ['ADV', 'VERB', 'ADJ'], 'annual': ['ADJ'], 'foreign': ['ADJ'], 'became': ['VERB'], 'long-term': ['ADJ'], 'whole': ['ADJ'], 'Yet': ['ADV', 'CONJ'], 'point': ['VERB', 'NOUN'], 'reasons': ['NOUN'], 'simply': ['ADV'], 'decline': ['NOUN'], 'raise': ['VERB', 'NOUN'], 'political': ['ADJ'], 'due': ['ADJ'], 'appears': ['VERB'], 'Most': ['ADV', 'ADJ'], 'California': ['NOUN'], 'meeting': ['VERB', 'NOUN'], 'firm': ['ADJ', 'NOUN'], 'fire': ['NOUN'], 'gas': ['NOUN'], 'else': ['ADV'], 'fund': ['NOUN'], 'lives': ['VERB', 'NOUN'], 'brokers': ['NOUN'], 'bidding': ['NOUN'], 'demand': ['VERB', 'NOUN'], 'prices': ['NOUN'], 'look': ['VERB', 'NOUN'], 'bill': ['NOUN'], 'Sales': ['NOUN'], 'while': ['NOUN', 'ADP'], 'Fees': ['NOUN'], 'Miss': ['NOUN'], 'Issues': ['NOUN'], 'Industries': ['NOUN'], 'City': ['NOUN'], 'cents': ['NOUN'], 'disappointing': ['ADJ'], 'itself': ['PRON'], 'Co.': ['NOUN'], 'widely': ['ADV'], '9': ['NUM'], 'Declining': ['VERB'], 'higher': ['ADV', 'ADJ'], 'development': ['NOUN'], 'About': ['ADP'], 'currencies': ['NOUN'], 'lawyers': ['NOUN'], 'yesterday': ['ADV', 'NOUN'], 'levels': ['NOUN'], 'Activity': ['NOUN'], 'recent': ['ADJ'], 'lower': ['ADV', 'ADJ'], 'using': ['VERB'], 'cut': ['VERB', 'NOUN'], 'also': ['ADV'], 'workers': ['NOUN'], 'location': ['NOUN'], 'Drexel': ['NOUN'], 'guarantees': ['VERB', 'NOUN'], '...': ['.'], 'customers': ['NOUN'], 'Last': ['ADJ'], 'forces': ['VERB', 'NOUN'], 'big': ['ADJ'], 'bid': ['VERB', 'NOUN'], 'Air': ['NOUN'], 'game': ['NOUN'], 'Aug.': ['NOUN'], 'lost': ['VERB'], 'follows': ['VERB'], 'continue': ['VERB'], 'yields': ['NOUN'], 'READY': ['ADJ', 'NOUN'], 'often': ['ADV'], 'some': ['DET'], 'back': ['ADV', 'PRT', 'ADJ', 'NOUN'], '3\\/4': ['NUM'], 'economic': ['ADJ'], 'pricing': ['NOUN'], '3\\/8': ['NUM'], 'Moscow': ['NOUN'], 'Fund': ['NOUN'], 'be': ['VERB'], 'run': ['VERB', 'NOUN'], 'agreement': ['NOUN'], 'David': ['NOUN'], 'step': ['VERB', 'NOUN'], 'by': ['ADP'], 'cautious': ['ADJ'], 'MERRILL': ['NOUN'], 'anything': ['NOUN'], 'most': ['ADV', 'ADJ'], 'range': ['VERB', 'NOUN'], 'Committee': ['NOUN'], 'into': ['ADP'], 'within': ['ADP'], 'nothing': ['NOUN'], 'NEW': ['ADJ', 'NOUN'], 'bankruptcy': ['NOUN'], 'computer': ['NOUN'], 'Composite': ['NOUN'], 'question': ['VERB', 'NOUN'], 'long': ['ADV', 'ADJ'], 'suit': ['NOUN'], ':': ['.'], 'himself': ['PRON'], 'Ms.': ['NOUN'], 'filed': ['VERB'], 'subsidiary': ['NOUN'], 'line': ['NOUN'], 'raising': ['VERB'], 'posted': ['VERB'], 'Office': ['NOUN'], 'Bell': ['NOUN'], 'Africa': ['NOUN'], 'up': ['ADV', 'PRT', 'ADP'], 'us': ['PRON'], "'re": ['VERB'], 'today': ['NOUN'], 'similar': ['ADJ'], 'called': ['VERB'], 'OTC': ['NOUN'], 'To': ['PRT'], 'New': ['ADJ', 'NOUN'], 'rally': ['NOUN'], '%': ['ADJ', 'NOUN'], 'TV': ['NOUN'], 'A.': ['ADJ', 'NOUN'], 'income': ['NOUN'], 'problems': ['NOUN'], 'sides': ['NOUN'], 'rallied': ['VERB'], 'land': ['NOUN'], 'Ad': ['NOUN'], 'vice': ['ADV', 'NOUN'], 'age': ['NOUN'], 'responded': ['VERB'], 'As': ['ADV', 'ADP'], 'At': ['ADP'], 'Soviets': ['NOUN'], 'once': ['ADV', 'ADP'], 'results': ['VERB', 'NOUN'], 'go': ['VERB'], 'issues': ['NOUN'], 'Business': ['NOUN'], 'Bonds': ['NOUN'], 'suits': ['NOUN'], 'UAL': ['NOUN'], 'Not': ['ADV'], 'Now': ['ADV'], 'Mrs.': ['NOUN'], 'continued': ['VERB'], 'entire': ['ADJ'], 'Investors': ['NOUN'], 'Wells': ['NOUN'], 'notes': ['VERB', 'NOUN'], 'fewer': ['ADJ'], 'challenge': ['VERB', 'NOUN'], 'Index': ['NOUN'], 'smaller': ['ADJ'], 'Hutton': ['NOUN'], 'makers': ['NOUN'], 'index': ['NOUN'], 'plays': ['VERB', 'NOUN'], 'power': ['NOUN'], 'access': ['NOUN'], 'waiting': ['VERB'], 'volatile': ['ADJ'], 'capital': ['NOUN'], 'firms': ['NOUN'], 'America': ['NOUN'], 'led': ['VERB'], 'exchange': ['NOUN'], 'commercial': ['NOUN', 'ADJ'], 'growing': ['VERB'], 'others': ['NOUN'], 'great': ['ADJ'], '32': ['NUM'], '31': ['NUM'], '30': ['NUM'], '36': ['NUM'], '35': ['NUM'], 'products': ['NOUN'], 'Motor': ['NOUN'], 'makes': ['VERB'], 'maker': ['NOUN'], 'named': ['VERB'], 'Robert': ['NOUN'], 'private': ['ADJ'], 'scandal': ['NOUN'], 'use': ['VERB', 'NOUN'], 'from': ['ADP'], '&': ['CONJ'], 'remains': ['VERB'], 'next': ['ADV', 'ADJ', 'ADP'], 'few': ['ADJ'], 'themselves': ['PRON'], 'reflects': ['VERB'], 'started': ['VERB'], 'benchmark': ['ADJ', 'NOUN'], 'Wednesday': ['NOUN'], 'customer': ['NOUN'], 'account': ['NOUN'], 'this': ['DET'], 'ride': ['NOUN'], 'clients': ['NOUN'], 'recession': ['NOUN'], 'obvious': ['ADJ'], 'thin': ['ADJ'], 'industrial': ['ADJ'], 'F.': ['NOUN'], 'control': ['VERB', 'NOUN'], 'process': ['NOUN'], '0.3': ['NUM'], 'tax': ['NOUN'], 'high': ['ADV', 'ADJ'], 'Mr.': ['NOUN'], 'something': ['NOUN'], 'employs': ['VERB'], 'six': ['NUM'], 'traders': ['NOUN'], 'LYNCH': ['NOUN'], 'Odds': ['NOUN'], 'stock': ['VERB', 'NOUN'], 'British': ['ADJ', 'NOUN'], 'light': ['ADJ'], 'lines': ['NOUN'], 'One': ['NUM'], 'Oct.': ['NOUN'], 'chief': ['NOUN', 'ADJ'], 'allow': ['VERB'], 'executives': ['NOUN'], 'holds': ['VERB'], 'move': ['VERB', 'NOUN'], 'Smith': ['NOUN'], 'including': ['VERB'], 'looks': ['VERB', 'NOUN'], 'industries': ['NOUN'], 'Exchange': ['NOUN'], 'labor': ['NOUN'], 'pending': ['VERB', 'ADJ'], 'crash': ['VERB', 'NOUN'], 'auto': ['NOUN'], 'practice': ['NOUN'], 'hands': ['NOUN'], 'investor': ['NOUN'], 'day': ['NOUN'], 'Supreme': ['NOUN'], 'San': ['ADJ', 'NOUN'], 'Fed': ['NOUN'], 'John': ['NOUN'], 'doing': ['VERB'], 'Next': ['ADJ', 'NOUN'], 'books': ['NOUN'], 'Treasury': ['NOUN'], 'our': ['PRON'], 'out': ['ADV', 'PRT', 'ADP'], "'": ['PRT', '.'], 'China': ['NOUN'], 'cause': ['VERB', 'NOUN'], 'announced': ['VERB'], 'disclose': ['VERB'], 'This': ['DET'], 'regulators': ['NOUN'], 'Daiwa': ['NOUN'], 'could': ['VERB'], 'times': ['CONJ', 'NOUN'], 'retain': ['VERB'], 'retail': ['ADJ'], 'management': ['NOUN'], 'North': ['ADJ', 'NOUN'], 'data': ['NOUN'], 'system': ['NOUN'], 'their': ['PRON'], 'Stocks': ['NOUN'], 'final': ['ADJ'], 'interests': ['NOUN'], 'Negotiable': ['ADJ'], 'unchanged': ['ADJ'], 'Other': ['NOUN', 'ADJ'], 'have': ['VERB', 'ADJ'], ';': ['.'], 'need': ['VERB', 'NOUN'], 'apparently': ['ADV'], 'clearly': ['ADV'], 'Oil': ['NOUN'], 'Goldman': ['NOUN'], 'agency': ['NOUN'], 'concerns': ['VERB', 'NOUN'], 'eight': ['NUM'], 'segment': ['NOUN'], 'Reserve': ['NOUN'], 'Some': ['DET'], 'face': ['VERB', 'NOUN'], 'Saturday': ['NOUN'], 'fact': ['NOUN'], 'portfolio': ['NOUN'], 'staff': ['NOUN'], 'partners': ['NOUN'], 'based': ['VERB'], 'Meanwhile': ['ADV'], 'should': ['VERB'], 'York': ['NOUN'], 'local': ['ADJ'], 'bonds': ['NOUN'], 'overall': ['ADJ'], 'joint': ['NOUN', 'ADJ'], 'made': ['VERB'], 'words': ['NOUN'], 'timing': ['VERB', 'NOUN'], 'THE': ['DET'], 'Once': ['ADV'], 'ended': ['VERB'], 'Both': ['CONJ', 'DET'], 'she': ['PRON'], 'view': ['VERB', 'NOUN'], 'national': ['ADJ'], 'operates': ['VERB'], 'Source': ['NOUN'], 'RATE': ['NOUN'], 'officials': ['NOUN'], 'changes': ['NOUN'], 'Wisconsin': ['NOUN'], 'reform': ['NOUN'], 'Noriega': ['NOUN'], 'pattern': ['NOUN'], 'Britain': ['NOUN'], 'favor': ['VERB', 'NOUN'], 'state': ['VERB', 'NOUN'], 'closed': ['VERB', 'ADJ'], 'July': ['NOUN'], 'bought': ['VERB'], 'comparable': ['ADJ'], 'a.m': ['ADV'], 'job': ['NOUN'], 'takeover': ['NOUN'], 'key': ['NOUN', 'ADJ'], 'approval': ['NOUN'], 'equal': ['VERB', 'ADJ'], 'drug': ['NOUN'], '1\\/2': ['NUM'], 'figures': ['NOUN'], '1\\/4': ['NUM'], '1\\/8': ['NUM'], 'comment': ['VERB', 'NOUN'], 'Her': ['PRON'], 'ca': ['VERB'], 'Los': ['NOUN'], 'industrials': ['NOUN'], 'addition': ['NOUN'], 'proposal': ['NOUN'], 'Among': ['ADP'], 'Airlines': ['NOUN'], 'finished': ['VERB'], 'CDs': ['NOUN'], 'improved': ['VERB', 'ADJ'], 'an': ['DET'], 'And': ['CONJ'], 'General': ['NOUN'], 'will': ['VERB', 'NOUN'], 'owns': ['VERB'], 'supply': ['NOUN'], 'almost': ['ADV'], 'optimistic': ['ADJ'], 'You': ['PRON'], 'began': ['VERB'], 'administration': ['NOUN'], 'parts': ['NOUN'], 'largest': ['ADJ'], 'units': ['NOUN'], 'effect': ['NOUN'], 'transaction': ['NOUN'], 'off': ['ADV', 'PRT', 'ADP'], 'center': ['NOUN'], 'Senate': ['NOUN'], 'well': ['ADV', 'X', 'ADJ'], 'position': ['NOUN'], 'decliners': ['NOUN'], 'Total': ['ADJ'], 'Shearson': ['NOUN'], 'latest': ['ADJ'], 'stores': ['NOUN'], 'less': ['ADV', 'ADJ'], 'increasingly': ['ADV'], 'executive': ['NOUN', 'ADJ'], 'Chairman': ['NOUN'], 'seats': ['NOUN'], 'Friday': ['NOUN'], 'By': ['ADP'], 'boom': ['NOUN'], 'increased': ['VERB', 'ADJ'], 'government': ['NOUN'], 'increases': ['VERB', 'NOUN'], 'five': ['NUM'], 'know': ['VERB'], 'immediately': ['ADV'], 'loss': ['NOUN'], 'like': ['VERB', 'ADJ', 'ADP'], 'success': ['NOUN'], 'B.': ['NOUN'], 'become': ['VERB'], 'indications': ['NOUN'], 'because': ['ADV', 'ADP'], 'Sun': ['NOUN'], 'growth': ['NOUN'], 'home': ['NOUN'], 'However': ['ADV'], 'does': ['VERB'], 'leader': ['NOUN'], '?': ['.'], 'Futures': ['NOUN'], 'expansion': ['NOUN'], 'pressure': ['NOUN'], 'loans': ['NOUN'], 'gained': ['VERB'], 'about': ['ADV', 'ADP'], 'getting': ['VERB'], 'union': ['NOUN'], 'Americans': ['NOUN'], 'Japan': ['NOUN'], 'software': ['NOUN'], 'ACCOUNT': ['NOUN'], 'own': ['VERB', 'ADJ'], 'Two': ['NUM'], 'quickly': ['ADV'], '1986': ['NUM'], '1987': ['NUM'], '1985': ['NUM'], '1980': ['NUM'], '1988': ['NUM'], '1989': ['NUM'], 'biggest': ['ADJ'], 'buy': ['VERB', 'NOUN'], 'funds': ['NOUN'], 'but': ['CONJ'], 'volume': ['NOUN'], 'construction': ['NOUN'], 'gain': ['VERB', 'NOUN'], 'highest': ['ADJ'], 'he': ['PRON'], 'Industrial': ['ADJ', 'NOUN'], 'Wall': ['NOUN'], 'official': ['ADJ', 'NOUN'], 'record': ['ADJ', 'NOUN'], 'below': ['ADP'], 'problem': ['NOUN'], 'minutes': ['NOUN'], 'Mortgage-Backed': ['NOUN', 'ADJ'], 'compared': ['VERB'], "'ll": ['VERB'], '49': ['NUM'], '45': ['NUM'], '42': ['NUM'], 'Yesterday': ['NOUN'], '40': ['NUM'], 'Volume': ['NOUN'], 'other': ['ADJ'], 'details': ['NOUN'], 'Corp.': ['NOUN'], 'junk': ['NOUN'], 'stay': ['VERB'], 'April': ['NOUN'], 'Morgan': ['NOUN'], 'South': ['ADJ', 'NOUN'], 'Thus': ['ADV'], 'rule': ['VERB', 'NOUN']}
