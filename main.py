# main.py

from voice_input import get_voice_input
import re
import geocoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ytmusicapi import YTMusic
from fuzzywuzzy import process
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import pickle
import os
from collections import Counter

analyzer = SentimentIntensityAnalyzer()
ytmusic = YTMusic()

# Expanded mood keywords
mood_keywords = {
    "depressed": [
        "depressed", "numb", "hopeless", "empty", "lost", "crying", "worthless", "broken", "drowning", "done with life",
        "i feel so low", "nothing matters", "i'm not okay", "want to disappear", "tired of everything",
        "can't take it anymore", "my heart feels heavy", "everything feels dark", "can't feel anything", "life is meaningless",
        "suicidal", "despair", "bleak", "devastated", "forsaken", "inconsolable",
        "miserable", "despondent", "desolate", "rock bottom", "grief-stricken",
        "somber", "defeated", "anguish", "spiritless", "lifeless",
        "soul-crushing", "i've given up", "nothing feels right", "what's the point"
    ],
    "sad": [
        "sad", "blue", "down", "unhappy", "melancholy", "gloomy", "teary", "feeling low", "heavy-hearted", "lonely",
        "i miss someone", "not feeling good", "it's a bad day", "feeling down", "i need a hug",
        "emotional mess", "heartbroken", "i want to cry", "feeling hopeless", "lost in thoughts",
        "upset", "disappointed", "dismal", "hurting", "grieving", "sorrowful",
        "weeping", "mourning", "nostalgic", "regretful", "wistful", "homesick",
        "feeling blue", "disheartened", "troubled", "heartache", "forlorn",
        "bummed out", "feeling grey", "under the weather", "not in a good place"
    ],
    "happy": [
        "happy", "joyful", "cheerful", "excited", "great", "delighted", "thrilled", "on cloud nine", "overjoyed", "smiling",
        "best day ever", "feeling awesome", "everything's perfect", "i'm glowing", "grateful heart",
        "laughing out loud", "can't stop smiling", "loving life", "so pumped", "pure joy",
        "elated", "ecstatic", "blissful", "content", "delightful", "pleasant",
        "jubilant", "gleeful", "merry", "jolly", "carefree", "uplifted",
        "blessed", "on top of the world", "walking on sunshine", "beaming", "radiant"
    ],
    "chill": [
        "chill", "relax", "calm", "soothing", "peaceful", "laid back", "serene", "zen", "cool vibes", "easygoing",
        "winding down", "just chilling", "slow day", "peaceful mind", "need to relax",
        "soft mood", "sunday vibes", "taking it slow", "just breathing", "mellow mood",
        "tranquil", "gentle", "restful", "quiet", "composed", "untroubled",
        "cozy", "comfortable", "at ease", "unwinding", "relaxation time", "me time",
        "mindful", "balanced", "harmony", "stress-free", "no worries", "steady"
    ],
    "hype": [
        "hype", "energetic", "party", "pumped", "upbeat", "excited", "wild", "crazy night", "full power", "let's goooo",
        "turn up", "ready to rock", "dance time", "feeling electric", "get lit",
        "hyped up", "supercharged", "blast the beats", "on fire", "adrenaline rush",
        "turnt", "lit", "amped up", "stoked", "fired up", "ready to party",
        "full throttle", "high energy", "going hard", "feeling alive", "unstoppable",
        "bouncing off walls", "celebration", "festival vibes", "night out", "dancing"
    ],
    "romantic": [
        "romantic", "love", "affection", "crush", "valentine", "passion", "cuddles", "date night", "sweetheart", "heartbeats",
        "thinking of you", "miss my babe", "in love", "sweet vibes", "roses and kisses",
        "i found the one", "love songs", "romance in the air", "sweet memories", "emotional love",
        "infatuated", "smitten", "adore", "cherish", "intimate", "tender",
        "dreamy", "butterflies", "enamored", "loving feelings", "heart eyes",
        "loving", "relationship", "soulmate", "significant other", "true love"
    ],
    "angry": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage", "pissed", "losing it", "fuming", "short-tempered",
        "i'm done", "sick of it", "boiling inside", "mad as hell", "frustrated as ever",
        "nothing's right", "get out of my way", "not today", "i need space", "can't hold back",
        "livid", "seething", "irate", "outraged", "heated", "enraged",
        "bitter", "indignant", "incensed", "steaming", "hot-headed", "inflamed",
        "triggered", "had enough", "exploding", "raging", "hostile", "agitated"
    ],
    "anxious": [
        "anxious", "worried", "nervous", "stressed", "overwhelmed", "panicking",
        "restless", "uneasy", "tense", "frantic", "dreading", "freaking out",
        "overthinking", "on edge", "can't relax", "butterflies", "fearing",
        "apprehensive", "troubled", "agitated", "scared", "frightened",
        "paranoid", "jittery", "jumpy", "distressed", "concerned"
    ],
    "nostalgic": [
        "nostalgic", "memories", "throwback", "good old days", "childhood", "reminiscing",
        "remember when", "missing the past", "retro", "vintage vibes", "time machine",
        "back then", "oldies", "classics", "feeling sentimental", "take me back",
        "flashback", "old school", "golden era", "memory lane", "yearning"
    ],
    "motivated": [
        "motivated", "inspired", "determined", "focused", "productive", "driven",
        "ready to conquer", "ambitious", "goal-oriented", "disciplined", "dedicated",
        "pumped to work", "getting things done", "on a mission", "crushing it",
        "unstoppable", "resilient", "pushing forward", "hustle mode", "grind time",
        "achieving", "success mindset", "winning", "determined"
    ]
}

genre_keywords = {
    "pop": ["pop", "mainstream", "top hits", "popular songs", "radio hits", "billboard", "charts", "trending"],
    "rock": ["rock", "guitar", "bands", "classic rock", "alt rock", "alternative", "punk", "hard rock", "grunge"],
    "hip hop": ["hip hop", "rap", "bars", "beats", "trap", "freestyle", "rhymes", "mc", "flow", "hip-hop"],
    "lofi": ["lofi", "study music", "chill beats", "focus", "low fidelity", "beats to relax", "ambient", "background"],
    "classical": ["classical", "symphony", "orchestra", "beethoven", "mozart", "concerto", "sonata", "piano", "violin"],
    "jazz": ["jazz", "saxophone", "blues", "smooth jazz", "soulful", "trumpet", "bebop", "swing", "improvisation"],
    "edm": ["edm", "electronic", "dance", "club music", "house", "techno", "dubstep", "trance", "drop", "beat"],
    "indie": ["indie", "alternative", "underground", "indie rock", "indie pop", "hipster", "folk", "acoustic"],
    "bollywood": ["bollywood", "indian songs", "hindi music", "desi vibes", "filmy", "hindi songs", "punjabi", "indian pop", "bhangra"],
    "metal": ["metal", "heavy metal", "headbang", "screamo", "thrash", "death metal", "black metal", "hardcore", "metalcore"],
    "rnb": ["rnb", "r&b", "rhythm and blues", "soul", "neo soul", "contemporary r&b", "slow jams"],
    "country": ["country", "western", "nashville", "folk", "americana", "cowboy", "southern"],
    "kpop": ["kpop", "k-pop", "korean pop", "korean", "idols", "bts", "blackpink", "twice"]
}

# ============= KNN Implementation =============

# Create a training dataset for KNN
def create_knn_training_dataset():
    """
    Create a comprehensive training dataset for the KNN mood classifier
    """
    training_data = []

    # Generate examples from mood_keywords
    for mood, keywords in mood_keywords.items():
        # Create example sentences for each mood
        for i in range(10):  # Create 10 examples per mood
            # Select 2-4 random keywords for this mood
            num_keywords = random.randint(2, 4)
            selected_keywords = random.sample(keywords, num_keywords)

            # Create a sentence template
            templates = [
                "I feel {} today.",
                "I'm feeling {} right now.",
                "Today I am {}.",
                "I'm so {} and I can't even explain why.",
                "Everything feels {} at the moment.",
                "I'm in a {} mood.",
                "Feeling {} as always.",
                "Just feeling {} lately.",
                "I've been {} all day.",
                "My mood is really {}."
            ]

            template = random.choice(templates)

            # For single keyword replacement
            if "{}" in template:
                example_text = template.format(" and ".join(selected_keywords))
            else:
                # If no placeholder in template, just add keywords
                example_text = template + " " + " ".join(selected_keywords)

            training_data.append((example_text, mood))

        # Add more complex examples
        if mood == "happy":
            training_data.extend([
                ("I just got a promotion at work, I'm so excited!", mood),
                ("This has been the best day ever, everything is going my way!", mood),
                ("I can't stop smiling after hearing the good news.", mood),
                ("Just won a competition and feeling on top of the world!", mood)
            ])
        elif mood == "sad":
            training_data.extend([
                ("I miss my friends and family so much it hurts.", mood),
                ("Nothing seems to be going right these days.", mood),
                ("I feel so alone even when surrounded by people.", mood),
                ("Lost my pet last week and still feeling heartbroken.", mood)
            ])
        elif mood == "angry":
            training_data.extend([
                ("Can't believe they cancelled my favorite show, this is ridiculous!", mood),
                ("The customer service was terrible and they didn't even apologize.", mood),
                ("My neighbor keeps playing loud music at night and I'm fuming.", mood),
                ("Someone cut me off in traffic today and I almost had an accident.", mood)
            ])
        elif mood == "anxious":
            training_data.extend([
                ("I have a big presentation tomorrow and can't stop worrying about it.", mood),
                ("My mind keeps racing with all the things that could go wrong.", mood),
                ("The uncertainty about my future is keeping me up at night.", mood),
                ("I feel like something bad is about to happen but I don't know what.", mood)
            ])
        elif mood == "chill":
            training_data.extend([
                ("Just relaxing with a good book and some tea.", mood),
                ("Taking it easy today, no rush, no stress.", mood),
                ("Watching the sunset and feeling at peace with everything.", mood),
                ("Going with the flow and enjoying the moment.", mood)
            ])

    # Add some neutral examples
    neutral_examples = [
        ("I'm just going about my day as usual.", "neutral"),
        ("Nothing special happening today.", "neutral"),
        ("Just another regular day for me.", "neutral"),
        ("I don't feel particularly good or bad.", "neutral"),
        ("Everything is normal, nothing exciting.", "neutral"),
        ("It's an ordinary day, nothing to report.", "neutral"),
        ("I feel okay, not great but not bad either.", "neutral"),
        ("Just existing, no strong feelings right now.", "neutral"),
        ("My mood is quite balanced today.", "neutral"),
        ("Neither happy nor sad, just in between.", "neutral")
    ]
    training_data.extend(neutral_examples)

    # Add mixed mood examples with complex sentiment
    mixed_examples = [
        ("I'm happy about my promotion but worried about the new responsibilities.", "anxious"),
        ("Feeling nostalgic but also a bit sad about the past.", "nostalgic"),
        ("Excited about the party tonight but nervous about meeting new people.", "anxious"),
        ("I'm angry they cancelled the event but relieved I don't have to give a speech.", "angry"),
        ("Happy to be on vacation but missing my family at home.", "happy"),
        ("Motivated to start this project but anxious about the deadline.", "motivated"),
        ("Relaxed about the weekend but dreading Monday morning.", "chill"),
        ("Sad about the breakup but hopeful for the future.", "sad"),
        ("Nostalgic for my childhood but grateful for where I am now.", "nostalgic"),
        ("Angry about the argument but proud I kept my composure.", "angry")
    ]
    training_data.extend(mixed_examples)

    return training_data

def extract_knn_features(text):
    """
    Extract numerical features from text for KNN classification
    """
    features = []
    text_lower = text.lower()

    # 1. VADER sentiment scores (4 features)
    vader_scores = analyzer.polarity_scores(text)
    features.extend([vader_scores['pos'], vader_scores['neg'], vader_scores['neu'], vader_scores['compound']])

    # 2. Mood keyword presence (count per mood category) (10 features)
    for mood, keywords in mood_keywords.items():
        # Count keywords from this mood category in the text
        mood_score = sum(1 for keyword in keywords if keyword in text_lower)
        features.append(mood_score)

    # 3. Text statistics (2 features)
    # Word count
    words = text.split()
    word_count = len(words)
    features.append(min(50, word_count) / 50)  # Normalize to 0-1 range, cap at 50

    # Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
        features.append(min(15, avg_word_length) / 15)  # Normalize, cap at 15
    else:
        features.append(0)

    # 4. Punctuation and capitalization (emotional intensity markers) (3 features)
    # Exclamation marks
    exclamation_count = text.count('!')
    features.append(min(5, exclamation_count) / 5)  # Normalize, cap at 5

    # Question marks (uncertainty)
    question_count = text.count('?')
    features.append(min(5, question_count) / 5)  # Normalize, cap at 5

    # ALL CAPS words (intensity)
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    features.append(min(5, caps_words) / 5)  # Normalize, cap at 5

    return features

def train_knn_model(k=5):
    # Check if we have a saved model
    model_path = 'knn_mood_model.pkl'
    features_path = 'knn_features_data.pkl'

    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            # Load saved model and data
            with open(model_path, 'rb') as f:
                knn = pickle.load(f)
            with open(features_path, 'rb') as f:
                X_train, y_train = pickle.load(f)
            print("Loaded saved KNN model")
            return knn, X_train, y_train
        except:
            print("Error loading saved model, training new one")

    # Create new training data
    training_data = create_knn_training_dataset()

    # Extract features from training data
    X_train = []
    y_train = []

    for text, mood in training_data:
        X_train.append(extract_knn_features(text))
        y_train.append(mood)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)

    # Save the model for future use
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        with open(features_path, 'wb') as f:
            pickle.dump((X_train, y_train), f)
        print("Saved KNN model for future use")
    except:
        print("Error saving model, continuing without saving")

    return knn, X_train, y_train

def knn_classify_mood(text, knn_model):
    """
    Classify mood using the trained KNN model
    """
    # Extract features from input text
    features = extract_knn_features(text)

    # Predict mood using KNN model
    mood = knn_model.predict([features])[0]

    # Get probability scores for confidence evaluation
    proba = knn_model.predict_proba([features])[0]
    max_proba = max(proba)

    return mood, max_proba

# ============= Original Mood Detection Functions =============

def extract_time_range(text):
    match = re.search(r'(\d{1,2})\s*(year|years)', text.lower())
    if match:
        years = int(match.group(1))
        if years in [1, 5, 10, 20]:
            return years
    return 1  # default

def get_user_country():
    try:
        g = geocoder.ip('me')
        country = g.country
        if country:
            return country
    except:
        pass
    return 'IN'  # Default to India instead of falling back

def detect_mood_genre_keywords(text):
    text_lower = text.lower()
    mood_scores = {}
    genre_scores = {}

    # Score each mood by counting keyword matches
    for m, keys in mood_keywords.items():
        mood_scores[m] = sum(1 for k in keys if k in text_lower)

    # Score each genre by counting keyword matches
    for g, keys in genre_keywords.items():
        genre_scores[g] = sum(1 for k in keys if k in text_lower)

    # Find mood and genre with highest scores, if any
    mood = max(mood_scores.items(), key=lambda x: x[1])[0] if any(mood_scores.values()) else None
    genre = max(genre_scores.items(), key=lambda x: x[1])[0] if any(genre_scores.values()) else None

    # Only return matches if they scored at least 1
    mood = mood if mood_scores.get(mood, 0) > 0 else None
    genre = genre if genre_scores.get(genre, 0) > 0 else None

    # Return confidence score as well
    mood_confidence = mood_scores.get(mood, 0) / max(1, sum(mood_scores.values())) if mood else 0

    return mood, genre, mood_confidence

def fuzzy_mood_match(text):
    # Create a list of all keywords across all moods
    all_keywords = []
    keyword_to_mood = {}
    for mood, keywords in mood_keywords.items():
        for kw in keywords:
            all_keywords.append(kw)
            keyword_to_mood[kw] = mood

    # Extract words from the input text
    words = text.lower().split()

    # For each word, find the best matching keyword
    matches = []
    for word in words:
        if len(word) > 3:  # Only consider words longer than 3 chars
            match, score = process.extractOne(word, all_keywords)
            if score > 80:  # Only accept good matches
                matches.append((match, keyword_to_mood[match], score))

    # Return the highest scoring mood
    if matches:
        matches.sort(key=lambda x: x[2], reverse=True)
        best_match = matches[0]
        confidence = best_match[2] / 100  # Convert score to confidence (0-1 range)
        return best_match[1], confidence
    return None, 0

def detect_vader_mood(text):
    # Count negation words
    negations = ['not', 'no', "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"]
    has_negation = any(neg in text.lower().split() for neg in negations)

    # Check for intensity modifiers
    intensifiers = ['very', 'extremely', 'so', 'really', 'incredibly', 'absolutely', 'totally']
    has_intensifier = any(intens in text.lower().split() for intens in intensifiers)

    # Get base sentiment scores
    score = analyzer.polarity_scores(text)

    # If there are negations but VADER didn't catch them well
    if has_negation and score['compound'] > 0:
        score['compound'] -= 0.3

    # If there are intensifiers, amplify the sentiment
    if has_intensifier:
        score['compound'] = score['compound'] * 1.5
        # Cap at -1 to 1
        score['compound'] = max(-1, min(1, score['compound']))

    # Determine mood from adjusted scores
    comp = score['compound']

    # Calculate confidence based on compound score magnitude
    confidence = abs(comp)  # Higher magnitude = higher confidence

    if comp >= 0.5:
        return 'happy', score, confidence
    elif comp <= -0.5:
        return 'depressed', score, confidence
    elif 0.2 < comp < 0.5:
        return 'chill', score, confidence
    elif -0.5 < comp < -0.2:
        return 'sad', score, confidence
    else:
        return 'neutral', score, min(0.5, confidence)  # Lower confidence for neutral

# ============= Enhanced Combined Mood Detection =============

def determine_final_mood(text, knn_model=None):
    results = {}
    confidence_scores = {}

    # 1. VADER Analysis
    vader_mood, vader_scores, vader_confidence = detect_vader_mood(text)
    results['vader'] = vader_mood
    confidence_scores['vader'] = vader_confidence

    # 2. Keyword Detection
    keyword_mood, genre, keyword_confidence = detect_mood_genre_keywords(text)
    if keyword_mood:
        results['keyword'] = keyword_mood
        confidence_scores['keyword'] = keyword_confidence

    # 3. Fuzzy Matching
    fuzzy_mood, fuzzy_confidence = fuzzy_mood_match(text)
    if fuzzy_mood:
        results['fuzzy'] = fuzzy_mood
        confidence_scores['fuzzy'] = fuzzy_confidence

    # 4. KNN Classification
    if knn_model:
        knn_mood, knn_confidence = knn_classify_mood(text, knn_model)
        results['knn'] = knn_mood
        confidence_scores['knn'] = knn_confidence

    # Weighted voting system
    mood_votes = {}

    # Method weights (can be adjusted)
    method_weights = {
        'vader': 0.8,  # VADER is reliable for general sentiment
        'keyword': 1.0,  # Direct keyword matches are highly reliable
        'fuzzy': 0.7,  # Fuzzy matching is less reliable
        'knn': 1.0      # KNN has been trained specifically for this task
    }

    # Calculate weighted votes
    for method, mood in results.items():
        if mood:
            # Weight = base method weight * confidence
            weight = method_weights[method] * confidence_scores[method]
            mood_votes[mood] = mood_votes.get(mood, 0) + weight

    # Simple fallback if no moods were detected
    if not mood_votes:
        return "neutral", {"vader": vader_mood}, confidence_scores

    # Find mood with highest weighted votes
    final_mood = max(mood_votes.items(), key=lambda x: x[1])[0]

    # Return the final mood along with all individual results for analysis
    return final_mood, results, confidence_scores

def evaluate_mood_detection_methods(test_cases):
    """
    Evaluate different mood detection methods on a set of test cases
    """
    # Train KNN model
    knn_model, _, _ = train_knn_model(k=5)

    results = {
        'vader': [],
        'keyword': [],
        'fuzzy': [],
        'knn': [],
        'combined': []
    }

    # Process each test case
    for text, expected_mood in test_cases:
        # Get individual method results
        vader_mood, _, _ = detect_vader_mood(text)
        keyword_mood, _, _ = detect_mood_genre_keywords(text)
        fuzzy_mood, _ = fuzzy_mood_match(text)
        knn_mood, _ = knn_classify_mood(text, knn_model)

        # Get combined result
        combined_mood, _, _ = determine_final_mood(text, knn_model)

        # Store results
        results['vader'].append(vader_mood == expected_mood)
        results['keyword'].append(keyword_mood == expected_mood if keyword_mood else False)
        results['fuzzy'].append(fuzzy_mood == expected_mood if fuzzy_mood else False)
        results['knn'].append(knn_mood == expected_mood)
        results['combined'].append(combined_mood == expected_mood)

    # Calculate accuracy for each method
    accuracy = {}
    for method, method_results in results.items():
        accuracy[method] = sum(method_results) / len(method_results) if method_results else 0

    return accuracy
def fetch_playlists(mood, genre, country, years):
    """
    Improved playlist fetching function that uses direct URL construction similar to how
    users would search on YouTube Music.
    """
    playlists = []
    
    # Set defaults if not provided
    country = country or 'IN'  # Default to India
    years = years or 1  # Default to last year
    
    print(f"Finding playlists for: mood={mood}, genre={genre}, country={country}, years={years}")
    
    # Construct search queries in various formats
    search_queries = []
    
    # Primary queries based on available parameters
    if mood and genre:
        search_queries.append(f"{mood} {genre} playlist")
        search_queries.append(f"{genre} for {mood} mood")
    elif mood:
        search_queries.append(f"{mood} music playlist")
        search_queries.append(f"songs for {mood} mood")
    elif genre:
        search_queries.append(f"best {genre} playlist")
    
    # Add time-based modifiers if years specified
    if years and years != 1:
        time_queries = []
        for query in search_queries[:]:  # Use a copy to avoid modifying during iteration
            time_queries.append(f"{query} from last {years} years")
        search_queries.extend(time_queries)
    
    # Add direct YouTube Music search links
    for query in search_queries:
        # Construct a direct search URL with the encoded query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        # Create a playlist entry with direct YouTube Music search URL
        playlist = {
            'title': f"{query.capitalize()}",
            'url': f"https://music.youtube.com/search?q={encoded_query}"
        }
        playlists.append(playlist)
    
    # Add reliable mood-based YouTube Music playlists with their titles
    mood_playlists = {
        'happy': [
            {'title': "Happy Hits!", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_k5SpO3oCR2Z5jvzrsAJnLVUREahKoKGIM"},
            {'title': "Feel Good Hits", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lBNUteBRencHzKelu5iDHwLF6mYqjL2aU"}
        ],
        'sad': [
            {'title': "Sad Songs", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mnBvdP4PsKkZrTj2N3_tujOgPOiGNBs8"},
            {'title': "Feeling Blue", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mz2nqB0kLJp-QB4KtMSVhkGujh-a1pFI"}
        ],
        'depressed': [
            {'title': "Down Tempo", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mThMp3TXBl5Lw2Uu0Xn42VSQuofzLkhs0"},
            {'title': "Melancholy Moods", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_k8GNYFerO5HCf3D1hjOuYQQTizcyETuAQ"}
        ],
        'chill': [
            {'title': "Chill Hits", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lFBMNMlZ9rjGdI9-X_WItDjLFfEJq9o2M"},
            {'title': "Relaxing Music", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mg_Hyn_NC-WTJvrJMK05D911X-ncWNJr8"}
        ],
        'hype': [
            {'title': "Workout Motivation", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_konUtTAKyJX5cGnlmgL7-lPYcgpQmUBxA"},
            {'title': "Energy Booster", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_kOEV19QCQvJdHNQTBlfMK8E3CFQlVOyrU"}
        ],
        'romantic': [
            {'title': "Love Songs", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_msGK0kT4-xQu-EXHjgZ2eqDOWFd8i2UtE"},
            {'title': "Romantic Ballads", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_kuBu2Xp_H1_XjIXTHkiASKr5dQAv_Xf8k"}
        ],
        'angry': [
            {'title': "Rock Anthems", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_nuSqBWPNbk8gOaBj2IzWsNJ5hJJu2JzDM"},
            {'title': "Metal & Hard Rock", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lpTG2Wu5GXJtldKzd3rZQXa0Eb9nLzug"}
        ],
        'anxious': [
            {'title': "Calming Music", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lBNUteBRencHzKelu5iDHwLF6mYqjL2aU"},
            {'title': "Anxiety Relief", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mFkGNr_0nyzJyD-FfK3VHrdmBZD5BzJE4"}
        ],
        'nostalgic': [
            {'title': "Throwback Hits", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lX3rC56NBFfoBYvR6VW-D2v3_JAKcjFw"},
            {'title': "Golden Oldies", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_lKJ1YBtkGNPImEXoRevyQ1mSWded9Iahk"}
        ],
        'motivated': [
            {'title': "Motivation Mix", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_l8eHAJiLN_HnM58qzTwIMN2hHl-70W0Y"},
            {'title': "Success Beats", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_nVEMzZYBGFEtAUoTXnO6o_QCgQFHCkJR4"}
        ],
        'neutral': [
            {'title': "Today's Hits", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_ke-hHPeKhXX3dZSY-XeHD4RZKzx_dwhWA"},
            {'title': "Popular Music", 'url': "https://music.youtube.com/playlist?list=RDCLAK5uy_mrh9kUdOdHggkv_NwmPAtrqVNdnvL83PQ"}
        ]
    }
    
    # Add the pre-configured playlists for the detected mood
    if mood and mood.lower() in mood_playlists:
        playlists.extend(mood_playlists[mood.lower()])
    else:
        # If no specific mood or unrecognized mood, add the neutral playlists
        playlists.extend(mood_playlists['neutral'])
    
    # Add genre-specific playlists if genre is detected
    if genre:
        genre_specific = [
            {'title': f"Best {genre.capitalize()} Songs", 'url': f"https://music.youtube.com/search?q=best+{genre}+songs"},
            {'title': f"Popular {genre.capitalize()} Playlist", 'url': f"https://music.youtube.com/search?q=popular+{genre}+playlist"}
        ]
        playlists.extend(genre_specific)
    
    # Add direct plain YouTube links as a fallback
    for query in search_queries[:2]:  # Just use the first two queries to avoid too many results
        encoded_query = urllib.parse.quote(query)
        playlist = {
            'title': f"{query.capitalize()} (YouTube)",
            'url': f"https://www.youtube.com/results?search_query={encoded_query}+playlist"
        }
        playlists.append(playlist)
    
    print(f"Successfully found {len(playlists)} playlists")
    return playlists
# Test function - run this to make sure the function works correctly
def test_fetch_playlists():
    """
    Test the fetch_playlists function with different mood/genre combinations
    """
    test_cases = [
        ('happy', 'pop', 'US', 1),
        ('sad', None, 'US', 1),
        (None, 'rock', 'US', 1),
        ('angry', None, 'US', 1),
        ('chill', 'lofi', 'US', 1)
    ]
    
    for mood, genre, country, years in test_cases:
        print(f"\n--- Testing: mood={mood}, genre={genre} ---")
        playlists = fetch_playlists(mood, genre, country, years)
        
        if playlists and len(playlists) > 0:
            print(f"✅ Found {len(playlists)} playlists:")
            for i, playlist in enumerate(playlists, 1):
                print(f"  {i}. {playlist['title']}")
                print(f"     {playlist['url']}")
        else:
            print("❌ No playlists found!")
            
    print("\nTesting complete. If you see any issues, check the code.")

def recommend_music():
    """
    Main function that handles voice input, mood detection, and music recommendations
    """
    print("\n🎵 MOOD MUSIC RECOMMENDER 🎵")
    print("Tell me how you're feeling, and I'll suggest some music...")

    # Initialize the KNN model
    knn_model, _, _ = train_knn_model()

    # Get user input (voice or text)
    user_input = get_voice_input()
    if not user_input:
        user_input = input("\nDescribe your mood or what kind of music you want: ")

    print(f"\nProcessing: '{user_input}'")

    # Detect mood using combined approach
    mood, mood_results, confidence_scores = determine_final_mood(user_input, knn_model)

    # Extract genre keywords
    _, genre, _ = detect_mood_genre_keywords(user_input)

    # Get time range preference
    years = extract_time_range(user_input)

    # Get user country for localized recommendations
    country = get_user_country()

    # Show detection results
    print("\n📊 Mood Analysis:")
    print(f"- Your mood appears to be: {mood.upper()}")

    # Show individual method results (for debugging and transparency)
    method_names = {
        'vader': 'Sentiment Analysis',
        'keyword': 'Keyword Detection',
        'fuzzy': 'Fuzzy Matching',
        'knn': 'ML Classification'
    }

    for method, result in mood_results.items():
        if result:
            confidence = confidence_scores.get(method, 0)
            confidence_str = f"{confidence:.0%}" if confidence else "N/A"
            print(f"  • {method_names[method]}: {result.upper()} (confidence: {confidence_str})")

    # Show genre if detected
    if genre:
        print(f"- Music genre preference: {genre.upper()}")

    # Show time range preference
    year_str = "Recent" if years == 1 else f"Last {years} years"
    print(f"- Time range preference: {year_str}")

    # Fetch and display recommended playlists
    print("\nFinding the perfect playlists for you...")
    playlists = fetch_playlists(mood, genre, country, years)
    
    # Display the playlists
    if playlists and len(playlists) > 0:
        print("\n🎵 Recommended Playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"{i}. {playlist['title']}")
            print(f"   Link: {playlist['url']}")
            print()
    else:
        print("\nSorry, couldn't find playlists matching your mood right now.")

    print("\nEnjoy your music! ✨")
    
    return playlists

if __name__ == "__main__":
    recommend_music()