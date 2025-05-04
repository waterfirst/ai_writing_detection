import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import docx
import io
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# 간단한 문장 분리 함수 정의
def simple_sent_tokenize(text):
    if not text:
        return []
    # 마침표, 물음표, 느낌표로 문장 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 빈 문장 제거
    return [s for s in sentences if s.strip()]

# NLTK의 sent_tokenize 대신 사용
nltk.tokenize.sent_tokenize = simple_sent_tokenize

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
)


# App title and description
st.title("AI Text Detector")
st.markdown("""
This app analyzes the probability that a text was written by AI.  
It uses sophisticated linguistic features and statistical patterns for detection.
""")

# Sidebar settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 75, 5)
detailed_analysis = st.sidebar.checkbox("Show Detailed Analysis", True)

# Feature weight settings (for advanced users)
st.sidebar.subheader("Feature Weights (Advanced)")
show_weights = st.sidebar.checkbox("Show Weight Adjustments", False)

if show_weights:
    sentence_variety_weight = st.sidebar.slider("Sentence Variety Weight", 0.0, 2.0, 1.0, 0.1)
    lexical_diversity_weight = st.sidebar.slider("Lexical Diversity Weight", 0.0, 2.0, 1.0, 0.1)
    personal_references_weight = st.sidebar.slider("Personal Expression Weight", 0.0, 2.0, 1.0, 0.1)
    repetition_weight = st.sidebar.slider("Repetition Pattern Weight", 0.0, 2.0, 1.0, 0.1)
    emotional_variance_weight = st.sidebar.slider("Emotional Expression Diversity Weight", 0.0, 2.0, 1.0, 0.1)
else:
    # Default weights
    sentence_variety_weight = 1.0
    lexical_diversity_weight = 1.0
    personal_references_weight = 1.0
    repetition_weight = 1.0
    emotional_variance_weight = 1.0

# Text file processing function
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return uploaded_file.getvalue().decode('utf-8')
    
    elif file_extension == 'pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return ""
    
    elif file_extension == 'docx':
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing DOCX file: {e}")
            return ""
    
    elif file_extension == 'html':
        try:
            soup = BeautifulSoup(uploaded_file.getvalue(), 'html.parser')
            return soup.get_text()
        except Exception as e:
            st.error(f"Error processing HTML file: {e}")
            return ""
    
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return ""
 
 # Sentence variety analysis
def analyze_sentence_variety(text):
    sentences = sent_tokenize(text)
    
    if not sentences:
        return 0.0, {}
    
    # Analyze sentence length and complexity
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    std_dev = np.std(sentence_lengths)
    
    # Sentence starter diversity
    sentence_starters = {}
    for s in sentences:
        words = s.split()
        if words:
            starter = words[0].lower()
            sentence_starters[starter] = sentence_starters.get(starter, 0) + 1
    
    starter_diversity = len(sentence_starters) / len(sentences)
    
    # Punctuation diversity
    punctuation_types = sum(1 for s in sentences if '?' in s)
    punctuation_types += sum(1 for s in sentences if '!' in s)
    punctuation_types += sum(1 for s in sentences if '...' in s)
    punctuation_diversity = punctuation_types / len(sentences)
    
    # Composite score (higher = less likely to be AI)
    # Higher sentence length standard deviation, more diverse starters, more diverse punctuation = more likely human
    variety_score = (std_dev / avg_length) * 0.5 + starter_diversity * 0.3 + punctuation_diversity * 0.2
    
    # Normalize (0-1 range)
    variety_score = max(0, min(1, variety_score))
    
    details = {
        "Average Sentence Length": round(avg_length, 2),
        "Sentence Length Standard Deviation": round(std_dev, 2),
        "Sentence Starter Diversity": round(starter_diversity, 2),
        "Punctuation Diversity": round(punctuation_diversity, 2)
    }
    
    return variety_score, details

# Lexical diversity analysis
def analyze_lexical_diversity(text):
    words = word_tokenize(text.lower())
    
    if not words:
        return 0.0, {}
    
    # Basic statistics
    total_words = len(words)
    unique_words = len(set(words))
    
    # Type-Token Ratio (TTR)
    ttr = unique_words / total_words
    
    # Rare word ratio
    rare_words = sum(1 for word in set(words) if words.count(word) == 1)
    rare_word_ratio = rare_words / unique_words if unique_words > 0 else 0
    
    # Verb/adjective diversity - support for both Korean and English
    # Korean patterns
    korean_verb_adj_pattern = r'\w+다|\w+는|\w+게|\w+고|\w+며'
    # English patterns
    english_verb_adj_pattern = r'\w+ed|\w+ing|\w+ly|\w+ful|\w+ive|\w+ous|\w+ent|\w+able'
    
    # Try both patterns
    korean_verbs_adjs = re.findall(korean_verb_adj_pattern, text)
    english_verbs_adjs = re.findall(english_verb_adj_pattern, text)
    
    # Combine results
    potential_verbs_adjs = korean_verbs_adjs + english_verbs_adjs
    verb_adj_unique = len(set(potential_verbs_adjs))
    verb_adj_ratio = verb_adj_unique / len(potential_verbs_adjs) if potential_verbs_adjs else 0
    
    # Composite score (higher = less likely to be AI)
    diversity_score = ttr * 0.5 + rare_word_ratio * 0.3 + verb_adj_ratio * 0.2
    
    # Normalize (0-1 range)
    diversity_score = max(0, min(1, diversity_score))
    
    details = {
        "Total Words": total_words,
        "Unique Words": unique_words,
        "Type-Token Ratio (TTR)": round(ttr, 3),
        "Rare Word Ratio": round(rare_word_ratio, 3),
        "Verb/Adjective Diversity": round(verb_adj_ratio, 3)
    }
    
    return diversity_score, details

# Personal expression analysis - significantly improved
def analyze_personal_references(text):
    # First person expressions - both Korean and English
    # Korean
    kr_first_person = re.findall(r'\b(나는|내가|나의|나를|나에게|내|우리|우리의)\b', text)
    # English
    en_first_person = re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', text.lower())
    first_person = kr_first_person + en_first_person
    
    # Personal thought/feeling expressions - both Korean and English
    # Korean - expanded
    kr_personal_thoughts = re.findall(r'(생각|느껴|느낌|기억|추억|경험|생각나|느껴졌|좋았|싫었|행복|슬펐|화났|기뻤|멘붕|자괴감|웃겼|웃긴|아쉬|기대|놀랐)', text)
    # English - expanded
    en_personal_thoughts = re.findall(r'(think|feel|believe|remember|experience|felt|happy|sad|angry|glad|excited|disappointed|surprised|shocked|worried)', text.lower())
    personal_thoughts = kr_personal_thoughts + en_personal_thoughts
    
    # Casual expressions and texting/online speech patterns - heavily weighted
    # Korean - significantly expanded
    kr_casual_expressions = re.findall(r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|그냥|진짜|완전|넘|너무|좀|약간|뭔가|ㅋㅋㅋ|ㅠ|ㅜ|ㅋ|ㅎ|!!|\.{3,}|\?{2,}|~+)', text)
    # English 
    en_casual_expressions = re.findall(r'(lol|haha|wow|oh|just|really|totally|kinda|sort of|like|you know|omg|btw|tbh|idk)', text.lower())
    casual_expressions = kr_casual_expressions + en_casual_expressions
    
    # Emojis and emoticons - fixed pattern
    # Simplified to avoid character range issues
    emojis = re.findall(r'(:\)|:\(|;\)|:D|:P|:O|<3|:3|XD|T_T|>_<)', text)
    # Add a simple emoji counter (without using ranges)
    emoji_count = sum(1 for c in text if ord(c) > 8000)  # Rough check for emoji Unicode characters
    
    # Abbreviations and shortened words - another human indicator
    abbreviations = re.findall(r'\b(gonna|wanna|gotta|dunno|y\'know|u|r|ur|ya|bout|cause|cuz|tho|thru)\b', text.lower())
    
    # Exclamations and emphasis - strong human indicators
    exclamations = re.findall(r'(!{2,}|\?{2,}|\.{3,}|[A-Z]{3,})', text)
    
    # Specific person/place references - both Korean and English
    # Korean - expanded
    kr_specific_references = re.findall(r'(우리|친구|동생|형|누나|오빠|언니|아빠|엄마|선생님|교수님|이름|장소|민지|지은|할머니)', text)
    # English
    en_specific_references = re.findall(r'(friend|brother|sister|mom|dad|teacher|professor|name|place)', text.lower())
    specific_references = kr_specific_references + en_specific_references
    
    # Calculate composite score
    total_words = len(text.split())
    if total_words == 0:
        return 0.0, {}
    
    # New weighted scoring system
    personal_ratio = (len(first_person) + len(personal_thoughts)) / total_words
    casual_ratio = len(casual_expressions) / total_words
    emoji_ratio = (len(emojis) + emoji_count) / total_words * 3.0  # Triple weight for emojis
    abbrev_ratio = len(abbreviations) / total_words * 2.0  # Double weight for abbreviations
    exclamation_ratio = len(exclamations) / total_words * 2.0  # Double weight for exclamations
    specific_ratio = len(specific_references) / total_words
    
    # Highly weighted personal score calculation
    personal_score = (
        personal_ratio * 0.3 + 
        casual_ratio * 0.3 + 
        emoji_ratio * 0.15 + 
        abbrev_ratio * 0.1 + 
        exclamation_ratio * 0.1 + 
        specific_ratio * 0.05
    )
    
    # Normalize (0-1 range) - significantly increased multiplier
    personal_score = max(0, min(1, personal_score * 25))  # Increased multiplier from 15 to 25
    
    details = {
        "First Person Expressions": len(first_person),
        "Personal Thought/Feeling Expressions": len(personal_thoughts),
        "Casual/Texting Expressions": len(casual_expressions),
        "Emojis/Emoticons": len(emojis) + emoji_count,
        "Abbreviations": len(abbreviations),
        "Exclamations/Emphasis": len(exclamations),
        "Specific Person/Place References": len(specific_references),
        "Personal Expression Ratio": round(personal_ratio, 4),
        "Casual Expression Ratio": round(casual_ratio, 4)
    }
    
    return personal_score, details


# Repetition pattern analysis
def analyze_repetition_patterns(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    if not sentences or not words:
        return 0.0, {}
    
    # Sentence structure repetition detection
    sentence_starts = [s.split()[:2] for s in sentences if len(s.split()) >= 2]
    sentence_start_counts = {}
    for start in sentence_starts:
        key = ' '.join(start)
        sentence_start_counts[key] = sentence_start_counts.get(key, 0) + 1
    
    repeated_starts = sum(1 for count in sentence_start_counts.values() if count > 1)
    start_repetition_ratio = repeated_starts / len(sentences) if sentences else 0
    
    # Word repetition patterns
    word_repeat_pattern = {}
    for word in set(words):
        if len(word) > 3:  # Only for meaningful words
            count = words.count(word)
            if count > 1:
                word_repeat_pattern[word] = count
    
    word_repetition_ratio = len(word_repeat_pattern) / len(set(words)) if words else 0
    
    # Clause length patterns
    clause_lengths = []
    for s in sentences:
        clauses = re.split(r'[,;:]', s)
        clause_lengths.extend([len(c.split()) for c in clauses if c.strip()])
    
    if clause_lengths:
        clause_std = np.std(clause_lengths)
        clause_mean = np.mean(clause_lengths)
        clause_variation = clause_std / clause_mean if clause_mean > 0 else 0
    else:
        clause_variation = 0
    
    # Improve the repetition score calculation
    # AI shows more consistent patterns (lower score = higher AI probability)
    repetition_score = (1 - start_repetition_ratio) * 0.4 + (1 - word_repetition_ratio) * 0.4 + clause_variation * 0.2
    
    # Normalize (0-1 range)
    repetition_score = max(0, min(1, repetition_score))
    
    details = {
        "Sentence Start Repetition Ratio": round(start_repetition_ratio, 3),
        "Word Repetition Pattern Ratio": round(word_repetition_ratio, 3),
        "Clause Length Variability": round(clause_variation, 3)
    }
    
    return repetition_score, details

# Emotional expression diversity analysis
def analyze_emotional_variance(text):
    # Emotional expression patterns - both Korean and English
    # Korean
    kr_positive_patterns = [r'좋았|행복|기뻤|재미있|신났|즐거웠|좋은|멋진|훌륭한|아름다운|감동|최고']
    kr_negative_patterns = [r'싫었|슬펐|화났|짜증|힘들|어려웠|나빴|그냥|별로|후회|실망|최악']
    kr_surprised_patterns = [r'놀랐|깜짝|예상치|뜻밖|어머|어이|헐|대박|엄청|굉장']
    kr_nuanced_patterns = [r'미묘|애매|그저|뭔가|어쩌면|아마도|가끔|간혹|다소|약간']
    
    # English
    en_positive_patterns = [r'good|happy|joy|fun|exciting|enjoyed|nice|great|excellent|beautiful|moving|best']
    en_negative_patterns = [r'bad|sad|angry|annoying|difficult|hard|poor|just|not really|regret|disappoint|worst']
    en_surprised_patterns = [r'surprised|shocked|unexpected|wow|oh my|whoa|amazing|awesome|incredible|huge']
    en_nuanced_patterns = [r'subtle|ambiguous|somewhat|somehow|perhaps|maybe|sometimes|occasionally|rather|slightly']
    
    # Combine patterns
    positive_patterns = kr_positive_patterns + en_positive_patterns
    negative_patterns = kr_negative_patterns + en_negative_patterns
    surprised_patterns = kr_surprised_patterns + en_surprised_patterns
    nuanced_patterns = kr_nuanced_patterns + en_nuanced_patterns
    
    # Pattern search
    positive_count = sum(len(re.findall(pattern, text.lower())) for pattern in positive_patterns)
    negative_count = sum(len(re.findall(pattern, text.lower())) for pattern in negative_patterns)
    surprised_count = sum(len(re.findall(pattern, text.lower())) for pattern in surprised_patterns)
    nuanced_count = sum(len(re.findall(pattern, text.lower())) for pattern in nuanced_patterns)
    
    total_emotional = positive_count + negative_count + surprised_count + nuanced_count
    total_words = len(text.split())
    
    if total_words == 0:
        return 0.0, {}
    
    # Emotional expression density
    emotion_density = total_emotional / total_words
    
    # Emotional expression diversity
    emotion_types = sum(1 for count in [positive_count, negative_count, surprised_count, nuanced_count] if count > 0)
    emotion_diversity = emotion_types / 4
    
    # Nuanced expression ratio
    nuance_ratio = nuanced_count / total_emotional if total_emotional > 0 else 0
    
    # Improve the emotion score calculation
    # Composite score (higher = less likely to be AI)
    emotion_score = emotion_density * 0.4 + emotion_diversity * 0.4 + nuance_ratio * 0.2
    
    # Normalize (0-1 range) - increase multiplier to better detect human writing
    emotion_score = max(0, min(1, emotion_score * 7))  # Increased multiplier from 5 to 7
    
    details = {
        "Positive Emotion Expressions": positive_count,
        "Negative Emotion Expressions": negative_count,
        "Surprise Expressions": surprised_count,
        "Nuanced Expressions": nuanced_count,
        "Emotional Expression Density": round(emotion_density, 4),
        "Emotional Expression Diversity": round(emotion_diversity, 2)
    }
    
    return emotion_score, details


# Run complete analysis with enhanced AI mimicry detection
def analyze_text(text):
    if not text.strip():
        return {
            "ai_probability": 0,
            "human_probability": 0,
            "details": {},
            "feature_scores": {}
        }
    
    # Standard analysis
    sentence_score, sentence_details = analyze_sentence_variety(text)
    lexical_score, lexical_details = analyze_lexical_diversity(text)
    personal_score, personal_details = analyze_personal_references(text)
    repetition_score, repetition_details = analyze_repetition_patterns(text)
    emotion_score, emotion_details = analyze_emotional_variance(text)
    
    # NEW: AI mimicry detection
    mimicry_score, mimicry_details, mimicry_signals = detect_ai_mimicry(text)
    
    # Adjust scores based on mimicry detection
    # A high mimicry score increases AI probability even if other human indicators are strong
    mimicry_factor = mimicry_score * 2.0  # Double the impact of mimicry detection
    
    # Check for strong human indicators, but consider mimicry signals
    has_emojis = bool(re.search(r'(:\)|:\(|;\)|:D|:P|:O|<3|:3|XD|T_T|>_<)', text)) or any(ord(c) > 8000 for c in text)
    has_texting_style = bool(re.search(r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|!!|\?{2,}|\.{3,}|~+|lol|omg|wtf)', text))
    has_informal_speech = bool(re.search(r'\b(그냥|진짜|완전|넘|너무|좀|약간|뭔가|gonna|wanna|gotta|dunno|y\'know)\b', text.lower()))
    
    # Apply weights with adjustments for better detection
    weighted_sentence = sentence_score * sentence_variety_weight
    weighted_lexical = lexical_score * lexical_diversity_weight
    weighted_personal = personal_score * personal_references_weight * 1.5  # Reduce from 2.0 to be less susceptible to mimicry
    weighted_repetition = repetition_score * repetition_weight
    weighted_emotion = emotion_score * emotional_variance_weight * 1.2  # Reduce from 1.5
    
    total_weight = (sentence_variety_weight + lexical_diversity_weight + 
                   personal_references_weight * 1.5 + repetition_weight + 
                   emotional_variance_weight * 1.2)
    
    # Calculate human writing probability with strong bias for human indicators
    human_score = (weighted_sentence + weighted_lexical + weighted_personal + 
                  weighted_repetition + weighted_emotion) / total_weight
    
    # Apply bonus for human indicators - but reduced if high mimicry score
    mimicry_reduction = mimicry_score * 0.7  # Reduces the effect of human indicators
    
    if has_emojis:
        human_score += 0.12 * (1 - mimicry_reduction)  # Reduced from 0.15
    if has_texting_style:
        human_score += 0.12 * (1 - mimicry_reduction)  # Reduced from 0.15
    if has_informal_speech:
        human_score += 0.08 * (1 - mimicry_reduction)  # Reduced from 0.10
    
    # Apply mimicry penalty - this directly reduces human score based on mimicry signals
    human_score = max(0, human_score - mimicry_factor)
    
    # Cap at 1.0
    human_score = min(1.0, human_score)
    
    # AI writing probability
    ai_probability = (1 - human_score) * 100
    human_probability = human_score * 100
    
    # Special case handling for obvious AI mimicry
    if mimicry_score > 0.5:
        ai_probability = max(ai_probability, 70)  # Minimum 70% AI probability for high mimicry
        human_probability = 100 - ai_probability
    
    # Special case for formal, structured text that's likely AI
    if personal_score < 0.2 and emotion_score < 0.3 and lexical_score > 0.7:
        ai_probability = max(ai_probability, 75)  # Minimum 75% AI probability
        human_probability = 100 - ai_probability
    
    # Detect overused casual expressions - a sign of AI trying too hard
    casual_expr_pattern = r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ)'
    casual_expr_matches = re.findall(casual_expr_pattern, text)
    casual_expr_count = len(casual_expr_matches)
    casual_expr_varieties = len(set(casual_expr_matches))
    
    # If many casual expressions but few varieties, likely AI mimicry
    if casual_expr_count > 5 and casual_expr_varieties < 3:
        ai_probability = max(ai_probability, 65)
        human_probability = 100 - ai_probability
    
    # Detect perfectly balanced story structure - often a sign of AI
    day_markers = ['첫날', '둘째 날', '셋째 날', '마지막 날']
    found_markers = [marker for marker in day_markers if marker in text]
    if len(found_markers) >= 3:  # Finding 3+ day markers in perfect sequence is suspicious
        ai_probability = max(ai_probability, 60)
        human_probability = 100 - ai_probability
    
    feature_scores = {
        "Sentence Variety": round(sentence_score * 100, 1),
        "Lexical Diversity": round(lexical_score * 100, 1),
        "Personal Expression": round(personal_score * 100, 1),
        "Repetition Patterns": round(repetition_score * 100, 1),
        "Emotional Expression Diversity": round(emotion_score * 100, 1),
        "AI Mimicry Score": round(mimicry_score * 100, 1)  # New score
    }
    
    # Add mimicry signals to details
    mimicry_flags = {k: "Yes" if v else "No" for k, v in mimicry_signals.items()}
    
    return {
        "ai_probability": round(ai_probability, 1),
        "human_probability": round(human_probability, 1),
        "details": {
            "Sentence Variety": sentence_details,
            "Lexical Diversity": lexical_details,
            "Personal Expression": personal_details,
            "Repetition Patterns": repetition_details,
            "Emotional Expression Diversity": emotion_details,
            "AI Mimicry Detection": mimicry_details,
            "Mimicry Warning Flags": mimicry_flags
        },
        "feature_scores": feature_scores
    }


# Enhanced pattern analysis function - detects AI attempting to mimic human writing
def detect_ai_mimicry(text):
    # AI mimicry detection signals
    
    # 1. Overly consistent casual markers - AIs tend to overuse or misuse casual markers
    casual_markers = re.findall(r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|ㅋㅋㅋ|ㅠ|ㅜ|ㅋ|ㅎ)', text)
    # Check distribution and pattern of casual markers
    casual_distribution = []
    current_pos = 0
    for marker in casual_markers:
        pos = text.find(marker, current_pos)
        if pos != -1:
            casual_distribution.append(pos / len(text))  # Normalized position
            current_pos = pos + len(marker)
    
    # Calculate variance in distribution - too uniform distribution is suspicious
    distribution_variance = np.var(casual_distribution) if casual_distribution else 0
    too_uniform_distribution = distribution_variance < 0.03 and len(casual_distribution) > 3
    
    # 2. Check for unnatural transitions between formal and casual language
    formal_segments = re.findall(r'([가-힣\s]{20,}[.!?])', text)  # Longer formal Korean segments
    casual_segments = re.findall(r'([가-힣\s]{5,}[ㅋㅎㅠㅜ]+)', text)  # Segments ending with emoticons
    
    unnatural_transitions = 0
    for i in range(len(casual_segments)):
        for j in range(len(formal_segments)):
            if casual_segments[i] in formal_segments[j] or formal_segments[j] in casual_segments[i]:
                unnatural_transitions += 1
    
    # 3. Analyze emoji/emoticon usage patterns
    emoticon_pattern = re.findall(r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|\^\^|:D|;\))', text)
    emoticon_variety = len(set(emoticon_pattern)) / len(emoticon_pattern) if emoticon_pattern else 0
    limited_emoticon_variety = emoticon_variety < 0.3 and len(emoticon_pattern) > 5
    
    # 4. Check for highly formulaic storytelling patterns common in AI-generated text
    formulaic_patterns = re.findall(r'(첫날|둘째 날|셋째 날|마지막 날)', text)
    formulaic_storytelling = len(formulaic_patterns) >= 3  # Highly structured day-by-day narrative
    
    # 5. Analyze consistency of writing style
    # Extract sentence structures to check for overly consistent patterns
    sentences = sent_tokenize(text)
    sentence_structures = []
    for s in sentences:
        # Simplify to detect structure patterns (starts with subject, verb positions, etc.)
        words = s.split()
        if len(words) > 3:
            # Extract basic structural characteristics
            starts_with_subject = bool(re.match(r'[가-힣]+은|[가-힣]+는|[가-힣]+이|[가-힣]+가|우리|저|나', words[0]))
            has_mid_exclamation = any('!' in w for w in words[1:-1])
            ends_with_emotion = bool(re.search(r'(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|[!?]{2,}|\.{3,})', words[-1])) if words else False
            structure = (starts_with_subject, has_mid_exclamation, ends_with_emotion)
            sentence_structures.append(structure)
    
    # Calculate repeating structure ratio
    structure_counts = {}
    for structure in sentence_structures:
        structure_counts[structure] = structure_counts.get(structure, 0) + 1
    
    most_common_structure_count = max(structure_counts.values()) if structure_counts else 0
    structure_repetition_ratio = most_common_structure_count / len(sentence_structures) if sentence_structures else 0
    overly_consistent_structure = structure_repetition_ratio > 0.4 and len(sentences) > 10
    
    # 6. Look for contextual inconsistencies
    # Check for logical contradictions or inconsistent emotions
    positive_emotive = len(re.findall(r'(좋았|행복|기뻤|즐거웠|좋은|멋진|최고)', text))
    negative_emotive = len(re.findall(r'(싫었|슬펐|화났|짜증|힘들|어려웠|나빴)', text))
    
    # Unusual emotional variance could be sign of AI trying too hard to seem human
    extreme_mood_swings = 0
    for i in range(1, len(sentences)):
        # Check for abrupt shifts between very positive and very negative
        prev_positive = bool(re.search(r'(좋았|행복|기뻤|즐거웠|좋은|멋진|최고)', sentences[i-1]))
        prev_negative = bool(re.search(r'(싫었|슬펐|화났|짜증|힘들|어려웠|나빴)', sentences[i-1]))
        curr_positive = bool(re.search(r'(좋았|행복|기뻤|즐거웠|좋은|멋진|최고)', sentences[i]))
        curr_negative = bool(re.search(r'(싫었|슬펐|화났|짜증|힘들|어려웠|나빴)', sentences[i]))
        
        if (prev_positive and curr_negative) or (prev_negative and curr_positive):
            # Check if there's no transition phrase
            if not re.search(r'(그래도|하지만|그런데|근데|그러나)', sentences[i]):
                extreme_mood_swings += 1
    
    unnatural_mood_changes = extreme_mood_swings > 3
    
    # 7. Calculate excess token patterns
    # AIs often use slightly excessive punctuation or markers when mimicking humans
    excess_punctuation = len(re.findall(r'!{3,}|\?{3,}|\.{4,}', text))
    excess_emoticons = len(re.findall(r'ㅋ{4,}|ㅠ{3,}|ㅜ{3,}|ㅎ{3,}', text))
    
    # Compile AI mimicry signals
    mimicry_signals = {
        "too_uniform_casual_markers": too_uniform_distribution,
        "unnatural_formal_casual_transitions": unnatural_transitions > 2,
        "limited_emoticon_variety": limited_emoticon_variety,
        "formulaic_storytelling": formulaic_storytelling,
        "overly_consistent_structure": overly_consistent_structure,
        "unnatural_mood_changes": unnatural_mood_changes,
        "excess_punctuation": excess_punctuation > 5,
        "excess_emoticons": excess_emoticons > 3
    }
    
    # Calculate mimicry score (higher means more likely to be AI mimicking human)
    mimicry_score = sum(1 for signal in mimicry_signals.values() if signal)
    normalized_mimicry_score = mimicry_score / len(mimicry_signals)
    
    # Detailed analysis data
    details = {
        "Casual Marker Distribution Variance": round(distribution_variance, 4),
        "Unnatural Transitions Count": unnatural_transitions,
        "Emoticon Variety": round(emoticon_variety, 2),
        "Formulaic Pattern Count": len(formulaic_patterns),
        "Structure Repetition Ratio": round(structure_repetition_ratio, 2),
        "Extreme Mood Swings": extreme_mood_swings,
        "Excess Punctuation Count": excess_punctuation,
        "Excess Emoticon Count": excess_emoticons,
        "Mimicry Detection Score": round(normalized_mimicry_score, 2)
    }
    
    return normalized_mimicry_score, details, mimicry_signals


# Results visualization function
def visualize_results(results):
    ai_prob = results["ai_probability"]
    human_prob = results["human_probability"]
    
    # Gauge chart
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0], [100], color='lightgray')
    ax.barh([0], [human_prob], color='#27AE60' if human_prob > 60 else '#FFC300' if human_prob > 40 else '#FF5733')
    
    # Set labels - show human probability instead of AI probability
    ax.text(human_prob + 2, 0, f"{human_prob}%", va='center', 
            color='#27AE60' if human_prob > 60 else '#FFC300' if human_prob > 40 else '#FF5733', 
            fontweight='bold', fontsize=14)
    
    ax.text(0, 0, "Human", va='center', ha='left', fontsize=10)
    ax.text(100, 0, "AI", va='center', ha='right', fontsize=10)
    
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Feature scores chart
    feature_scores = results["feature_scores"]
    features = list(feature_scores.keys())
    scores = list(feature_scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, scores, color=['#27AE60' if s > 60 else '#FFC300' if s > 30 else '#FF5733' for s in scores])
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{scores[i]}%", va='center')
    
    ax.set_xlim(0, 110)
    ax.set_xlabel('Human Characteristics Score (%)')
    ax.set_title('Text Feature Analysis')
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)

# Main app UI
tabs = st.tabs(["Text Input", "File Upload", "Sample Text", "Help"])

with tabs[0]:
    st.header("Text Input")
    input_text = st.text_area("Enter text to analyze:", height=300)
    analyze_button = st.button("Analyze", key="analyze_text")
    
    if analyze_button and input_text:
        with st.spinner("Analyzing text..."):
            results = analyze_text(input_text)
            
            st.subheader("Analysis Results")
            st.markdown(f"""
            ### AI Writing Probability: **{results['ai_probability']}%**
            ### Human Writing Probability: **{results['human_probability']}%**
            """)
            
            visualize_results(results)
            
            if detailed_analysis:
                with st.expander("View Detailed Analysis Results"):
                    for feature, details in results["details"].items():
                        st.subheader(feature)
                        for key, value in details.items():
                            st.write(f"- {key}: {value}")

with tabs[1]:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload File (TXT, PDF, DOCX, HTML)", type=['txt', 'pdf', 'docx', 'html'])
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            file_text = extract_text_from_file(uploaded_file)
            st.subheader("Extracted Text")
            with st.expander("Text Preview"):
                st.write(file_text[:1000] + ("..." if len(file_text) > 1000 else ""))
            
            analyze_file_button = st.button("Analyze File", key="analyze_file")
            
            if analyze_file_button:
                results = analyze_text(file_text)
                
                st.subheader("Analysis Results")
                st.markdown(f"""
                ### AI Writing Probability: **{results['ai_probability']}%**
                ### Human Writing Probability: **{results['human_probability']}%**
                """)
                
                visualize_results(results)
                
                if detailed_analysis:
                    with st.expander("View Detailed Analysis Results"):
                        for feature, details in results["details"].items():
                            st.subheader(feature)
                            for key, value in details.items():
                                st.write(f"- {key}: {value}")
with tabs[2]:
    st.header("Sample Test")
    
    sample_texts = {
        "AI Writing Example": "Jeju Island is a beautiful island in South Korea, popular with many tourists. Jeju Island was formed by volcanic activity, with Mount Halla as a dormant volcano at its center. The natural scenery of Jeju Island is very beautiful. The blue sea, black volcanic rocks, and green meadows create a unique landscape. There are various activities to enjoy on Jeju Island. You can swim at the beaches, climb oreum (volcanic cones), or taste Jeju's unique food. Black pork barbecue, seafood dishes, and Jeju tangerine desserts are must-try foods. Jeju Island is beautiful all year round, but spring and autumn are the best times to visit. During these seasons, the weather is mild and the scenery is most beautiful. Jeju Island is like a treasure of Korea, and if you visit, you will create unforgettable memories.",
        "Human Writing Example": "Finally went to Jeju Island! I was really looking forward to this trip, but it started raining on the first day and I was depressed ㅠㅠ But thanks to our team leader who quickly searched YouTube and found rainy day spots, we had fun anyway. Actually, this is my third time visiting Jeju during rain... I'm used to it now lol. The second day had perfect weather, so we rented a car and explored the west side, which was prettier than expected! Saw the ocean at Hyeopjae Beach and built sandcastles at Gwakji Beach... then my friend's sandcastle was destroyed by the waves and she went crazy😂 We had black pork for lunch since it's famous, but I thought it would taste better than Seoul's samgyeopsal but it was just okay. The price made me want to cry too... But I think it's because we ate at a tourist restaurant. Last day was Seongsan Ilchulbong! Climbing up the many stairs made me huff and puff. I'm usually ready to have a heart attack when climbing stairs, but when I got to the top, wow... it was really amazing. I want to go to Jeju Island again!"
    }
    
    selected_sample = st.selectbox("Select Sample Text", list(sample_texts.keys()))
    st.text_area("Sample Text", value=sample_texts[selected_sample], height=200)
    
    test_sample_button = st.button("Analyze Sample", key="analyze_sample")
    
    if test_sample_button:
        with st.spinner("Analyzing text..."):
            results = analyze_text(sample_texts[selected_sample])
            
            st.subheader("Analysis Results")
            st.markdown(f"""
            ### AI Writing Probability: **{results['ai_probability']}%**
            ### Human Writing Probability: **{results['human_probability']}%**
            """)
            
            visualize_results(results)
            
            if detailed_analysis:
                with st.expander("View Detailed Analysis Results"):
                    for feature, details in results["details"].items():
                        st.subheader(feature)
                        for key, value in details.items():
                            st.write(f"- {key}: {value}")

with tabs[3]:
    st.header("Help & Instructions")
    
    st.markdown("""
    ### What is the AI Text Detector?
    
    This app is a tool that determines whether a text was written by AI or a human. It analyzes various linguistic features to predict the probability of AI authorship.
    
    ### How does it work?
    
    It analyzes the following features:
    
    1. **Sentence Variety**: Diversity in sentence length, structure, starting words, and punctuation
    2. **Lexical Diversity**: Variety in word usage, rare word usage, and parts of speech diversity
    3. **Personal Expression**: First-person expressions, emotional descriptions, and personal experience mentions
    4. **Repetition Patterns**: Word and structural repetition, detection of regular patterns
    5. **Emotional Expression Diversity**: Usage of various emotions and nuanced expressions
    
    ### Interpreting Results
    
    - **AI Writing Probability 90%+**: Very high probability of AI authorship
    - **AI Writing Probability 70-90%**: Likely AI authored
    - **AI Writing Probability 40-70%**: Ambiguous determination
    - **AI Writing Probability 10-40%**: Likely human authored
    - **AI Writing Probability <10%**: Very high probability of human authorship
    
    ### Limitations
    
    - Short texts (less than 100 words) may be difficult to accurately classify.
    - Certain text formats (e.g., legal, scientific papers) may be classified as AI even when written by humans.
    - Results may be inaccurate if humans mimic AI style or AI is designed to mimic human style.
    - While optimized for English, the detector may not perfectly recognize all dialects or writing styles.
    """)

# App description footer
st.markdown("""
---
### AI Text Detector v1.0
This tool was developed for educational and research purposes. Results may not be 100% accurate and should be used for reference only.
""")
