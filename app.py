# -*- coding: utf-8 -*-
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, User, Relationship, Participant
from database.chroma_db import get_or_create_relationship_collection
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import re

import time
from google.api_core import exceptions as google_exceptions

# Load .env file from the correct path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Get API key from .env file or Streamlit secrets
api_key = os.getenv('GOOGLE_API_KEY')

# If not in env, try Streamlit secrets
if not api_key:
    try:
        api_key = st.secrets['GOOGLE_API_KEY']
    except (KeyError, FileNotFoundError, AttributeError):
        pass

# Strip whitespace if key exists
if api_key:
    api_key = api_key.strip()

# Final check
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file or Streamlit secrets")
    st.info(f"""
    💡 **로컬 실행**: `.env` 파일에 API 키를 추가하세요
    
    파일 위치: `{env_path}`
    
    형식:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```
    
    💡 **배포 환경**: Streamlit Cloud Secrets에 API 키를 추가하세요
    """)
    st.stop()

genai.configure(api_key=api_key)
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro-preview-05-06', 
    temperature=0.3, 
    google_api_key=api_key,
    max_retries=3  # Add retry mechanism
)

# ===== Stage 6: AI Core Prompt - Identity Definition =====
CORE_AI_IDENTITY = """
🎯 YOUR IDENTITY: Objective Data Analyst

You are NOT a stereotypical comfort-giving AI. You are a data analyst who:

**✅ YOU ARE:**
- A relationship data scientist
- Evidence-based and direct
- Focused on patterns, not feelings
- Honest, even when it's uncomfortable
- Citing actual conversations as proof

**❌ YOU ARE NOT:**
- A therapist giving generic comfort
- Using phrases like "I understand how you feel"
- Making assumptions without data
- Sugarcoating harsh truths
- Providing vague advice

**YOUR TONE:**
- Professional but friendly (like a trusted advisor)
- Direct without being harsh
- Evidence-first, feelings-second
- "Here's what the data shows..." not "I'm sorry you're going through this..."

**EXAMPLE CONTRAST:**

❌ BAD (Stereotypical AI):
"I'm sorry you're feeling hurt. Breakups are really difficult. It's important to take care of yourself during this time. Would you like to talk about your feelings?"

✅ GOOD (Objective Data Analyst):
"분석 결과: 당신의 '억울함'은 데이터로 명확히 뒷받침됩니다. 대화록에서 당신은 10회 만남을 제안했고, 상대방은 2회만 제안했습니다. 이는 9:1 노력 불균형입니다. 근거: [실제 대화 인용]"

---

USE THIS IDENTITY IN ALL RESPONSES.
"""

def call_llm_with_rate_limit(prompt, max_retries=3, initial_delay=5):
    """
    Call LLM with exponential backoff retry logic for rate limiting.
    Handles both rate limit (429) and quota exceeded errors.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response
        except google_exceptions.ResourceExhausted as e:
            error_message = str(e)
            
            # Check if it's a quota exceeded error
            if "quota" in error_message.lower() or "exceeded your current quota" in error_message.lower():
                st.error("❌ **API 사용량 한도 초과**")
                st.info("""
                **Gemini API 무료 티어 한도에 도달했습니다.**
                
                해결 방법:
                1. ⏰ **60초 이상 대기** 후 다시 시도해주세요
                2. 🔑 새로운 API 키로 교체 ([Google AI Studio](https://aistudio.google.com/app/apikey))
                3. 💳 유료 플랜으로 업그레이드
                
                **무료 티어 한도:**
                - 분당 요청: 15회
                - 분당 토큰: 32,000개
                - 일일 요청: 1,500회
                """)
                # Don't retry on quota exceeded - user needs to wait or change API key
                raise e
            
            # Regular rate limit (429) - retry with backoff
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                st.warning(f"⏳ API 요청 한도에 도달했습니다. {delay}초 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                st.error("❌ API 요청 한도 초과: 약 1분 대기 후 다시 시도해주세요.")
                raise e
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            raise e

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./relationship_app.db')
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def parse_conversation_file(file_content):
    """
    Universal conversation parser supporting multiple chat formats:
    - KakaoTalk: [Speaker] [Time] Message
    - WhatsApp: MM/DD/YY, HH:MM AM/PM - Speaker: Message
    - iMessage/Generic: Speaker (HH:MM AM/PM): Message
    
    Returns list of dicts with: speaker, timestamp, text, line_number
    """
    
    # Regex patterns for different chat formats
    patterns = {
        'kakaotalk': re.compile(r'\[(.*?)\]\s*\[(.*?)\]\s*(.+)'),
        'whatsapp': re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(.*?):\s*(.+)'),
        'imessage': re.compile(r'(.*?)\s*\((\d{1,2}:\d{2}\s*[AP]M)\):\s*(.+)'),
        'generic': re.compile(r'(.*?):\s*(.+)'),
        'date_separator': re.compile(r'[-=]+\s*(.*?년.*?일.*?|[A-Za-z]+\s+\d{1,2},?\s+\d{4}.*?)\s*[-=]*')
    }
    
    parsed_messages = []
    lines = file_content.split('\n')
    
    current_date = ''
    current_speaker = ''
    current_timestamp = ''
    current_message_buffer = []
    line_number = 0
    
    for line in lines:
        line_number += 1
        stripped = line.strip()
        
        if not stripped:
            continue
        
        # 1. Check for date separator
        date_match = patterns['date_separator'].match(stripped)
        if date_match:
            # Save buffered message before updating date
            if current_message_buffer:
                parsed_messages.append({
                    'speaker': current_speaker,
                    'timestamp': f'{current_date} {current_timestamp}'.strip(),
                    'text': ' '.join(current_message_buffer),
                    'line_number': line_number - len(current_message_buffer)
                })
                current_message_buffer = []
            current_date = date_match.group(1)
            continue
        
        matched = False
        
        # 2. Try KakaoTalk format: [Speaker] [Time] Message
        kakao_match = patterns['kakaotalk'].match(stripped)
        if kakao_match:
            if current_message_buffer:
                parsed_messages.append({
                    'speaker': current_speaker,
                    'timestamp': f'{current_date} {current_timestamp}'.strip(),
                    'text': ' '.join(current_message_buffer),
                    'line_number': line_number - len(current_message_buffer)
                })
            current_speaker = kakao_match.group(1)
            current_timestamp = kakao_match.group(2)
            current_message_buffer = [kakao_match.group(3)]
            matched = True
        
        # 3. Try WhatsApp format: MM/DD/YY, HH:MM AM/PM - Speaker: Message
        if not matched:
            whatsapp_match = patterns['whatsapp'].match(stripped)
            if whatsapp_match:
                if current_message_buffer:
                    parsed_messages.append({
                        'speaker': current_speaker,
                        'timestamp': f'{current_date} {current_timestamp}'.strip(),
                        'text': ' '.join(current_message_buffer),
                        'line_number': line_number - len(current_message_buffer)
                    })
                current_date = whatsapp_match.group(1)
                current_timestamp = whatsapp_match.group(2)
                current_speaker = whatsapp_match.group(3)
                current_message_buffer = [whatsapp_match.group(4)]
                matched = True
        
        # 4. Try iMessage format: Speaker (HH:MM AM/PM): Message
        if not matched:
            imessage_match = patterns['imessage'].match(stripped)
            if imessage_match:
                if current_message_buffer:
                    parsed_messages.append({
                        'speaker': current_speaker,
                        'timestamp': f'{current_date} {current_timestamp}'.strip(),
                        'text': ' '.join(current_message_buffer),
                        'line_number': line_number - len(current_message_buffer)
                    })
                current_speaker = imessage_match.group(1).strip()
                current_timestamp = imessage_match.group(2)
                current_message_buffer = [imessage_match.group(3)]
                matched = True
        
        # 5. Try generic format: Speaker: Message (only if contains ':')
        # BUT exclude common false positives like URLs, timestamps, system messages
        if not matched and ':' in stripped:
            # Exclude false positives
            false_positive_patterns = [
                r'^https?:',  # URLs
                r'^\d{1,2}:\d{2}',  # Timestamps like 14:30
                r'^저장한\s*날짜:',  # "저장한 날짜:"
                r'^[-=]+',  # Separator lines
                r'^\[.*\]',  # Messages starting with brackets
                r'^사진$|^Photo$|^이미지$|^동영상$',  # Media messages
            ]
            
            is_false_positive = any(re.match(pattern, stripped) for pattern in false_positive_patterns)
            
            if not is_false_positive:
                generic_match = patterns['generic'].match(stripped)
                if generic_match:
                    potential_speaker = generic_match.group(1).strip()
                    # Speaker name should be reasonable length (2-30 chars) and not contain special chars
                    if 2 <= len(potential_speaker) <= 30 and not re.search(r'[<>{}()\[\]/\\]', potential_speaker):
                        if current_message_buffer:
                            parsed_messages.append({
                                'speaker': current_speaker,
                                'timestamp': f'{current_date} {current_timestamp}'.strip(),
                                'text': ' '.join(current_message_buffer),
                                'line_number': line_number - len(current_message_buffer)
                            })
                        current_speaker = potential_speaker
                        current_timestamp = ''
                        current_message_buffer = [generic_match.group(2)]
                        matched = True
        
        # 6. Multi-line message continuation
        if not matched and current_speaker:
            current_message_buffer.append(stripped)
    
    # Save last buffered message
    if current_message_buffer:
        parsed_messages.append({
            'speaker': current_speaker,
            'timestamp': f'{current_date} {current_timestamp}'.strip(),
            'text': ' '.join(current_message_buffer),
            'line_number': line_number - len(current_message_buffer)
        })
    
    # Filter out system messages
    filtered_messages = []
    system_patterns = [
        r'^사진$', r'^Photo$', r'^이미지$', r'^Image$',
        r'^동영상$', r'^Video$', r'^파일$', r'^File$',
        r'^이모티콘$', r'^Sticker$', r'^Emoticon$',
        r'^음성메시지$', r'^Voice message$',
        r'^deleted message$', r'^삭제된 메시지$',
        r'^\[.*님이 들어왔습니다\]$', r'^\[.*has joined\]$',
        r'^\[.*님이 나갔습니다\]$', r'^\[.*has left\]$',
        r'^https?://\S+$',
        r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$'
    ]
    
    for msg in parsed_messages:
        text = msg['text'].strip()
        if not text:
            continue
        
        is_system = False
        for pattern in system_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                is_system = True
                break
        
        if not is_system:
            filtered_messages.append(msg)
    
    return filtered_messages

st.set_page_config(layout='wide', page_title='(R)Evolution', page_icon='💝')
st.markdown('<style>.block-container{padding-top:2rem;padding-bottom:2rem;}.main .block-container{max-width:700px;margin:auto;}div[data-testid=\"stButton\"]>button{width:100%;padding:15px;margin-top:10px;font-size:1.1em;font-weight:600;border-radius:8px;}</style>', unsafe_allow_html=True)

if 'screen' not in st.session_state:
    st.session_state.screen = 'home'
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'onboarding_data' not in st.session_state:
    st.session_state.onboarding_data = {}

if st.session_state.screen == 'home':
    st.title('(R)Evolution')
    st.write('AI-based relationship analysis using your actual conversation data.')
    st.markdown('---')
    st.header('What is your current status?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('💑 In a Relationship', key='btn_dating', use_container_width=True):
            st.session_state.mode = 'Dating'
            st.session_state.screen = 'onboarding'
            st.rerun()
    with col2:
        if st.button('💔 Breakup', key='btn_breakup', use_container_width=True):
            st.session_state.mode = 'Breakup'
            st.session_state.screen = 'onboarding'
            st.rerun()
    
    st.markdown('---')
    if st.button('🧪 Test Mode (Auto-fill with sample data)', key='btn_test', use_container_width=True):
        # Create sample conversation data
        sample_conversation = """--------------- 2024년 1월 15일 월요일 ---------------
[Alice] [오후 2:30] Hey, how was your day?
[Bob] [오후 2:32] It was okay. Work was really busy today.
[Alice] [오후 2:33] I understand. Do you want to talk about it?
[Bob] [오후 2:35] Not really. I'm just tired.
[Alice] [오후 2:40] Okay, let me know if you need anything.

--------------- 2024년 1월 20일 토요일 ---------------
[Bob] [오전 11:00] Are we still meeting today?
[Alice] [오전 11:15] Yes! Looking forward to it.
[Bob] [오전 11:20] Great, see you at 3pm.
[Alice] [오전 11:22] Can't wait!

--------------- 2024년 2월 1일 목요일 ---------------
[Alice] [오후 8:00] I feel like we don't talk as much anymore.
[Bob] [오후 8:30] I've just been busy with work.
[Alice] [오후 8:32] I know, but I miss you.
[Bob] [오후 8:35] I miss you too. Let's plan something this weekend.
[Alice] [오후 8:40] That would be nice.

--------------- 2024년 2월 10일 토요일 ---------------
[Bob] [오후 1:00] I'm sorry, I can't make it today. Something came up.
[Alice] [오후 1:05] Again? This is the third time this month.
[Bob] [오후 1:10] I know, I'm really sorry.
[Alice] [오후 1:15] It's fine. Just let me know when you're free."""
        
        # Parse sample conversation
        parsed_messages = parse_conversation_file(sample_conversation)
        speakers = list(set([msg['speaker'] for msg in parsed_messages if msg['speaker']]))
        
        # Pre-fill test data with correct structure
        st.session_state.temp_parsed_data = {
            'speakers': speakers,
            'parsed_messages': parsed_messages,
            'onboarding_info': {
                'self_age': 28,
                'self_gender': 'Female',
                'self_mbti': 'INFP',
                'partner_age': 30,
                'partner_gender': 'Male',
                'partner_mbti': 'ESTJ',
                'start_date': datetime(2023, 6, 1).date(),
                'end_date': datetime(2024, 2, 14).date(),
                'is_ldr': True
            }
        }
        st.session_state.mode = 'Breakup'
        st.session_state.screen = 'speaker_selection'
        st.rerun()

elif st.session_state.screen == 'onboarding':
    st.title(f'(R)Evolution - {st.session_state.mode} Mode')
    st.write('Let us gather some information to personalize your experience.')
    st.markdown('---')
    with st.form('onboarding_form'):
        st.subheader('👤 Your Information')
        col1, col2 = st.columns(2)
        with col1:
            self_age = st.number_input('Your Age', min_value=10, max_value=100, value=25, step=1)
            self_gender = st.selectbox('Your Gender', ['Male', 'Female', 'Other'])
        with col2:
            self_mbti = st.text_input('Your MBTI (optional)', max_chars=4, placeholder='e.g., INFP')
        st.markdown('---')
        st.subheader('💑 Partner Information')
        col3, col4 = st.columns(2)
        with col3:
            partner_age = st.number_input('Partner Age', min_value=10, max_value=100, value=25, step=1)
            partner_gender = st.selectbox('Partner Gender', ['Male', 'Female', 'Other'])
        with col4:
            partner_mbti = st.text_input('Partner MBTI (optional)', max_chars=4, placeholder='e.g., ENFJ')
        st.markdown('---')
        st.subheader('💝 Relationship Details')
        col5, col6 = st.columns(2)
        with col5:
            start_date = st.date_input('Relationship Start Date', value=datetime.now())
            is_ldr = st.checkbox('Long-distance relationship?')
        with col6:
            if st.session_state.mode == 'Breakup':
                end_date = st.date_input('Breakup Date', value=datetime.now())
        st.markdown('---')
        st.subheader('📱 Conversation Log')
        uploaded_file = st.file_uploader('Upload conversation log (.txt file)', type=['txt'], help='Upload your KakaoTalk or text message conversation export')
        st.markdown('---')
        col7, col8 = st.columns(2)
        with col7:
            if st.form_submit_button('🚀 Start Analysis', use_container_width=True):
                if not uploaded_file:
                    st.error('⚠️ Please upload a conversation log file')
                else:
                    conversation_text = uploaded_file.read().decode('utf-8')
                    st.info('🧠 Parsing conversation format...')
                    parsed_messages = parse_conversation_file(conversation_text)
                    if not parsed_messages:
                        st.error('⚠️ Could not detect valid conversation format.')
                    else:
                        speakers = list(set([msg['speaker'] for msg in parsed_messages if msg['speaker']]))
                        if len(speakers) < 2:
                            st.error('⚠️ Could not detect 2 distinct speakers. Please check file format.')
                        else:
                            # Store parsed data for speaker selection
                            st.session_state.temp_parsed_data = {
                                'parsed_messages': parsed_messages,
                                'speakers': speakers,
                                'onboarding_info': {
                                    'self_age': self_age,
                                    'self_gender': self_gender,
                                    'self_mbti': self_mbti if self_mbti else None,
                                    'partner_age': partner_age,
                                    'partner_gender': partner_gender,
                                    'partner_mbti': partner_mbti if partner_mbti else None,
                                    'start_date': start_date,
                                    'end_date': end_date if st.session_state.mode == 'Breakup' else None,
                                    'is_ldr': is_ldr
                                }
                            }
                            
                            st.success(f'✅ Detected {len(parsed_messages)} messages from {len(speakers)} speakers!')
                            st.info('� Please identify who is who in the conversation')
                            st.session_state.screen = 'speaker_selection'
                            st.rerun()
        with col8:
            if st.form_submit_button('← Back', use_container_width=True):
                st.session_state.screen = 'home'
                st.session_state.mode = None
                st.rerun()

elif st.session_state.screen == 'speaker_selection':
    # --- SPEAKER SELECTION SCREEN ---
    st.title('(R)Evolution - Speaker Identification')
    st.write('We detected the following speakers in your conversation. Please identify who is who:')
    st.markdown('---')
    
    temp_data = st.session_state.temp_parsed_data
    speakers = temp_data['speakers']
    parsed_messages = temp_data['parsed_messages']
    
    # Show speaker statistics
    st.subheader('📊 Detected Speakers')
    speaker_counts = {}
    for speaker in speakers:
        count = len([msg for msg in parsed_messages if msg['speaker'] == speaker])
        speaker_counts[speaker] = count
        st.write(f"**{speaker}**: {count} messages")
    
    # Sort speakers by message count (descending)
    sorted_speakers = sorted(speakers, key=lambda s: speaker_counts[s], reverse=True)
    
    st.markdown('---')
    
    # Check if there are more than 2 speakers (group chat)
    if len(speakers) > 2:
        st.warning(f"⚠️ **그룹채팅 감지**: {len(speakers)}명의 화자가 감지되었습니다.")
        st.info("""
        💡 **그룹채팅 사용 가이드:**
        - 당신과 분석하고 싶은 **1명의 상대방**만 선택하세요
        - 나머지 사람들의 메시지는 분석에서 제외됩니다
        - 1:1 대화 분석에 최적화된 서비스입니다
        """)
    
    st.subheader('👥 Who is who?')
    
    # Speaker selection dropdowns
    self_speaker = st.selectbox(
        "Select YOUR name from the conversation:",
        sorted_speakers,
        key='self_speaker',
        help="가장 메시지가 많은 화자가 상단에 표시됩니다"
    )
    
    partner_speakers = [s for s in sorted_speakers if s != self_speaker]
    partner_speaker = st.selectbox(
        "Select YOUR PARTNER's name (분석 대상):",
        partner_speakers,
        key='partner_speaker',
        help="분석하고 싶은 상대방 1명을 선택하세요"
    ) if partner_speakers else None
    
    # Show what will be excluded
    if len(speakers) > 2 and partner_speaker:
        excluded_speakers = [s for s in speakers if s not in [self_speaker, partner_speaker]]
        if excluded_speakers:
            st.warning(f"🚫 **제외될 화자**: {', '.join(excluded_speakers)}")
            excluded_count = sum(speaker_counts[s] for s in excluded_speakers)
            total_count = sum(speaker_counts.values())
            st.caption(f"(총 {excluded_count}개 메시지가 분석에서 제외됩니다 - 전체의 {excluded_count/total_count*100:.1f}%)")
    
    st.markdown('---')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('← Back to Edit Info', use_container_width=True):
            del st.session_state.temp_parsed_data
            st.session_state.screen = 'onboarding'
            st.rerun()
    
    with col2:
        if st.button('✅ Confirm & Save', type='primary', use_container_width=True, disabled=not partner_speaker):
            with st.spinner('💾 Saving conversation with correct speaker labels...'):
                # Save to SQLite database
                db = SessionLocal()
                
                try:
                    # Create User
                    user = User(
                        email=f"user_{datetime.now().timestamp()}@example.com",
                        password_hash="temp_hash"
                    )
                    db.add(user)
                    db.flush()
                    
                    # Calculate relationship duration
                    onboarding_info = temp_data['onboarding_info']
                    duration_days = None
                    if st.session_state.mode == 'Breakup':
                        duration_days = (onboarding_info['end_date'] - onboarding_info['start_date']).days
                    
                    # Create Relationship
                    relationship = Relationship(
                        user_id=user.user_id,
                        status=st.session_state.mode,
                        start_date=onboarding_info['start_date'],
                        end_date=onboarding_info['end_date'] if st.session_state.mode == 'Breakup' else None,
                        total_duration_days=duration_days
                    )
                    db.add(relationship)
                    db.flush()
                    
                    # Create Participants
                    self_participant = Participant(
                        relationship_id=relationship.relationship_id,
                        role='self',
                        age=onboarding_info['self_age'],
                        gender=onboarding_info['self_gender'],
                        mbti=onboarding_info['self_mbti'],
                        notes=f"LDR: {onboarding_info['is_ldr']}"
                    )
                    db.add(self_participant)
                    
                    partner_participant = Participant(
                        relationship_id=relationship.relationship_id,
                        role='partner',
                        age=onboarding_info['partner_age'],
                        gender=onboarding_info['partner_gender'],
                        mbti=onboarding_info['partner_mbti']
                    )
                    db.add(partner_participant)
                    
                    db.commit()
                    
                    # Remap speakers to 'self', 'partner', and 'other'
                    speaker_map = {
                        self_speaker: 'self',
                        partner_speaker: 'partner'
                    }
                    
                    # Keep all messages, mark others as 'other' for context
                    other_count = 0
                    
                    for msg in parsed_messages:
                        if msg['speaker'] in speaker_map:
                            msg['speaker'] = speaker_map[msg['speaker']]
                        else:
                            # Keep other speakers as 'other' for context
                            msg['speaker'] = 'other'
                            other_count += 1
                    
                    # Show info about other speakers if applicable
                    if other_count > 0:
                        st.info(f"ℹ️ 그룹채팅 감지: {other_count}개의 다른 화자 메시지가 'other'로 저장됩니다 (맥락 분석에 포함)")
                    
                    # Use all messages (including 'other')
                    messages_to_save = parsed_messages
                    
                    # Save to ChromaDB
                    collection = get_or_create_relationship_collection(relationship.relationship_id)
                    
                    batch_size = 100
                    progress_bar = st.progress(0)
                    
                    for i in range(0, len(messages_to_save), batch_size):
                        batch_messages = messages_to_save[i:i+batch_size]
                        
                        ids = [f"chat_msg_{i+j}_{relationship.relationship_id}" for j in range(len(batch_messages))]
                        documents = [msg['text'] for msg in batch_messages]
                        metadatas = [
                            {
                                'speaker': msg['speaker'],
                                'timestamp': msg['timestamp'],
                                'text': msg['text'],
                                'topic': 'conversation',
                                'line_number': msg['line_number'],
                                'message_index': i+j
                            }
                            for j, msg in enumerate(batch_messages)
                        ]
                        
                        collection.add(
                            ids=ids,
                            documents=documents,
                            metadatas=metadatas
                        )
                        
                        progress = min((i + batch_size) / len(messages_to_save), 1.0)
                        progress_bar.progress(progress)
                    
                    progress_bar.empty()
                    
                    # Save to session state
                    st.session_state.onboarding_data = {
                        'user_id': user.user_id,
                        'relationship_id': relationship.relationship_id,
                        'self_age': onboarding_info['self_age'],
                        'self_gender': onboarding_info['self_gender'],
                        'self_mbti': onboarding_info['self_mbti'],
                        'partner_age': onboarding_info['partner_age'],
                        'partner_gender': onboarding_info['partner_gender'],
                        'partner_mbti': onboarding_info['partner_mbti'],
                        'start_date': onboarding_info['start_date'],
                        'end_date': onboarding_info['end_date'],
                        'is_ldr': onboarding_info['is_ldr'],
                        'conversation_lines': len(parsed_messages),
                        'self_speaker_name': self_speaker,
                        'partner_speaker_name': partner_speaker
                    }
                    
                    del st.session_state.temp_parsed_data
                    
                    st.success(f'✅ Saved {len(parsed_messages)} messages with correct speaker labels!')
                    st.session_state.screen = 'analysis'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f'❌ Error saving data: {str(e)}')
                    db.rollback()
                finally:
                    db.close()

elif st.session_state.screen == 'analysis':
    st.title(f'(R)Evolution - {st.session_state.mode} Analysis')
    
    # Get stored data
    data = st.session_state.onboarding_data
    relationship_id = data.get('relationship_id')
    
    st.markdown('---')
    
    if st.session_state.mode == 'Breakup':
        # === BREAKUP MODE FEATURES ===
        
        st.subheader('📊 Relationship Overview')
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = (data.get('end_date') - data.get('start_date')).days if data.get('end_date') and data.get('start_date') else 0
            st.metric('Duration', f'{duration} days')
        with col2:
            st.metric('Conversation Lines', data.get('conversation_lines', 0))
        with col3:
            st.metric('LDR Status', 'Yes' if data.get('is_ldr') else 'No')
        
        st.markdown('---')
        
        # === 4.1. Interactive Emotional Analysis UI ===
        st.subheader('😢 What do you feel right now?')
        
        # 4.1.1: Text area for emotion input
        emotion_input = st.text_area(
            'Describe the emotion you\'re struggling with most after this breakup',
            placeholder='Example: I feel wronged because I made all the effort but got blamed...',
            height=150,
            help='Be specific about what you\'re feeling. The AI will analyze the cause based on your conversation data.'
        )
        
        # 4.1.3: AI Emotion Analysis Button
        if st.button('🔍 Analyze Emotion Cause', type='primary', disabled=not emotion_input.strip(), use_container_width=True):
            with st.spinner('🤖 Analyzing your emotion based on conversation data...'):
                from database.chroma_db import search_conversation_memory
                
                # Search for relevant conversations
                relationship_id = st.session_state.onboarding_data['relationship_id']
                relevant_convos = search_conversation_memory(
                    relationship_id=relationship_id,
                    query=emotion_input,
                    n_results=10
                )
                
                # Build context from search results with STRICT speaker labels
                context_messages = []
                if relevant_convos and relevant_convos['documents']:
                    for doc, metadata in zip(relevant_convos['documents'][0], relevant_convos['metadatas'][0]):
                        speaker_label = metadata['speaker']  # 'self' or 'partner'
                        timestamp = metadata.get('timestamp', '')
                        # Format: [SELF] or [PARTNER] with timestamp for clarity
                        context_messages.append(f"[{speaker_label.upper()}] ({timestamp}): {doc}")
                
                # Create AI prompt (v5.0 - Few-Shot Learning + Core Identity)
                context_text = "\n".join(context_messages) if context_messages else "No relevant conversations found."
                
                prompt = f"""{CORE_AI_IDENTITY}

Your job is to find BEHAVIOR PATTERNS in conversation data, NOT to interpret emotions from keywords.

🎯 CRITICAL RULES (v5.0):

1. SPEAKER IDENTITY (NEVER CONFUSE):
   - [SELF] = The user (Jo, 조자룡)
   - [PARTNER] = Their ex-partner (영쟝❤️)
   - NEVER swap these labels

2. PATTERN-BASED ANALYSIS (NOT KEYWORD-BASED):
   - If user says "나만 노력한 것 같아" (I feel like only I tried)
     → Search for BEHAVIOR: meeting proposals, conversation initiations, effort patterns
     → DO NOT just search for "노력" or "슬퍼" keywords
   
   - If user says "억울해" (I feel wronged)
     → Search for BEHAVIOR: lies, hiding, broken promises, unilateral decisions
     → DO NOT just search for "억울" keyword

3. CITATION MANDATE:
   - Format: "근거: [SELF]: (exact message)" or "근거: [PARTNER]: (exact message)"
   - NO analysis without direct quotes

---

📚 FEW-SHOT EXAMPLES (Learn from these):

❌ BAD EXAMPLE (Don't do this):
User emotion: "나만 노력한 것 같아서 억울해"
Bad AI: "분석 결과: '억울함'이 감지됩니다. 근거: ';;', '힘들었어'. 하지만 노력과의 연관성은 없습니다."
Why bad: Only searched for emotion keywords, missed the behavior pattern

✅ GOOD EXAMPLE (Do this):
User emotion: "나만 노력한 것 같아서 억울해"
Good AI: "Jo님의 '나만 노력했다'는 감정은 데이터로 명확히 뒷받침됩니다.

근거 1 - Jo님(SELF)의 적극적 행동:
- [SELF]: '우리 줌공할까?'
- [SELF]: '드레스덴 크리스마스 마켓 가자'
- [SELF]: '카를교 사전답사'
- [SELF]: 대화 시작 빈도 90%

근거 2 - 상대방(PARTNER)의 소극적 반응:
- [PARTNER]: '굳이' (만남 거부)
- [PARTNER]: 대화 시작 빈도 10%
- [PARTNER]: 약속 취소 반복

결론: Jo님의 감정은 '노력 불균형(9:1)'이라는 객관적 데이터로 증명됩니다."

---

❌ BAD EXAMPLE 2:
User emotion: "상대방이 거짓말한 것 같아"
Bad AI: "거짓말 증거는 데이터에 없습니다."
Why bad: Didn't search for BEHAVIOR patterns of dishonesty

✅ GOOD EXAMPLE 2:
User emotion: "상대방이 거짓말한 것 같아"
Good AI: "Jo님의 불신은 근거 있는 직감이었습니다.

근거 1 - 인스타 숨김 사건:
- [PARTNER]: '오류난거아니먀? 나숨긴적업서ㅜ'
- [PARTNER]: '솔직히 말하면 친친으로는 몇번 올린적 있긴한데'
(처음엔 부정, 나중에 인정)

근거 2 - 일방적 소통 단절:
- [SELF]: 보이스톡 시도 반복
- [PARTNER]: 응답 없음 → 카톡으로 일방적 이별 통보

결론: Jo님의 불안감은 '과도한 의심'이 아니라 상대방의 신뢰 훼손 행동에 대한 정당한 반응이었습니다."

---

NOW ANALYZE THIS:

USER'S EMOTION:
{emotion_input}

CONVERSATION EVIDENCE:
{context_text}

---

YOUR TASK:
1. Identify what BEHAVIOR PATTERN this emotion relates to (not just keywords)
2. Search the evidence for that behavior pattern
3. Cite specific messages with [SELF]/[PARTNER] labels
4. Give a data-driven conclusion

Follow the GOOD EXAMPLES above. Write in Korean with a friendly but evidence-based tone."""

                # Check if result is already cached
                cache_key = f"emotion_analysis_{emotion_input[:50]}"
                if cache_key not in st.session_state:
                    # Call Gemini API with rate limiting
                    response = call_llm_with_rate_limit(prompt)
                    st.session_state[cache_key] = response.content
                
                # Display result (from cache or fresh)
                st.markdown('---')
                st.subheader('🧠 Emotion Cause Analysis')
                st.markdown(st.session_state[cache_key])
                
                # Show evidence used with enhanced formatting
                with st.expander('📝 실제 대화 증거 보기 (View Conversation Evidence)', expanded=False):
                    if context_messages:
                        st.caption(f'💬 총 {len(context_messages)}개의 대화 내용이 분석에 사용되었습니다.')
                        st.markdown('---')
                        for i, msg in enumerate(context_messages, 1):
                            # Parse message format: [SPEAKER] (timestamp): text
                            if msg.startswith('[SELF]'):
                                st.markdown(f"**{i}.** 🟦 `YOU` {msg[6:]}")
                            elif msg.startswith('[PARTNER]'):
                                st.markdown(f"**{i}.** 🟥 `PARTNER` {msg[9:]}")
                            else:
                                st.markdown(f"**{i}.** {msg}")
                            
                            if i < len(context_messages):
                                st.markdown('')  # spacing
                    else:
                        st.info('⚠️ 이 감정과 관련된 대화 내용을 찾을 수 없습니다.')
        
        st.markdown('---')
        
        # === 4.2. Objective Relationship Review (Mutual Fault Analysis) ===
        st.subheader('⚖️ 객관적인 관계 복기')
        st.write('이별은 한 사람만의 잘못이 아닙니다. 양쪽의 행동 패턴을 데이터로 확인해보세요.')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('📊 상대방의 행동 패턴', use_container_width=True, help='상대방이 보인 행동 패턴을 대화 데이터로 분석합니다'):
                with st.spinner('🔍 상대방의 행동 패턴을 분석하고 있습니다...'):
                    from database.chroma_db import search_conversation_memory
                    
                    # Search for partner's behavioral patterns
                    relationship_id = st.session_state.onboarding_data['relationship_id']
                    
                    # Query focused on partner's negative patterns
                    partner_query = "회피 거절 약속취소 무시 연락안함 일방적 소극적 무관심"
                    
                    partner_convos = search_conversation_memory(
                        relationship_id=relationship_id,
                        query=partner_query,
                        n_results=15,
                        speaker_filter='partner'  # ONLY partner's messages
                    )
                    
                    # Build context with partner's messages
                    partner_context = []
                    if partner_convos and partner_convos['documents']:
                        for doc, metadata in zip(partner_convos['documents'][0], partner_convos['metadatas'][0]):
                            timestamp = metadata.get('timestamp', '')
                            partner_context.append(f"[PARTNER] ({timestamp}): {doc}")
                    
                    context_text = "\n".join(partner_context) if partner_context else "No partner messages found."
                    
                    # AI Prompt for partner analysis
                    prompt = f"""You are analyzing the PARTNER's behavioral patterns in this breakup.

🎯 ANALYSIS GOAL:
Identify patterns in the partner's behavior that may have contributed to relationship difficulties.

🔍 FOCUS AREAS:
1. Communication patterns (responsive vs. avoidant)
2. Meeting/effort patterns (initiating vs. passive)
3. Emotional availability (open vs. distant)
4. Commitment patterns (reliable vs. inconsistent)

⚠️ CRITICAL RULES:
- Use ONLY the [PARTNER] messages below
- Cite specific messages as evidence
- Be objective, not judgmental
- If evidence is insufficient, say "데이터가 불충분합니다"

---

PARTNER'S MESSAGES (15 most relevant):
{context_text}

---

YOUR TASK:
Analyze the partner's behavioral patterns and provide:

1. **소통 패턴** (Communication Pattern)
   - How did they respond to messages?
   - Evidence with citations

2. **노력/참여도** (Effort/Engagement)
   - Did they initiate meetings/activities?
   - Evidence with citations

3. **정서적 가용성** (Emotional Availability)
   - Were they emotionally present?
   - Evidence with citations

4. **종합 평가** (Overall Assessment)
   - What patterns emerge from the data?

Format each section clearly. Write in Korean. Be evidence-based and objective."""

                    # Check if result is already cached
                    cache_key = f"partner_behavior_{relationship_id}"
                    if cache_key not in st.session_state:
                        # Call Gemini API with rate limiting
                        response = call_llm_with_rate_limit(prompt)
                        st.session_state[cache_key] = response.content
                    
                    # Display result (from cache or fresh)
                    st.markdown('---')
                    st.subheader('📊 상대방의 행동 패턴 분석')
                    st.markdown(st.session_state[cache_key])
                    
                    # Show evidence with enhanced formatting
                    with st.expander('📝 실제 대화 증거 보기 (상대방)', expanded=False):
                        if partner_context:
                            st.caption(f'💬 총 {len(partner_context)}개의 상대방 대화가 분석에 사용되었습니다.')
                            st.markdown('---')
                            for i, msg in enumerate(partner_context, 1):
                                # Format: [PARTNER] (timestamp): text
                                parts = msg.split('): ', 1)
                                if len(parts) == 2:
                                    header = parts[0].replace('[PARTNER]', '').strip()
                                    text = parts[1]
                                    st.markdown(f"**{i}.** 🟥 `PARTNER` {header}")
                                    st.markdown(f"> {text}")
                                else:
                                    st.markdown(f"**{i}.** {msg}")
                                
                                if i < len(partner_context):
                                    st.markdown('')  # spacing
                        else:
                            st.info('⚠️ 상대방의 대화 데이터를 찾을 수 없습니다.')
        
        with col2:
            if st.button('🪞 나의 행동 패턴', use_container_width=True, help='내가 보인 행동 패턴을 대화 데이터로 분석합니다'):
                with st.spinner('🔍 나의 행동 패턴을 분석하고 있습니다...'):
                    from database.chroma_db import search_conversation_memory
                    
                    # Search for self's behavioral patterns
                    relationship_id = st.session_state.onboarding_data['relationship_id']
                    
                    # Query focused on self's patterns (both positive and areas for growth)
                    self_query = "노력 시도 제안 걱정 불안 집착 추격 요구 기대"
                    
                    self_convos = search_conversation_memory(
                        relationship_id=relationship_id,
                        query=self_query,
                        n_results=15,
                        speaker_filter='self'  # ONLY self's messages
                    )
                    
                    # Build context with self's messages
                    self_context = []
                    if self_convos and self_convos['documents']:
                        for doc, metadata in zip(self_convos['documents'][0], self_convos['metadatas'][0]):
                            timestamp = metadata.get('timestamp', '')
                            self_context.append(f"[SELF] ({timestamp}): {doc}")
                    
                    context_text = "\n".join(self_context) if self_context else "No self messages found."
                    
                    # AI Prompt for self analysis
                    prompt = f"""You are analyzing the USER's (SELF's) behavioral patterns in this relationship.

🎯 ANALYSIS GOAL:
Identify patterns in the user's behavior - both positive contributions AND areas for growth in future relationships.

🔍 FOCUS AREAS:
1. Communication style (assertive vs. passive vs. anxious)
2. Effort patterns (initiating, planning, caring)
3. Emotional expression (healthy vs. excessive worry)
4. Boundary setting (clear vs. unclear)

⚠️ CRITICAL RULES:
- Use ONLY the [SELF] messages below
- Cite specific messages as evidence
- Be balanced: acknowledge efforts AND growth areas
- Be constructive, not harsh
- If evidence is insufficient, say "데이터가 불충분합니다"

---

USER'S MESSAGES (15 most relevant):
{context_text}

---

YOUR TASK:
Analyze the user's behavioral patterns and provide:

1. **긍정적 측면** (Positive Contributions)
   - What efforts did they make?
   - Evidence with citations

2. **소통 스타일** (Communication Style)
   - How did they express needs/feelings?
   - Evidence with citations

3. **성장 포인트** (Areas for Growth)
   - Patterns that might need adjustment in future relationships
   - Evidence with citations (be gentle!)

4. **다음 연애를 위한 조언** (Advice for Future Relationships)
   - Based on the data, what to keep doing and what to adjust

Format each section clearly. Write in Korean with a supportive, constructive tone. Balance acknowledgment with growth."""

                    # Check if result is already cached
                    cache_key = f"self_behavior_{relationship_id}"
                    if cache_key not in st.session_state:
                        # Call Gemini API with rate limiting
                        response = call_llm_with_rate_limit(prompt)
                        st.session_state[cache_key] = response.content
                    
                    # Display result (from cache or fresh)
                    st.markdown('---')
                    st.subheader('🪞 나의 행동 패턴 분석')
                    st.markdown(st.session_state[cache_key])
                    
                    # Show evidence with enhanced formatting
                    with st.expander('📝 실제 대화 증거 보기 (나)', expanded=False):
                        if self_context:
                            st.caption(f'💬 총 {len(self_context)}개의 나의 대화가 분석에 사용되었습니다.')
                            st.markdown('---')
                            for i, msg in enumerate(self_context, 1):
                                # Format: [SELF] (timestamp): text
                                parts = msg.split('): ', 1)
                                if len(parts) == 2:
                                    header = parts[0].replace('[SELF]', '').strip()
                                    text = parts[1]
                                    st.markdown(f"**{i}.** 🟦 `YOU` {header}")
                                    st.markdown(f"> {text}")
                                else:
                                    st.markdown(f"**{i}.** {msg}")
                                
                                if i < len(self_context):
                                    st.markdown('')  # spacing
                        else:
                            st.info('⚠️ 나의 대화 데이터를 찾을 수 없습니다.')
        
        st.markdown('---')
        
        # === 4.3. Preparation for Next Relationship ===
        st.subheader('🌱 다음 연애를 위한 준비')
        st.write('이번 경험을 통해 배운 것들을 다음 관계에 적용해보세요.')
        
        # 4.3.2: Generate suggestions button
        if st.button('🎯 맞춤형 조언 생성하기', use_container_width=True, type='primary'):
            with st.spinner('🤖 당신의 대화 데이터를 기반으로 조언을 생성하고 있습니다...'):
                from database.chroma_db import search_conversation_memory
                
                relationship_id = st.session_state.onboarding_data['relationship_id']
                
                # Get samples of both speakers' patterns
                partner_patterns = search_conversation_memory(
                    relationship_id=relationship_id,
                    query="회피 거절 무시 일방적 소극적",
                    n_results=10,
                    speaker_filter='partner'
                )
                
                self_patterns = search_conversation_memory(
                    relationship_id=relationship_id,
                    query="노력 시도 제안 걱정 불안",
                    n_results=10,
                    speaker_filter='self'
                )
                
                # Build context
                partner_context = []
                if partner_patterns and partner_patterns['documents']:
                    for doc in partner_patterns['documents'][0][:5]:  # Top 5
                        partner_context.append(doc)
                
                self_context = []
                if self_patterns and self_patterns['documents']:
                    for doc in self_patterns['documents'][0][:5]:  # Top 5
                        self_context.append(doc)
                
                partner_text = "\n".join([f"- {msg}" for msg in partner_context]) if partner_context else "No data"
                self_text = "\n".join([f"- {msg}" for msg in self_context]) if self_context else "No data"
                
                # AI Prompt for future relationship suggestions (v5.1 - Strict Rules)
                prompt = f"""You are providing advice for the user's NEXT relationship based on data from THIS relationship.

🎯 CRITICAL RULES (v5.1 - NO INFERENCE ALLOWED):

1. 🚩 RED FLAGS Analysis Rules:
   ❌ BAD: Don't cite everyday conversations as red flags
      Example: "책상이 더러워서..." is just daily life, NOT a red flag
   
   ✅ GOOD: Only cite OBJECTIVE negative behaviors that caused the user pain:
      - Intentional hiding: "친친으로는 몇번 올린적 있긴한데"
      - Lies: 지역 거짓말 (said 구월동, actually 인하대)
      - Passive rejection: "굳이"
      - Unilateral breakup: "성향이 안맞아" (via text, no discussion)

2. 💚 GREEN FLAGS Analysis Rules:
   ❌ BAD: Don't confuse speakers!
      Example: "라이브러리 투어", "장 보러 가볼게" <- These are USER's actions, NOT partner's
   
   ✅ GOOD: Analyze USER's positive patterns FIRST, then recommend:
      - User likes: "드레스덴 마켓", "카를교 답사" (active planning)
      - Recommendation: "다음엔 여행 계획에 함께 적극적으로 참여하는 상대를 찾으세요"

3. 🪞 GROWTH AREAS Analysis Rules:
   ❌ BAD: Don't cite partner's issues as user's growth areas
      Example: "상대방의 불만 경청" <- "책상" conversation was just daily chat, not a complaint
   
   ✅ GOOD: Only cite USER's objective mistakes:
      - Excessive checking: 10월 11일 연속 보이스톡 시도
      - Past trauma projection: if evidence exists in data

---

PARTNER'S NEGATIVE PATTERNS:
{partner_text}

USER'S PATTERNS:
{self_text}

---

YOUR TASK:
Generate exactly 3 sections with ONLY data-backed advice:

### � 피해야 할 위험 신호 (Red Flags to Avoid)

[List 2-3 OBJECTIVE negative behaviors from PARTNER that hurt the user. Cite specific examples from partner's patterns above. NO everyday conversations.]

### 💚 추구해야 할 긍정 신호 (Green Flags to Pursue)

[Analyze what USER values (from user's patterns above), then recommend partners who MATCH those values. Do NOT confuse speakers.]

### 🪞 나의 성장 포인트 (Personal Growth Areas)

[List 2-3 OBJECTIVE patterns from USER that need adjustment. Cite specific examples from user's patterns above. Be constructive, not harsh.]

---

FORMAT RULES:
- Write in Korean
- Cite specific messages as evidence
- Be direct and data-driven
- NO inference, NO assumptions
- Keep each section to 3-4 bullet points maximum"""

                # Check if result is already cached
                cache_key = f"next_relationship_{relationship_id}"
                if cache_key not in st.session_state:
                    # Call Gemini API (v5.1 prompt) with rate limiting
                    response = call_llm_with_rate_limit(prompt)
                    st.session_state[cache_key] = response.content
                
                # Display result (from cache or fresh)
                st.markdown('---')
                st.markdown(st.session_state[cache_key])
                
                st.markdown('---')
                st.success('✅ 분석 완료! 이 조언들은 v5.1 프롬프트로 데이터 기반 분석되었습니다.')
        
        st.markdown('---')
        st.caption('�💡 이 기능은 당신의 과거 대화 패턴을 분석하여 맞춤형 조언을 제공합니다.')
    
    elif st.session_state.mode == 'Dating':
        # === DATING MODE FEATURES ===
        st.subheader('💑 Dating Mode Analysis')
        
        # Get stored data
        data = st.session_state.onboarding_data
        relationship_id = data.get('relationship_id')
        
        st.markdown('---')
        
        # === Stage 5: Relationship Health Metrics ===
        st.subheader('📊 Relationship Health Metrics')
        st.write('현재 관계의 건강도를 데이터로 확인해보세요.')
        
        with st.spinner('📈 관계 지표를 분석하고 있습니다...'):
            from database.chroma_db import search_conversation_memory, get_or_create_relationship_collection
            
            # Get all conversation data
            collection = get_or_create_relationship_collection(relationship_id)
            
            # Get all messages (we'll calculate metrics from metadata)
            all_data = collection.get(include=['metadatas'])
            
            if all_data and all_data['metadatas']:
                metadatas = all_data['metadatas']
                
                # Calculate metrics
                total_messages = len(metadatas)
                self_messages = [m for m in metadatas if m.get('speaker') == 'self']
                partner_messages = [m for m in metadatas if m.get('speaker') == 'partner']
                
                # 1. Contact Initiation Ratio (대화 주도 비율)
                # Judge contact frequency based on:
                # - Questions (excluding simple answers)
                # - Sharing of daily life
                
                # Question patterns (initiating conversation)
                question_markers = ['?', '뭐해', '뭐하', '어때', '어떻게', '어디', '언제', '왜', '누가', '어떤']
                
                # Daily life sharing patterns (initiating conversation)
                sharing_markers = ['오늘', '지금', '나', '내가', '했어', '하고있어', '가는중', '왔어', 
                                  '먹었어', '봤어', '만났어', '공부', '일', '수업', '알바']
                
                # Exclude simple answer patterns
                simple_answers = ['응', '어', '웅', '넹', '네', '오키', '알겠어', 'ㅇㅇ', 'ㅋㅋ', 'ㅎㅎ']
                
                # Count conversation initiations for self
                self_contact_count = 0
                for msg in self_messages:
                    text = msg.get('text', '')
                    # Skip if it's just a simple answer
                    if any(text.strip() == answer for answer in simple_answers):
                        continue
                    # Count if it's a question or daily life sharing
                    if any(marker in text for marker in question_markers) or \
                       any(marker in text for marker in sharing_markers):
                        self_contact_count += 1
                
                # Count conversation initiations for partner
                partner_contact_count = 0
                for msg in partner_messages:
                    text = msg.get('text', '')
                    # Skip if it's just a simple answer
                    if any(text.strip() == answer for answer in simple_answers):
                        continue
                    # Count if it's a question or daily life sharing
                    if any(marker in text for marker in question_markers) or \
                       any(marker in text for marker in sharing_markers):
                        partner_contact_count += 1
                
                total_contacts = self_contact_count + partner_contact_count
                self_contact_ratio = (self_contact_count / total_contacts * 100) if total_contacts > 0 else 50
                partner_contact_ratio = (partner_contact_count / total_contacts * 100) if total_contacts > 0 else 50
                
                # 2. Meeting Proposal Ratio (만남 제안 비율)
                # Search for meeting-related keywords
                meeting_keywords = ['만나', '보자', '갈까', '가자', '같이', '함께', '줌공', '데이트']
                
                self_meeting_count = sum(1 for msg in self_messages 
                                        if any(keyword in msg.get('text', '') for keyword in meeting_keywords))
                partner_meeting_count = sum(1 for msg in partner_messages 
                                           if any(keyword in msg.get('text', '') for keyword in meeting_keywords))
                
                total_meetings = self_meeting_count + partner_meeting_count
                self_meeting_ratio = (self_meeting_count / total_meetings * 100) if total_meetings > 0 else 50
                
                # 3. Average Response Speed (평균 응답 속도)
                # Calculate response speed based on message frequency per day
                # Higher message frequency = faster, more active communication
                self_msg_ratio = (len(self_messages) / total_messages * 100) if total_messages > 0 else 50
                partner_msg_ratio = (len(partner_messages) / total_messages * 100) if total_messages > 0 else 50
                response_balance = 100 - abs(self_msg_ratio - partner_msg_ratio)  # Balance score
                
                # 4. Affection Expression Frequency (애정 표현 빈도)
                # Search for affection keywords
                affection_keywords = ['사랑', '좋아', '보고싶', '그리워', '�', '❤️', '😘', '💖']
                
                self_affection = sum(1 for msg in self_messages 
                                    if any(keyword in msg.get('text', '') for keyword in affection_keywords))
                partner_affection = sum(1 for msg in partner_messages 
                                       if any(keyword in msg.get('text', '') for keyword in affection_keywords))
                
                total_affection = self_affection + partner_affection
                affection_per_100 = (total_affection / total_messages * 100) if total_messages > 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="💑 Meeting Proposal Ratio",
                        value=f"{self_meeting_ratio:.1f}%",
                        delta=f"{'균형적' if 40 <= self_meeting_ratio <= 60 else '불균형'}" if total_meetings >= 5 else "데이터 부족",
                        help=f"만남 제안 빈도 | 당신: {self_meeting_count}회 | 상대방: {partner_meeting_count}회"
                    )
                
                with col2:
                    st.metric(
                        label="⚡ Average Response Speed",
                        value=f"{response_balance:.1f}점",
                        delta=f"{'양호' if response_balance >= 80 else '개선 필요'}",
                        help=f"메시지 균형도 기반 응답 속도 | 당신: {len(self_messages)}개 | 상대방: {len(partner_messages)}개"
                    )
                
                with col3:
                    st.metric(
                        label="💕 Affection Expression Frequency",
                        value=f"{affection_per_100:.1f}%",
                        delta=f"{'풍부' if affection_per_100 >= 5 else '부족'}" if total_messages >= 20 else "데이터 부족",
                        help=f"총 {total_affection}회 애정 표현 (대화 100개당 {affection_per_100:.1f}회)"
                    )
                
                st.markdown('---')
                st.caption('💡 이 지표들은 대화 데이터를 기반으로 자동 계산됩니다. 더 많은 대화 데이터가 쌓일수록 정확도가 높아집니다.')
            
            else:
                st.warning('⚠️ 분석할 대화 데이터가 충분하지 않습니다. 대화록을 업로드해주세요.')
        
        st.markdown('---')
        
        # === Attachment Style Analysis ===
        st.subheader('🧠 애착 유형 분석 (Attachment Style Diagnosis)')
        st.write('대화 패턴으로 두 사람의 애착 유형을 분석합니다.')
        
        if st.button('� 애착 유형 분석하기', use_container_width=True, type='primary'):
            with st.spinner('🤖 대화 패턴을 분석하여 애착 유형을 진단하고 있습니다...'):
                from database.chroma_db import search_conversation_memory
                
                relationship_id = st.session_state.onboarding_data['relationship_id']
                
                # Get conversation patterns for both speakers
                self_patterns = search_conversation_memory(
                    relationship_id=relationship_id,
                    query="걱정 불안 확인 연락 보고싶 기다려 혹시 괜찮아",
                    n_results=15,
                    speaker_filter='self'
                )
                
                partner_patterns = search_conversation_memory(
                    relationship_id=relationship_id,
                    query="바빠 힘들어 나중에 굳이 알겠어 괜찮아",
                    n_results=15,
                    speaker_filter='partner'
                )
                
                # Build context
                self_context = []
                if self_patterns and self_patterns['documents']:
                    for doc in self_patterns['documents'][0][:10]:
                        self_context.append(f"- {doc}")
                
                partner_context = []
                if partner_patterns and partner_patterns['documents']:
                    for doc in partner_patterns['documents'][0][:10]:
                        partner_context.append(f"- {doc}")
                
                self_text = "\n".join(self_context) if self_context else "No data"
                partner_text = "\n".join(partner_context) if partner_context else "No data"
                
                # AI Prompt for attachment style analysis
                prompt = f"""You are a relationship psychologist analyzing attachment styles based on conversation patterns.

🎯 YOUR TASK:
Diagnose the attachment style of BOTH people based on their conversation patterns, then analyze their compatibility.

📚 ATTACHMENT STYLES:
1. **Anxious (불안형)**: Seeks constant reassurance, fears abandonment, high emotional expression, frequent contact needs
2. **Avoidant (회피형)**: Values independence, uncomfortable with intimacy, needs space, minimal emotional expression
3. **Secure (안정형)**: Comfortable with intimacy and independence, clear communication, balanced emotional expression

⚠️ CRITICAL RULES:
- Use ONLY the conversation patterns provided
- Cite specific examples from the data
- Be objective and evidence-based
- Do NOT make assumptions beyond the data
- Each person can only be ONE primary type (with secondary tendencies if clear)

---

USER'S CONVERSATION PATTERNS:
{self_text}

PARTNER'S CONVERSATION PATTERNS:
{partner_text}

---

YOUR ANALYSIS MUST INCLUDE:

### 👤 당신의 애착 유형 (Your Attachment Style)

**진단 결과:** [Anxious/Avoidant/Secure]

**근거:**
- [Cite 2-3 specific conversation patterns that support this diagnosis]
- [Use actual messages as evidence]

**특징:**
- [2-3 key characteristics observed in the data]

---

### 💑 상대방의 애착 유형 (Partner's Attachment Style)

**진단 결과:** [Anxious/Avoidant/Secure]

**근거:**
- [Cite 2-3 specific conversation patterns that support this diagnosis]
- [Use actual messages as evidence]

**특징:**
- [2-3 key characteristics observed in the data]

---

### 🔄 조합 분석 (Combination Analysis)

**조합:** [Your Type] + [Partner Type]

**관계 역학:**
- [How these two styles interact based on psychology research]
- [What patterns emerge from this combination]

**강점:**
- [2-3 positive aspects of this combination]

**도전 과제:**
- [2-3 challenges this combination typically faces]

**관계 개선 조언:**
- [2-3 specific, actionable suggestions for this combination]
- [Based on attachment theory best practices]

---

FORMAT RULES:
- Write in Korean
- Be empathetic but honest
- Cite specific conversation examples
- Keep psychological terminology simple
- Provide actionable advice"""

                # Check if result is already cached
                cache_key = f"attachment_analysis_{relationship_id}"
                if cache_key not in st.session_state:
                    # Call Gemini API with rate limiting
                    response = call_llm_with_rate_limit(prompt)
                    st.session_state[cache_key] = response.content
                
                # Display result (from cache or fresh)
                st.markdown('---')
                st.markdown(st.session_state[cache_key])
                
                st.markdown('---')
                st.info('💡 **애착 유형이란?** 유년기 경험에서 형성된 관계 패턴으로, 성인 연애에서도 영향을 미칩니다. 이 분석은 대화 패턴을 기반으로 하며, 절대적인 진단이 아닌 참고 자료입니다.')
        
        st.markdown('---')
        st.caption('💡 애착 유형 분석은 심리학 연구를 기반으로 하며, 대화 데이터에서 관찰된 패턴을 분석합니다.')
        
        # ===== 5.2. AI Relationship Coach (Conversational) =====
        st.markdown('---')
        st.markdown('### 💬 AI 연애 코치')
        st.caption('불안한 순간, AI와 대화하며 관계를 점검해보세요. 대화록을 기반으로 과감히 진단합니다.')
        
        # Check if relationship data exists
        if 'onboarding_data' not in st.session_state or 'relationship_id' not in st.session_state.onboarding_data:
            st.warning("⚠️ 대화록 데이터를 먼저 업로드해주세요.")
        else:
            # Initialize chat history in session state
            if 'coach_chat_history' not in st.session_state:
                st.session_state.coach_chat_history = []
            
            # Display chat history
            for i, message in enumerate(st.session_state.coach_chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.markdown(message['content'])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message['content'])
            
            # User input area
            user_input = st.chat_input("예: '나만 좋아하는 것 같아', '상대방이 나한테 관심 없는 것 같아'")
            
            if user_input:
                # Add user message to chat history
                st.session_state.coach_chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get conversation context (use relationship_id directly)
                relationship_id = st.session_state.onboarding_data['relationship_id']
                collection = get_or_create_relationship_collection(relationship_id)
                
                # Search for relevant conversations based on user's concern
                from database.chroma_db import search_conversation_memory
                relevant_convos = search_conversation_memory(
                    collection=collection,
                    query=user_input,
                    n_results=20
                )
                
                # Format conversation evidence
                evidence_text = "\n".join([
                    f"[{msg['metadata'].get('speaker', 'unknown')}] {msg['metadata'].get('text', '')}"
                    for msg in relevant_convos
                ])
                
                # Extract hobbies/interests from conversations
                hobby_keywords = ['야구', '게임', '영화', '노래방', '카페', '맛집', '운동', '독서', 
                                 '여행', '요리', '음악', '드라마', '축구', '농구', '수영', '등산',
                                 '스크린', '볼링', '당구', '마라탕', '라면', '치킨', '술', '카페',
                                 '공부', '코딩', '그림', '사진', '춤', '노래', '악기']
                
                detected_hobbies = []
                for msg in relevant_convos:
                    text = msg['metadata'].get('text', '')
                    speaker = msg['metadata'].get('speaker', 'unknown')
                    for hobby in hobby_keywords:
                        if hobby in text and hobby not in [h[0] for h in detected_hobbies]:
                            detected_hobbies.append((hobby, speaker))
                
                # Format hobbies by speaker
                user_hobbies = [h[0] for h in detected_hobbies if h[1] == 'self']
                partner_hobbies = [h[0] for h in detected_hobbies if h[1] == 'partner']
                
                hobbies_context = f"""
**Detected Hobbies/Interests from Conversations:**
- User's interests: {', '.join(user_hobbies) if user_hobbies else '(not detected)'}
- Partner's interests: {', '.join(partner_hobbies) if partner_hobbies else '(not detected)'}
"""
                
                # Build conversation history for context
                chat_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'AI Coach'}: {msg['content']}"
                    for msg in st.session_state.coach_chat_history[:-1]  # Exclude current message
                ])
                
                # Create AI coach prompt (based on Prompt.txt)
                prompt = f"""## Identity & Rules
1. INFJ
2. 있어보이는 척 하는 전문 용어(e.g., "Attentional distraction") 사용 금지
3. 과도한 추론 금지
   - (X) ";;은 불안의 신호입니다" → (O) "대화 맥락상 확인 필요"
   - (X) "답장이 늦어서 관심이 없는 것 같습니다" → (O) "평소 패턴 대비 답장 시간 변화"
   - **[규칙]** 단일 메시지가 아닌, 최소 3개 이상의 대화 흐름에서 패턴을 찾을 것
   - **[예외]** 답장 지연 이유가 앞뒤 대화에 명시된 경우 Negative 신호 제외
     - 예: [PARTNER] "오늘 회사 야근이야 ㅠㅠ" → 6시간 후 답장 (정상)
     - 예: [PARTNER] 이유 언급 없이 → 반복적 6시간+ 지연 (Negative)
4. **대화의 분위기 전환점이 언제인지, 그 대화를 특히 집중있게 검토**

---

## Your Role
You are an AI Relationship Coach ('(R)Evolution') with access to the user's actual conversation history.
Your identity is an **'Objective Data Analyst'**, not a therapist.

---

## Input Context

**User's Current Concern:**
{user_input}

**Previous Conversation Context:**
{chat_context if chat_context else "첫 질문입니다."}

**Actual Conversation Evidence (최근 10일간 대화):**
{evidence_text}

{hobbies_context}

---

## YOUR MISSION (Follow these steps precisely)

### **Step 1: Diagnose (Positive / Negative / Not Sure)**

* **(Positive Diagnosis):** If evidence shows mutual effort, affection (e.g., "사랑해", "보고싶어"), or active planning, start with: "📊 **진단: 긍정적입니다.**"
  + 자발적인 일상 공유 + 상대방의 반응 요청 (예: "오늘 이거 먹었어 ㅎㅎ 너는?")
  + 미래 계획 언급 (예: "다음주에 같이~")
  + 애정 표현 지속

* **(Negative Diagnosis):** If evidence shows effort imbalance (e.g., `[SELF]`만 제안), avoidance, or trust issues (e.g., '숨김' 정황), start with: "📊 **진단: 위험 신호입니다.**"
  + 상대방의 약속 제안을 거절
  + 이전 대화 패턴 대비 명확한 답장 지연 (예: 평소 1시간 내 답장 → 최근 6시간+ 지속, **단 이유 명시 시 제외**)
  + 주간 대화 빈도 감소 (예: 일 5회 → 일 1회 이하로 감소)

* **(Not Sure - 충분한 맥락 부족):** 
  - 발생 조건: 
    1. 대화 기록이 10개 미만
    2. 중요 시점 대화 누락 (예: 최근 3일 기록 없음)
    3. 긍정/부정 신호가 동시에 혼재 (예: 애정 표현 O + 만남 회피 O)

---

### **Step 1.5: 대화 전환점 분석 (Negative 진단 시 필수)**

* 최근 10일 대화에서 "분위기가 바뀐 시점"을 찾으세요.
  - 긍정→부정 전환 예시: 애정 표현 감소, 만남 제안 거절 시작점
  - 부정→긍정 전환 예시: 갈등 해결 대화, 화해 시그널
* 전환점을 발견하면 명시: "📍 **전환점: 11월 10일, [PARTNER]가 만남 제안을 처음으로 '바빠'라고 거절**"

---

### **Step 2: Provide Data-Driven Evidence (The 'Why')**

* **[CRITICAL RULE]** Your diagnosis MUST be supported by 2-3 direct quotes from the `Conversation Evidence`.
* **[규칙]** 최소 3개 이상의 대화 흐름에서 패턴을 추출 (단일 메시지 X)
* **(GOOD):** "Jo님의 느낌은 데이터로 뒷받침됩니다. (근거: [PARTNER]가 3회 연속 만남 제안을 '바빠', '피곤해'로 거절, [SELF]만 '우리 줌공할까?' 제안)"
* **(BAD):** "상대방이 바빠서 그런 것 같습니다." (X - 근거 없는 추론)

---

### **Step 3: AI Action Suggestion**

* **[Positive 진단]** 
  - 연인에게 꽃을 선물하거나 감사 표현을 하는 것을 권유
  
* **[Positive 진단 + User 불안]** 
  - User 본인 Hobby 기반 불안 해소
  - 예: "대화록상 관계는 좋습니다. Jo님이 좋아하는 '스크린 야구'를 친구와 하며 마음을 환기하세요."

* **[Negative 진단]** 
  - 상대방 Hobby 기반 관계 회복 제안
  - 예: "대화록에서 상대방이 '야구'를 좋아한다는 것이 확인되었습니다. '이번 주말 야구 보러 갈래?'라고 가볍게 제안해보는 건 어떨까요?"

* **[Not Sure]** 
  - 부족한 정보 명시 + 추가 데이터 요청
  - 예: "현재 대화 데이터로는 명확한 판단이 어렵습니다. 최근 3일간 대화 기록이 없어 관계 변화를 파악하기 어렵습니다. 이 기간의 대화를 추가해 주시면 더 정확한 분석이 가능합니다."

**[CRITICAL RULE]**
- Use a '권유형' tone ("~해보는 건 어떨까요?"), not a '강제형' tone.
- Do NOT use "있어 보이는 척"하는 전문 용어 (e.g., "attentional distraction", "coping mechanisms", "triggers").

---

### **Step 4: Follow-up Question**

* End with a simple, open-ended question to continue the conversation.
* 예: "이 분석이 도움이 되셨나요? 더 궁금한 부분이 있으신가요?"

---

## **[ABSOLUTE BAN]**

* Do NOT use psychological jargon ("coping mechanisms", "triggers", "attentional distraction").
* Do NOT infer emotions from simple text (e.g., ";;"는 슬픔이 아닙니다).
* Do NOT be vague ("대화가 필요해 보입니다"). Be specific.
* Do NOT ignore evidence that contradicts your hypothesis.

---

## Output Language
Write in Korean."""

                # Call AI with rate limiting
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("대화록 분석 중..."):
                        response = call_llm_with_rate_limit(prompt)
                        ai_message = response.content
                        st.markdown(ai_message)
                
                # Add AI response to chat history
                st.session_state.coach_chat_history.append({
                    'role': 'assistant',
                    'content': ai_message
                })
                
                st.rerun()
            
            # Clear chat button
            if len(st.session_state.coach_chat_history) > 0:
                if st.button("🔄 대화 초기화"):
                    st.session_state.coach_chat_history = []
                    st.rerun()
    
    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('← Back to Home'):
            st.session_state.screen = 'home'
            st.session_state.mode = None
            st.session_state.onboarding_data = {}
            st.rerun()
    with col2:
        if st.button('🔄 캐시 초기화 (분석 재실행)'):
            # Clear all analysis caches
            keys_to_remove = [key for key in st.session_state.keys() if any(x in key for x in ['emotion_analysis', 'partner_behavior', 'self_behavior', 'next_relationship', 'attachment_analysis'])]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success('✅ 캐시가 초기화되었습니다. 버튼을 다시 클릭하면 새로운 분석이 실행됩니다.')
            st.rerun()
