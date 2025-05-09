# AI 텍스트 판별기

## 프로젝트 소개
이 프로젝트는 텍스트가 AI에 의해 작성되었는지 인간에 의해 작성되었는지를 판별하는 도구입니다. 다양한 언어적 특성과 패턴을 분석하여 AI 작성 확률을 예측합니다. 특히 최신 AI가 인간처럼 글을 모방하는 능력을 감지하기 위한 고급 알고리즘을 포함하고 있습니다.

## 주요 기능
- 텍스트 내 문장 다양성, 어휘 다양성, 개인적 표현, 반복 패턴, 감정 표현 다양성 분석
- AI 모방 감지 알고리즘 (AI가 인간처럼 보이려고 시도하는 패턴 감지)
- 간편한 웹 인터페이스로 빠른 텍스트 분석
- TXT, PDF, DOCX, HTML 파일 업로드 및 분석 지원
- 시각적 결과 표시 (확률 게이지 및 특성 점수 차트)

## 설치 방법
1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
```

2. NLTK 데이터 다운로드:
```python
import nltk
nltk.download('punkt')
```

3. 애플리케이션 실행:
```bash
streamlit run app.py
```

## 사용 방법
1. 웹 인터페이스에서 "텍스트 입력" 탭을 선택하고 분석할 텍스트를 입력합니다.
2. "파일 업로드" 탭에서 TXT, PDF, DOCX, HTML 파일을 업로드하여 분석할 수 있습니다.
3. "샘플 텍스트" 탭에서 미리 준비된 샘플로 테스트할 수 있습니다.
4. "분석하기" 버튼을 클릭하여 결과를 확인합니다.

## 결과 해석
- **AI 작성 확률 90% 이상**: 매우 높은 확률로 AI가 작성한 텍스트
- **AI 작성 확률 70-90%**: 대체로 AI가 작성했을 가능성이 높음
- **AI 작성 확률 40-70%**: 판별이 모호함
- **AI 작성 확률 10-40%**: 대체로 인간이 작성했을 가능성이 높음
- **AI 작성 확률 10% 이하**: 매우 높은 확률로 인간이 작성한 텍스트

## 주요 분석 지표
1. **문장 다양성**: 문장 길이, 구조, 시작 단어, 구두점 사용의 다양성
2. **어휘 다양성**: 단어 사용의 다양성, 희귀 단어 사용률, 품사 다양성
3. **개인적 표현**: 1인칭 표현, 감정 묘사, 개인 경험 언급
4. **반복 패턴**: 단어 및 구조의 반복성, 규칙적인 패턴 감지
5. **감정 표현 다양성**: 다양한 감정과 뉘앙스 표현의 사용
6. **AI 모방 점수**: AI가 인간처럼 글을 쓰려고 시도할 때 나타나는 패턴 감지

## AI 모방 감지 기능
최신 버전에서는 다음과 같은 AI 모방 패턴을 감지합니다:
- 지나치게 일관된 감정 표현 사용
- 자연스럽지 못한 격식체-비격식체 전환
- 이모티콘 사용의 한정된 다양성
- 공식적인 이야기 구조 (첫날, 둘째 날 등)
- 지나치게 일관된 문장 구조
- 자연스럽지 못한 감정 변화
- 과도한 구두점이나 이모티콘 사용

## 한계점
- 짧은 텍스트(100단어 미만)는 정확한 판별이 어려울 수 있습니다.
- 특정 형식의 글(예: 법률, 과학 논문 등)은 인간이 작성해도 AI로 판별될 수 있습니다.
- 인간이 AI 스타일을 모방하거나, AI가 인간 스타일을 모방하도록 설계된 경우 결과가 부정확할 수 있습니다.
- 한국어와 영어 텍스트 분석에 최적화되어 있습니다.

## 기여하기
버그 제보나 기능 개선 제안은 이슈를 등록해주세요. 풀 리퀘스트도 환영합니다.

## 라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 주의사항
이 도구는 교육 및 연구 목적으로 개발되었습니다. 결과는 100% 정확하지 않을 수 있으며, 참고용으로만 사용해주세요.
