---
name: presentation-design
description: 프레젠테이션 디자인과 콘텐츠 생성을 위한 skill. 사용자가 (1) 다양한 포맷의 원천 데이터를 제공하고 슬라이드 디자인을 요청하거나, (2) 이미 작성된 슬라이드의 스타일 변경을 요청하거나, (3) 주제와 목차만 제공하여 슬라이드 작성을 요청할 때 사용. 기존 pptx skill을 호출하되, 전문적인 스타일과 톤을 자동으로 적용합니다.
---

# Presentation Design Skill

이 skill은 **기존 pptx skill의 강력한 기능을 활용**하면서, 전문적인 디자인 스타일과 톤을 자동으로 선택하고 적용합니다.

## 핵심 원칙

1. **기존 pptx skill 재사용**: pptx 생성, QA, 변환은 `/mnt/skills/public/pptx/SKILL.md` 참조
2. **스타일 자동 선택**: 주제/목적에 따라 최적의 디자인 스타일 추천
3. **체계적 워크플로우**: 개요 → 검토 → 내용 생성 → 검토 → 디자인 적용

## 사용 시나리오

### 시나리오 1: 원천 데이터로부터 슬라이드 생성

- 입력: md, docx, pdf, txt, xlsx 등
- 프로세스: 데이터 분석 → 스타일 선택 → pptx 생성

### 시나리오 2: 기존 슬라이드 스타일 변경

- 입력: 기존 .pptx 파일
- 프로세스: 내용 추출 → 새 스타일 선택 → 재생성

### 시나리오 3: 주제로부터 슬라이드 생성

- 입력: 주제, 목차 (선택)
- 프로세스: 리서치 → 개요 생성 → 검토 → 내용 생성 → 검토 → 디자인 적용

---

## 워크플로우

### 1. 요청 분석 (Request Analysis)

```python
from scripts.workflow import analyze_request

result = analyze_request(user_input)
# Returns: {
#   "scenario": 1|2|3,
#   "topic": str,
#   "content_type": str,  # "학회발표", "기술브리핑", "교육자료" 등
#   "target_audience": str
# }
```

### 2. 스타일 자동 선택 (Style Matching)

```python
from scripts.style_matcher import match_style, load_presets

presets = load_presets()  # presets.json 로드
recommended_style = match_style(
    topic=result["topic"],
    content_type=result["content_type"]
)

# 사용자에게 추천 스타일 제시 및 확인
print(f"추천 스타일: {recommended_style['name']}")
print(f"이유: {recommended_style['rationale']}")
# 사용자가 수정 원하면 다른 스타일 선택 가능
```

### 3. 콘텐츠 생성 (시나리오 3만 해당)

**3.1 개요 생성**

- web_search를 사용하여 주제 리서치
- 슬라이드 개요 초안 작성
- 사용자 검토 및 수정

**3.2 내용 생성**

- 개요를 바탕으로 각 슬라이드 내용 작성
- 선택된 톤(tone)에 맞춰 텍스트 작성
- 사용자 검토 및 수정

### 4. PPTX 생성

**기존 pptx skill 호출 시 스타일 적용:**

```python
# 1. presets.json에서 선택된 스타일 로드
style = presets[recommended_style["style_key"]]

# 2. pptx skill에 전달할 디자인 가이드 생성
design_guide = f"""
스타일: {style['description']}
배경색: {style['background']}
색상 팔레트:
- Primary: {style['colors']['primary']}
- Secondary: {style['colors']['secondary']}
- Accent: {style['colors']['accent']}

폰트:
- Header: {style['fonts']['header']}
- Body: {style['fonts']['body']}

AI 프롬프트 키워드: {', '.join(style['ai_keywords'])}

톤 & 매너: {style['tone']['description']}
텍스트 스타일: {style['tone']['text_style']}
"""

# 3. pptx skill 호출
# "pptx skill을 사용하여 위 디자인 가이드에 따라 슬라이드 생성"
```

---

## 스타일 프리셋

11가지 전문 스타일이 `presets.json`에 정의되어 있습니다:

1. **modern-minimalist** (모던 미니멀) - 학회 발표, 논문 심사
2. **flat-design** (플랫 디자인) - 대학 강의, 유튜브 콘텐츠
3. **bento-grid** (벤토 그리드) - 연구 결과 요약, 시스템 설명
4. **eco-minimalism** (에코 미니멀리즘) - ESG, 심리/상담 자료
5. **glassmorphism** (글래스모피즘) - AI 기술, 혁신 브리핑
6. **dark-contrast** (다크 모드) - 시청각 자료, 썸네일
7. **isometric-3d** (3D 아이소메트릭) - 복잡한 구조/프로세스
8. **retro-modern** (레트로 모던) - 이벤트, 트렌디 브리핑
9. **fresh-clean** (프레시 & 클린) - 식재료, 건강식품
10. **rustic-organic** (러스틱 & 오가닉) - 유기농, 환경 문제
11. **editorial-magazine** (에디토리얼) - 고급 식문화, 브랜딩

각 스타일의 상세 정의는 `presets.json` 참조.

---

## 톤 & 매너 (Tone)

각 스타일에는 기본 톤이 포함되어 있지만, 필요시 오버라이드 가능:

- **Professional** (전문적, 권위있는): 기업 보고서, IR 피칭
- **Persuasive** (설득적, 열정적): 신제품 런칭, 마케팅
- **Minimalist** (미니멀, 현대적): IT 발표, 디자인 포트폴리오
- **Analytical** (분석적, 논리적): 학술 발표, 데이터 분석

---

## 중요 규칙 (Crucial Rules)

### 텍스트 작성 시

- ❌ 이모지, 특수문자 사용 금지
- ✅ 명사형 종결 또는 단문 사용
- ✅ 슬라이드당 최대 5줄
- ✅ 키워드 중심 구성

### 디자인 적용 시

- 항상 기존 pptx skill의 디자인 가이드라인 준수
- 선택된 스타일의 색상/폰트 엄격히 적용
- QA 프로세스 필수 실행

---

## 통합 예시

### 예시 1: 간단한 자동 생성

**사용자 요청:**

```
"AI 윤리에 대한 학회 발표 자료 10장 만들어줘"
```

**Claude의 처리 과정:**

```python
# 1. 스타일 자동 선택
from scripts.style_matcher import match_style, format_design_guide

style_result = match_style(
    topic="AI 윤리",
    content_type="학회발표"
)
# → modern-minimalist 선택 (confidence: high)

# 2. 리서치 실행
# web_search로 최신 AI 윤리 관련 정보 수집

# 3. 디자인 가이드 생성
design_guide = format_design_guide(style_result)

# 4. pptx skill 호출
# "다음 디자인 가이드로 10장짜리 AI 윤리 발표자료를 만들어줘:
# [design_guide]
#
# pptx skill을 사용하여 생성해줘."
```

### 예시 2: 단계별 대화형 생성

**1단계: 개요 생성**

```python
from scripts.workflow import create_outline_prompt

outline_prompt = create_outline_prompt(
    topic="기후 변화와 지속가능성",
    content_type="환경 발표",
    num_slides=15
)
# → Claude가 개요 생성
# → 사용자 검토 및 수정
```

**2단계: 스타일 확인**

```python
# Claude가 자동으로 rustic-organic 추천
# "환경 관련 주제이므로 rustic-organic 스타일을 추천합니다.
#  따뜻한 톤과 자연스러운 색상이 주제와 잘 어울립니다.
#  다른 스타일을 원하시면 말씀해주세요."
```

**3단계: 내용 생성 및 pptx 제작**

```python
# 각 슬라이드 내용 생성 → 검토 → 최종 pptx 생성
```

### 예시 3: 기존 파일 스타일 변경

**사용자 요청:**

```
"이 발표자료를 glassmorphism 스타일로 바꿔줘"
[presentation.pptx 첨부]
```

**Claude의 처리:**

```python
# 1. pptx skill로 내용 추출
# python -m markitdown presentation.pptx

# 2. 새 스타일 로드
from scripts.style_matcher import get_style_by_key
style = get_style_by_key("glassmorphism")

# 3. 디자인 가이드 적용하여 재생성
# pptx skill로 새 스타일 적용
```

### 예시 4: 프로그래밍 방식 사용

```python
import sys
sys.path.append('/mnt/skills/user/presentation-design')

from scripts.style_matcher import match_style, format_design_guide, list_all_styles
from scripts.workflow import validate_slide_content

# 모든 스타일 확인
styles = list_all_styles()
for s in styles:
    print(f"{s['name']}: {s['description']}")

# 특정 주제에 맞는 스타일 찾기
result = match_style(
    topic="딥러닝 기초 강의",
    content_type="대학 강의"
)

print(f"추천 스타일: {result['name']}")
print(f"신뢰도: {result['confidence']}")
print(f"색상: {result['colors']}")

# 슬라이드 내용 검증
content = "딥러닝의 개념\n신경망 구조\n학습 알고리즘"
validation = validate_slide_content(content)
print(f"유효성: {validation['valid']}")
```

---

## 의존성

이 skill은 다음을 활용합니다:

1. **기존 pptx skill** (`/mnt/skills/public/pptx/SKILL.md`)
   - pptx 생성/편집 도구
   - QA 프로세스
   - 이미지 변환

2. **web_search** (콘텐츠 리서치용)

3. **Python 라이브러리**
   - `json` (presets 로드)
   - 기타는 pptx skill과 동일

---

## 파일 구조

```
presentation-design/
├── SKILL.md                    # Skill 정의 및 사용법
├── README.md                   # 이 파일
├── presets.json                # 11개 스타일 정의
├── QUICK_START.md              # 빠른 시작 가이드
├── scripts/
│   ├── integration_helper.py   # 통합 도우미
│   ├── style_matcher.py        # 스타일 자동 매칭
│   └── workflow.py             # 워크플로우 관리
└── evals/
    ├── evals.json              # 테스트 케이스
    └── files/                  # 테스트용 파일들
```

---

## 다음 단계

skill 사용 시:

1. `presets.json` 로드하여 사용 가능한 스타일 확인
2. `style_matcher.py`로 자동 스타일 추천 받기
3. 필요시 사용자가 스타일 직접 선택
4. `workflow.py`로 단계별 진행
5. 마지막에 pptx skill 호출하여 실제 파일 생성
