# Presentation Design Skill

프레젠테이션 디자인과 콘텐츠 생성을 자동화하는 skill입니다. 기존 `pptx` skill의 강력한 기능을 활용하면서, 전문적인 디자인 스타일과 톤을 자동으로 선택하고 적용합니다.

## 주요 기능

✨ **자동 스타일 선택**: 주제와 목적에 따라 11가지 전문 스타일 중 최적의 디자인 자동 추천  
📝 **체계적 워크플로우**: 개요 생성 → 검토 → 내용 생성 → 검토 → 디자인 적용  
🎨 **11가지 프로 스타일**: 모던 미니멀부터 에디토리얼까지 다양한 디자인 프리셋  
🔄 **기존 pptx skill 통합**: 중복 없이 검증된 도구 재사용

## 사용 방법

### 1. 기본 사용

```python
# 주제만 제공하여 슬라이드 생성
"AI 윤리에 대한 학회 발표 자료 10장 만들어줘"

# 자동으로:
# - 스타일 분석 및 추천 (modern-minimalist)
# - 주제 리서치 (web_search)
# - 개요 생성 및 사용자 검토
# - 슬라이드 내용 생성
# - pptx 파일 생성
```

### 2. 스타일 직접 지정

```python
# 특정 스타일 요청
"건강식품 소개 자료를 fresh-clean 스타일로 만들어줘"
```

### 3. 기존 파일 스타일 변경

```python
# 기존 pptx의 스타일 변경
"이 발표자료를 glassmorphism 스타일로 바꿔줘"
```

## 사용 가능한 스타일

| 스타일 | 추천 용도 |
|--------|-----------|
| **modern-minimalist** | 학회 발표, 논문 심사, 전문 강연 |
| **flat-design** | 대학 강의, 유튜브 콘텐츠 |
| **bento-grid** | 연구 결과 요약, 시스템 설명 |
| **eco-minimalism** | ESG, 심리/상담 자료 |
| **glassmorphism** | AI 기술, 혁신 브리핑 |
| **dark-contrast** | 시청각 자료, 썸네일 |
| **isometric-3d** | 복잡한 구조/프로세스 설명 |
| **retro-modern** | 이벤트, 트렌디한 브리핑 |
| **fresh-clean** | 식재료, 건강식품 소개 |
| **rustic-organic** | 유기농, 환경 문제 |
| **editorial-magazine** | 고급 식문화, 브랜딩 |

각 스타일의 상세 정보는 `presets.json`을 참고하세요.

## 스크립트 사용 (개발자용)

### 스타일 매칭 테스트

```bash
cd scripts
python style_matcher.py
```

### 워크플로우 테스트

```bash
python workflow.py
```

### 프로그래밍 방식 사용

```python
from scripts.style_matcher import match_style, format_design_guide

# 스타일 자동 매칭
result = match_style(
    topic="AI 윤리",
    content_type="학회발표"
)

print(result['name'])  # '모던 미니멀 (Modern Minimalist)'
print(result['rationale'])  # 선택 이유

# 디자인 가이드 생성
guide = format_design_guide(result)
print(guide)  # pptx skill에 전달할 가이드
```

## 파일 구조

```
presentation-design/
├── SKILL.md                    # Skill 정의 및 사용법
├── README.md                   # 이 파일
├── presets.json                # 11개 스타일 정의
├── scripts/
│   ├── __init__.py
│   ├── style_matcher.py        # 스타일 자동 매칭
│   └── workflow.py             # 워크플로우 관리
└── evals/
    ├── evals.json              # 테스트 케이스
    └── files/                  # 테스트용 파일들
```

## 워크플로우

### Scenario 1: 원천 데이터로부터 생성
1. 사용자가 md, docx, pdf 등 파일 제공
2. 데이터 분석 및 스타일 자동 선택
3. pptx skill로 슬라이드 생성

### Scenario 2: 기존 슬라이드 스타일 변경
1. 기존 .pptx 파일에서 내용 추출
2. 새 스타일 선택
3. pptx skill로 재생성

### Scenario 3: 주제로부터 생성 (대화형)
1. 요청 분석 (주제, 목적, 청중)
2. 스타일 자동 추천 및 확인
3. **리서치**: web_search로 최신 정보 수집
4. **개요 생성**: 슬라이드 구조 초안
5. **사용자 검토**: 개요 수정/승인
6. **내용 생성**: 각 슬라이드 상세 내용
7. **사용자 검토**: 내용 수정/승인
8. **디자인 적용**: pptx skill로 최종 생성
9. **QA**: 자동 검증 및 수정

## 규칙 (중요)

### 텍스트 작성 시
- ❌ 이모지, 특수문자 사용 금지
- ✅ 명사형 종결 또는 단문 사용
- ✅ 슬라이드당 최대 5줄
- ✅ 키워드 중심 구성

### 디자인 적용 시
- 선택된 스타일의 색상/폰트 엄격히 적용
- 기존 pptx skill의 디자인 가이드라인 준수
- QA 프로세스 필수 실행

## 의존성

- **기존 pptx skill** (`/mnt/skills/public/pptx/`)
- **web_search** (콘텐츠 리서치)
- **Python 3.8+**
- pptx skill의 모든 의존성 (markitdown, pptxgenjs 등)

## 테스트

```bash
# Eval 모드로 테스트
python -c "
from scripts.style_matcher import list_all_styles
styles = list_all_styles()
for s in styles:
    print(f'{s[\"name\"]}: {s[\"description\"]}')
"
```

## 라이선스

이 skill은 기존 pptx skill을 활용하므로, pptx skill의 라이선스를 따릅니다.

## 기여

개선 아이디어나 버그 리포트는 환영합니다!

## 다음 단계

1. `/mnt/skills/user/presentation-design/`로 이 skill 배포
2. skill-creator로 테스트 및 개선
3. 실제 프로젝트에 사용

---

**v1.0.0** - 2026-02-08
