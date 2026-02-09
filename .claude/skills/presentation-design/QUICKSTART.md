# 빠른 시작 가이드 (Quick Start)

## 1단계: Skill 설치

이 폴더를 `/mnt/skills/user/presentation-design/`으로 복사하세요:

```bash
cp -r presentation-design /mnt/skills/user/
```

## 2단계: 기본 사용

Claude에게 다음과 같이 요청하세요:

```
"AI 윤리에 대한 학회 발표 자료 10장 만들어줘"
```

Claude가 자동으로:
1. ✅ 주제 분석 ("AI 윤리" + "학회 발표")
2. ✅ 최적 스타일 추천 (modern-minimalist)
3. ✅ 웹 리서치 실행
4. ✅ 슬라이드 개요 생성
5. ✅ 사용자 검토 요청
6. ✅ 최종 pptx 생성

## 3단계: 스타일 커스터마이징

특정 스타일을 원하면:

```
"건강식품 소개 자료를 fresh-clean 스타일로 만들어줘"
```

사용 가능한 스타일:
- modern-minimalist (학회, 논문)
- flat-design (강의, 교육)
- bento-grid (연구, 시스템)
- eco-minimalism (ESG, 상담)
- glassmorphism (AI, 기술)
- dark-contrast (영상, 썸네일)
- isometric-3d (프로세스, 구조)
- retro-modern (이벤트, 트렌드)
- fresh-clean (식품, 건강)
- rustic-organic (유기농, 환경)
- editorial-magazine (고급, 브랜딩)

## 4단계: 고급 사용

### Python 스크립트 직접 사용

```python
import sys
sys.path.append('/mnt/skills/user/presentation-design')

from scripts.style_matcher import match_style, format_design_guide

# 자동 스타일 매칭
result = match_style(
    topic="딥러닝 기초",
    content_type="대학 강의"
)

print(result['name'])       # 추천된 스타일
print(result['rationale'])  # 추천 이유

# 디자인 가이드 생성
guide = format_design_guide(result)
```

### 개요만 먼저 생성

```python
from scripts.workflow import create_outline_prompt

prompt = create_outline_prompt(
    topic="기후 변화",
    content_type="환경 발표",
    num_slides=12
)

# 이 프롬프트를 Claude에게 전달
```

## 자주 묻는 질문 (FAQ)

**Q: 스타일이 자동으로 안 맞으면?**  
A: 직접 스타일 키를 지정하세요. 예: "glassmorphism 스타일로"

**Q: pptx skill이 없으면?**  
A: 이 skill은 기존 pptx skill (`/mnt/skills/public/pptx/`)을 호출합니다. Claude가 자동으로 사용합니다.

**Q: 검토 과정을 건너뛰려면?**  
A: "검토 없이 바로 만들어줘"라고 요청하세요.

**Q: 더 많은 슬라이드를 원하면?**  
A: "20장짜리 발표자료"처럼 명시하세요.

## 테스트

설치 후 테스트:

```bash
cd /mnt/skills/user/presentation-design
python scripts/style_matcher.py
```

정상 작동하면 11개 스타일 목록이 출력됩니다.

## 다음 단계

- 📖 상세 가이드는 `README.md` 참고
- 🎨 스타일 커스터마이징은 `presets.json` 편집
- 🧪 테스트 케이스는 `evals/evals.json` 참고

---

**문제가 있나요?** Claude에게 "presentation-design skill 문제 해결 도와줘"라고 요청하세요!
