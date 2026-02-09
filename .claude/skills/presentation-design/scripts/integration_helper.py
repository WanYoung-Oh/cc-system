"""
Integration Helper - Bridge between presentation-design and pptx skill
"""

from typing import Dict, List
from scripts.style_matcher import match_style, format_design_guide, get_style_by_key


def generate_pptx_instruction(
    topic: str,
    slides_content: List[Dict],
    style_key: str = None,
    content_type: str = "",
    num_slides: int = None
) -> str:
    """
    Generate complete instruction for pptx skill
    
    Args:
        topic: Presentation topic
        slides_content: List of slide dictionaries with 'title', 'content', 'visual_elements'
        style_key: Optional style key. If None, auto-match from topic
        content_type: Type of content for auto-matching
        num_slides: Target number of slides
    
    Returns:
        Complete instruction string for pptx skill
    """
    
    # Auto-select style if not provided
    if style_key is None:
        style_result = match_style(topic, content_type)
        style_key = style_result['style_key']
        style = style_result
    else:
        style = {**get_style_by_key(style_key), 'style_key': style_key}
    
    # Generate design guide
    design_guide = format_design_guide(style)
    
    # Build instruction
    instruction = f"""
프레젠테이션 제작 요청

주제: {topic}
스타일: {style['name']} (선택 이유: {style.get('rationale', '사용자 지정')})
목표 슬라이드 수: {num_slides if num_slides else len(slides_content)}장

{design_guide}

=== 슬라이드 내용 ===

"""
    
    # Add each slide
    for i, slide in enumerate(slides_content, 1):
        instruction += f"""
[슬라이드 {i}]
제목: {slide.get('title', f'슬라이드 {i}')}

내용:
{slide.get('content', '')}

{f"시각 요소: {slide.get('visual_elements', '')}" if slide.get('visual_elements') else ""}

---
"""
    
    instruction += """

pptx skill을 사용하여 위 디자인 가이드와 내용으로 전문적인 프레젠테이션을 생성해주세요.

중요:
- 선택된 스타일의 색상 팔레트를 정확히 적용
- 폰트는 지정된 Header/Body 폰트 사용
- 각 슬라이드는 5줄 이하로 제한
- 시각 요소(차트, 아이콘 등)를 적극 활용
- pptx skill의 QA 프로세스 실행
"""
    
    return instruction


def quick_generate_instruction(user_request: str) -> str:
    """
    Quick generation from user request
    
    Args:
        user_request: User's simple request like "AI 윤리 발표자료 10장"
    
    Returns:
        Instruction for pptx skill
    """
    from workflow import analyze_request
    
    # Analyze request
    analysis = analyze_request(user_request)
    
    # Match style
    style_result = match_style(
        topic=analysis['topic'],
        content_type=analysis['content_type']
    )
    
    # Generate simple instruction
    design_guide = format_design_guide(style_result)
    
    instruction = f"""
{user_request}

추천 스타일: {style_result['name']}
이유: {style_result['rationale']}

{design_guide}

pptx skill을 사용하여 위 스타일로 프레젠테이션을 생성해주세요.
"""
    
    return instruction


def suggest_visuals_for_slide(slide_title: str, slide_content: str, style_key: str) -> List[str]:
    """
    Suggest visual elements for a slide based on content and style
    
    Args:
        slide_title: Slide title
        slide_content: Slide content
        style_key: Style being used
    
    Returns:
        List of visual element suggestions
    """
    suggestions = []
    
    content_lower = f"{slide_title} {slide_content}".lower()
    
    # Data visualization keywords
    if any(word in content_lower for word in ['비교', 'vs', '대비', '차이']):
        suggestions.append("비교 차트 (Bar chart or Column chart)")
    
    if any(word in content_lower for word in ['변화', '추세', '증가', '감소', '시간']):
        suggestions.append("라인 차트 (Line chart)")
    
    if any(word in content_lower for word in ['비율', '구성', '점유율', '%']):
        suggestions.append("파이 차트 또는 도넛 차트")
    
    if any(word in content_lower for word in ['단계', '과정', 'step', '프로세스']):
        suggestions.append("프로세스 다이어그램 (화살표 + 번호)")
    
    if any(word in content_lower for word in ['구조', '계층', '관계']):
        suggestions.append("조직도 또는 마인드맵")
    
    # Style-specific suggestions
    if style_key == "isometric-3d":
        suggestions.append("3D 아이소메트릭 일러스트")
    elif style_key == "fresh-clean":
        suggestions.append("밝은 배경의 고해상도 이미지")
    elif style_key == "glassmorphism":
        suggestions.append("반투명 레이어 효과")
    
    # Default suggestions
    if not suggestions:
        suggestions.append("아이콘 + 텍스트 레이아웃")
        suggestions.append("이미지 또는 일러스트")
    
    return suggestions


# Example usage
if __name__ == "__main__":
    print("=== Integration Helper Test ===\n")
    
    # Test quick generation
    request = "AI 윤리에 대한 학회 발표 자료 10장"
    instruction = quick_generate_instruction(request)
    print("Quick instruction generated:")
    print(instruction[:300] + "...\n")
    
    # Test visual suggestions
    visuals = suggest_visuals_for_slide(
        "AI 기술의 발전 추세",
        "2020년부터 2024년까지 AI 기술의 성장률 비교",
        "glassmorphism"
    )
    print("Visual suggestions:")
    for v in visuals:
        print(f"  - {v}")
