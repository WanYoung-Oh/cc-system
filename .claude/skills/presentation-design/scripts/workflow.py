"""
Workflow Manager - Orchestrate the presentation creation process
"""

from typing import Dict, List, Optional
import re


def analyze_request(user_input: str) -> Dict:
    """
    Analyze user request to determine scenario and extract key information
    
    Args:
        user_input: User's request text
    
    Returns:
        Dictionary with scenario, topic, content_type, and target_audience
    """
    # Determine scenario based on keywords
    scenario = 3  # Default: create from topic
    
    # Check for existing file references (scenario 1 or 2)
    file_patterns = [r'\.pptx', r'\.docx', r'\.pdf', r'\.txt', r'\.md', r'\.xlsx']
    has_file = any(re.search(pattern, user_input.lower()) for pattern in file_patterns)
    
    if has_file:
        if '.pptx' in user_input.lower():
            scenario = 2  # Modify existing presentation
        else:
            scenario = 1  # Create from source data
    
    # Extract topic (simplified - look for common patterns)
    topic = _extract_topic(user_input)
    
    # Identify content type
    content_type = _identify_content_type(user_input)
    
    # Identify target audience (if mentioned)
    target_audience = _identify_audience(user_input)
    
    return {
        "scenario": scenario,
        "topic": topic,
        "content_type": content_type,
        "target_audience": target_audience,
        "original_request": user_input
    }


def _extract_topic(text: str) -> str:
    """
    Extract the main topic from user input
    
    This is a simplified extraction - in practice, might use LLM
    """
    # Remove common command words
    clean_text = text
    remove_words = ['만들어', '작성', '생성', '슬라이드', '프레젠테이션', '발표자료', 'ppt', 'pptx']
  
    for word in remove_words:
        clean_text = re.sub(re.escape(word), '', clean_text, flags=re.IGNORECASE)
    
    # Clean up
    clean_text = clean_text.strip()
    
    # If still too long, take first meaningful phrase
    if len(clean_text) > 50:
        # Try to find first sentence or phrase
        sentences = re.split(r'[.!?]', clean_text)
        clean_text = sentences[0] if sentences else clean_text[:50]
    
    return clean_text if clean_text else "주제 미상"


def _identify_content_type(text: str) -> str:
    """
    Identify the type of content based on keywords
    """
    content_types = {
        "학회발표": ["학회", "논문", "학술", "컨퍼런스"],
        "기술브리핑": ["기술", "AI", "인공지능", "시스템", "프로세스"],
        "교육자료": ["강의", "수업", "교육", "학습"],
        "마케팅": ["마케팅", "런칭", "브랜딩", "홍보"],
        "보고서": ["보고", "분석", "결과", "데이터"],
        "제안서": ["제안", "기획", "IR", "투자"],
    }
    
    text_lower = text.lower()
    
    for content_type, keywords in content_types.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            return content_type
    
    return "일반발표"


def _identify_audience(text: str) -> str:
    """
    Identify target audience from text
    """
    audiences = {
        "전문가": ["전문가", "연구자", "교수", "박사"],
        "일반인": ["일반", "대중", "시민"],
        "학생": ["학생", "대학생", "고등학생"],
        "경영진": ["경영진", "임원", "CEO", "CFO"],
        "투자자": ["투자자", "VC", "엔젤"],
    }
    
    text_lower = text.lower()
    
    for audience, keywords in audiences.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            return audience
    
    return "일반"


def create_outline_prompt(topic: str, content_type: str, num_slides: int = 10) -> str:
    """
    Create a prompt for generating presentation outline
    
    Args:
        topic: Presentation topic
        content_type: Type of content
        num_slides: Target number of slides
    
    Returns:
        Prompt string for outline generation
    """
    prompt = f"""
다음 주제로 {content_type} 프레젠테이션 개요를 작성해주세요.

주제: {topic}
목표 슬라이드 수: {num_slides}장

각 슬라이드에 대해 다음을 포함해주세요:
1. 슬라이드 번호와 제목
2. 주요 내용 요약 (2-3문장)
3. 포함될 시각 요소 제안 (차트, 이미지, 아이콘 등)

개요 작성 시 다음을 고려해주세요:
- 논리적 흐름 (도입 → 본론 → 결론)
- 각 슬라이드당 최대 5개 포인트
- {content_type}의 특성에 맞는 구성

개요를 작성해주세요.
"""
    return prompt


def create_content_prompt(outline: str, slide_number: int, tone_style: str) -> str:
    """
    Create a prompt for generating content for a specific slide
    
    Args:
        outline: Full presentation outline
        slide_number: Which slide to generate content for
        tone_style: Tone description (from preset)
    
    Returns:
        Prompt string for content generation
    """
    prompt = f"""
다음 개요를 바탕으로 슬라이드 {slide_number}의 상세 내용을 작성해주세요.

[전체 개요]
{outline}

[슬라이드 {slide_number} 작성 지침]
- 톤: {tone_style}
- 명사형 종결 또는 단문 사용
- 최대 5줄 (각 줄은 짧게)
- 이모지, 특수문자 사용 금지
- 키워드 중심 구성

다음 형식으로 작성해주세요:

## 슬라이드 제목

### 주요 내용
- 포인트 1
- 포인트 2
- ...

### 시각 요소 제안
[어떤 차트/이미지/아이콘을 사용할지]

### 발표자 노트
[발표 시 강조할 내용]
"""
    return prompt


def format_pptx_instruction(style_guide: str, all_content: List[Dict]) -> str:
    """
    Format instructions for pptx skill
    
    Args:
        style_guide: Design guide from format_design_guide()
        all_content: List of slide content dictionaries
    
    Returns:
        Complete instruction for pptx skill
    """
    instruction = f"""
{style_guide}

=== 슬라이드 내용 ===

"""
    
    for i, slide in enumerate(all_content, 1):
        instruction += f"""
--- 슬라이드 {i} ---
제목: {slide.get('title', '')}

내용:
{slide.get('content', '')}

시각 요소: {slide.get('visual_elements', '')}

---
"""
    
    instruction += """

위 디자인 가이드와 내용을 바탕으로 프레젠테이션을 생성해주세요.
pptx skill을 사용하여 전문적인 슬라이드를 만들어주세요.
"""
    
    return instruction


def validate_slide_content(content: str) -> Dict:
    """
    Validate slide content against rules
    
    Args:
        content: Slide content text
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check for emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    
    if emoji_pattern.search(content):
        issues.append("이모지가 포함되어 있습니다")
    
    # Check line count
    lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
    if len(lines) > 5:
        issues.append(f"텍스트 줄 수가 {len(lines)}개로 5줄을 초과합니다")
    
    # Check for excessive special characters
    special_chars = ['★', '●', '◆', '■', '□', '○', '◎', '※']
    if any(char in content for char in special_chars):
        issues.append("특수문자가 포함되어 있습니다")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "line_count": len(lines)
    }


# Workflow orchestration functions

def run_scenario_1(source_file: str, style_key: Optional[str] = None) -> Dict:
    """
    Scenario 1: Create presentation from source data
    
    Args:
        source_file: Path to source file (md, docx, pdf, etc.)
        style_key: Optional style key to use
    
    Returns:
        Workflow result
    """
    return {
        "scenario": 1,
        "message": "Scenario 1은 데이터 분석 후 pptx skill에 전달하는 방식으로 구현됩니다"
    }


def run_scenario_2(pptx_file: str, new_style_key: str) -> Dict:
    """
    Scenario 2: Change style of existing presentation
    
    Args:
        pptx_file: Path to existing .pptx file
        new_style_key: New style to apply
    
    Returns:
        Workflow result
    """
    return {
        "scenario": 2,
        "message": "Scenario 2는 pptx 내용 추출 후 새 스타일로 재생성하는 방식으로 구현됩니다"
    }


def run_scenario_3_interactive(topic: str, num_slides: int = 10, style_key: Optional[str] = None) -> Dict:
    """
    Scenario 3: Create presentation from topic (interactive)
    
    This is a simplified version - actual implementation would interact with user
    
    Args:
        topic: Presentation topic
        num_slides: Number of slides to create
        style_key: Optional style key to use
    
    Returns:
        Workflow status
    """
    steps = [
        "1. 주제 리서치 (web_search)",
        "2. 개요 생성",
        "3. 사용자 검토 → 개요 수정",
        "4. 슬라이드별 내용 생성",
        "5. 사용자 검토 → 내용 수정",
        "6. 스타일 적용 및 pptx 생성",
        "7. QA 및 최종 검토"
    ]
    
    return {
        "scenario": 3,
        "topic": topic,
        "num_slides": num_slides,
        "steps": steps,
        "message": "Scenario 3은 단계별 대화형 프로세스로 구현됩니다"
    }


if __name__ == "__main__":
    # Test workflow functions
    print("=== Workflow Manager Test ===\n")
    
    # Test request analysis
    request = "AI 윤리에 대한 학회 발표 자료 10장 만들어줘"
    result = analyze_request(request)
    print(f"Request: {request}")
    print(f"Analysis: {result}\n")
    
    # Test outline prompt generation
    prompt = create_outline_prompt("AI 윤리", "학회발표", 10)
    print("Outline Prompt:")
    print(prompt[:200] + "...\n")
    
    # Test content validation
    valid_content = "AI 윤리의 중요성\n데이터 편향 문제\n투명성 확보 방안"
    invalid_content = "AI 윤리 🤖\n" + "\n".join([f"포인트 {i}" for i in range(10)])
    
    print("Valid content check:", validate_slide_content(valid_content))
    print("Invalid content check:", validate_slide_content(invalid_content))
