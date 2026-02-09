"""
Style Matcher - Automatically match presentation topics to optimal design styles
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_presets(presets_path: Optional[str] = None) -> Dict:
    """
    Load style presets from presets.json
    
    Args:
        presets_path: Path to presets.json. If None, uses default location.
    
    Returns:
        Dictionary of style presets
    """
    if presets_path is None:
        # Default: look for presets.json in parent directory
        script_dir = Path(__file__).parent
        presets_path = script_dir.parent / "presets.json"
    
    with open(presets_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def match_style(topic: str, content_type: str = "", target_audience: str = "") -> Dict:
    """
    Match a topic to the most appropriate presentation style with priority-based matching
    
    Args:
        topic: Main topic of the presentation
        content_type: Type of content (e.g., "학회발표", "기술브리핑", "교육자료")
        target_audience: Target audience description
    
    Returns:
        Dictionary with recommended style and rationale
    """
    presets = load_presets()
    
    # Priority-based matching rules (higher priority = matched first)
    # Format: (keyword, style_key, priority)
    style_rules = [
        # Priority 1: Very specific keywords (highest priority)
        ("학회 발표", "modern-minimalist", 10),
        ("논문 심사", "modern-minimalist", 10),
        ("유기농", "rustic-organic", 10),
        ("식문화", "editorial-magazine", 10),
        
        # Priority 2: Academic & Professional
        ("학회", "modern-minimalist", 8),
        ("논문", "modern-minimalist", 8),
        ("학술", "bento-grid", 8),
        ("연구", "bento-grid", 7),
        ("분석", "bento-grid", 6),
        
        # Priority 3: Technology & Innovation
        ("ai", "glassmorphism", 9),
        ("인공지능", "glassmorphism", 9),
        ("딥러닝", "glassmorphism", 9),
        ("머신러닝", "glassmorphism", 9),
        ("기술 동향", "glassmorphism", 8),
        ("혁신", "glassmorphism", 7),
        ("프로세스", "isometric-3d", 7),
        ("시스템", "isometric-3d", 7),
        ("구조", "isometric-3d", 6),
        ("아키텍처", "isometric-3d", 7),
        
        # Priority 4: Education
        ("강의", "flat-design", 7),
        ("교육", "flat-design", 7),
        ("수업", "flat-design", 7),
        ("학습", "flat-design", 6),
        
        # Priority 5: Environment & Sustainability
        ("esg", "eco-minimalism", 9),
        ("지속가능", "eco-minimalism", 8),
        ("환경", "rustic-organic", 7),
        ("친환경", "eco-minimalism", 8),
        
        # Priority 6: Food & Health
        ("식품", "fresh-clean", 7),
        ("건강식품", "fresh-clean", 8),
        ("건강", "fresh-clean", 6),
        ("음식", "editorial-magazine", 6),
        ("요리", "editorial-magazine", 7),
        ("레시피", "fresh-clean", 7),
        ("셰프", "editorial-magazine", 8),
        
        # Priority 7: Marketing & Events
        ("마케팅", "retro-modern", 7),
        ("런칭", "dark-contrast", 8),
        ("신제품", "dark-contrast", 7),
        ("이벤트", "retro-modern", 6),
        ("브랜딩", "editorial-magazine", 7),
        
        # Priority 8: Visual & Media
        ("썸네일", "dark-contrast", 9),
        ("영상", "dark-contrast", 7),
        ("비디오", "dark-contrast", 7),
        
        # Priority 9: General fallbacks (lowest)
        ("발표", "modern-minimalist", 3),
        ("보고", "modern-minimalist", 3),
    ]
    
    # Convert inputs to lowercase for matching
    topic_lower = topic.lower()
    content_lower = content_type.lower()
    audience_lower = target_audience.lower()
    combined_text = f"{topic_lower} {content_lower} {audience_lower}"
    
    # Find best matching style based on priority
    best_match = None
    best_priority = -1
    
    for keyword, style_key, priority in style_rules:
        if keyword in combined_text:
            if priority > best_priority:
                best_priority = priority
                best_match = style_key
    
    # Default to modern-minimalist if no match
    matched_style_key = best_match if best_match else "modern-minimalist"
    
    style = presets[matched_style_key]
    
    return {
        "style_key": matched_style_key,
        "name": style["name"],
        "description": style["description"],
        "colors": style["colors"],
        "fonts": style["fonts"],
        "ai_keywords": style["ai_keywords"],
        "tone": style["tone"],
        "rationale": _generate_rationale(topic, content_type, style),
        "confidence": "high" if best_priority >= 8 else "medium" if best_priority >= 5 else "low"
    }


def _generate_rationale(topic: str, content_type: str, style: Dict) -> str:
    """
    Generate explanation for why this style was chosen
    """
    recommended_for = ", ".join(style["recommended_for"])
    return f"'{topic}' 주제는 {style['name']} 스타일이 적합합니다. 이 스타일은 {recommended_for}에 추천됩니다."


def list_all_styles() -> List[Dict]:
    """
    Get a list of all available styles with basic info
    
    Returns:
        List of style summaries
    """
    presets = load_presets()
    
    styles = []
    for key, style in presets.items():
        styles.append({
            "key": key,
            "name": style["name"],
            "description": style["description"],
            "recommended_for": style["recommended_for"]
        })
    
    return styles


def get_style_by_key(style_key: str) -> Dict:
    """
    Get full style definition by key
    
    Args:
        style_key: Style identifier (e.g., "modern-minimalist")
    
    Returns:
        Full style definition
    """
    presets = load_presets()
    
    if style_key not in presets:
        raise ValueError(f"Style '{style_key}' not found. Available styles: {list(presets.keys())}")
    
    return presets[style_key]


def format_design_guide(style: Dict) -> str:
    """
    Format style information as a design guide for pptx skill
    
    Args:
        style: Style dictionary from presets or from match_style result
    
    Returns:
        Formatted design guide string
    """
    # Check if this is a match_style result or a preset
    # match_style result has 'style_key', preset doesn't
    if 'style_key' in style:
        # This is from match_style - need to get full preset
        style_key = style['style_key']
        presets = load_presets()
        full_style = presets[style_key]
        # Merge with result (to keep rationale and confidence)
        style = {**full_style, **style}
    
    guide = f"""
=== 프레젠테이션 디자인 가이드 ===

스타일: {style['name']}
설명: {style['description']}

배경색: {style.get('background', '#FFFFFF')}

색상 팔레트:
- Primary: {style['colors']['primary']}
- Secondary: {style['colors']['secondary']}
- Accent: {style['colors']['accent']}

폰트:
- Header: {style['fonts']['header']}
- Body: {style['fonts']['body']}

AI 프롬프트 키워드:
{', '.join(style['ai_keywords'])}

톤 & 매너:
- 기본 톤: {style['tone']['default']}
- 설명: {style['tone']['description']}
- 텍스트 스타일: {style['tone']['text_style']}

추천 용도:
{', '.join(style['recommended_for'])}
"""
    
    return guide


if __name__ == "__main__":
    # Test the matcher
    print("=== Style Matcher Test ===\n")
    
    # Test case 1: AI topic
    result = match_style("AI 윤리", "학회발표")
    print("Topic: AI 윤리")
    print(f"Recommended: {result['name']}")
    print(f"Rationale: {result['rationale']}\n")
    
    # Test case 2: Food topic
    result = match_style("건강식품 소개", "교육자료")
    print("Topic: 건강식품 소개")
    print(f"Recommended: {result['name']}")
    print(f"Rationale: {result['rationale']}\n")
    
    # List all styles
    print("=== Available Styles ===")
    styles = list_all_styles()
    for style in styles:
        print(f"- {style['name']}: {style['description']}")
