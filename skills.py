import json
import os

def load_skills(question: str) -> str:
    question_lower = question.lower()
    active_instructions = []
    active_skill_names = []
    
    skills_file = os.path.join(os.path.dirname(__file__), "skills.json")
    try:
        with open(skills_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            registry = data.get("global", [])
    except Exception:
        registry = []
    
    for skill in registry:
        if "*" in skill.get("keywords", []):
            active_instructions.append(skill["instruction"])
            active_skill_names.append(skill["name"])
            continue
            
        for kw in skill.get("keywords", []):
            if kw.lower() in question_lower:
                active_instructions.append(skill["instruction"])
                active_skill_names.append(skill["name"])
                break
                
    if not active_instructions:
        return ""
        
    names_str = ", ".join(active_skill_names)
    instructions_str = "\n\n---\n\n".join(active_instructions)
    return f"\n\n============================================================\nACTIVE SKILLS: {names_str}\n============================================================\n{instructions_str}\n============================================================\n"
