"""
Persona Generator Module for Humanized Conversation with Personality

This module generates diverse personas based on Big Five personality traits.
"""

import json
import os
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from itertools import product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Prompt Templates
# =============================================================================

PERSONA_GENERATION_PROMPT = """The Big Five Personality is chosen: {big5_config}
(Corresponding to {{OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM}})

Here is the description of the personality:
{personality_description}

################################################################################

User Scenario:
{user_scenario}

################################################################################

TASK:
Create a highly detailed, realistic character profile based on the personality traits and scenario above.
You MUST output the result in strict, valid JSON format.
Do not include any text outside the JSON object.

CRITICAL INSTRUCTIONS FOR DIVERSITY:
1. Name & Region: Generate names and locations from a wide range of cultures. Do not default to USA/UK. Actively include regions such as Europe, Asia, South America, Africa, and Oceania.
2. Job Specifics: Be precise with job titles. Avoid generic terms like "Teacher" or "Artist"; instead, use specific roles like "High School Chemistry Teacher" or "Digital Illustrator."
3. Avoid Clichés: Avoid cliché personality-job pairings. For example, ensure high-neuroticism traits are not exclusively linked to "tortured artist" archetypes, nor high-conscientiousness to "accountants." Ensure unexpected but plausible combinations.

The JSON object must contain exactly the following keys with detailed content:

{{
    "name": "Full name (No repeated names, Cultural diversity encouraged)",
    "gender": "Gender (Male / Female / Non-binary)",
    "age": "Age (string) (Don't restrict to 20-40, especially 30-40 years old.)",
    "region": "City, State, Country (Ensure global diversity)",
    "tone": "Description of their speaking style and voice (Be diverse)",
    "job": "Occupation title (Avoid generic roles; encourage diverse fields)",
    "personality": "Detailed paragraph describing their personality traits, behaviors, and internal motivations",
    "advantages_and_disadvantages": "Strengths and weaknesses",
    "hobby": "Detailed description of hobbies and interests",
    "growth_experience": "Backstory about their childhood or education that shaped them",
    "family_relationship": "Details about their relationship with parents, siblings, or partner",
    "working_conditions": "Details about their work environment and work-life balance",
    "social_relationship": "Details about their friends and social circle",
    "emotional_state": "Typical emotional baseline and how they handle feelings",
    "living_conditions": "Description of their home and neighborhood",
    "recent_worry_or_anxiety": "Specific current concern or stressor consistent with their background",
    "additional_information": "Any other relevant details (e.g., community involvement, specific quirks, or even some small, trivial things)"
}}
"""

PERSONA_JSON_TEMPLATE = {
    "name": "",
    "gender": "",
    "age": "",
    "region": "",
    "tone": "",
    "job": "",
    "personality": "",
    "advantages_and_disadvantages": "",
    "hobby": "",
    "growth_experience": "",
    "family_relationship": "",
    "working_conditions": "",
    "social_relationship": "",
    "emotional_state": "",
    "living_conditions": "",
    "recent_worry_or_anxiety": "",
    "additional_information": ""
}

# =============================================================================
# User Scenario Templates
# =============================================================================

USER_SCENARIOS = [
    """You are an adult with a dreamy and artistic appearance. Your style is eclectic and colorful, perhaps wearing mismatched clothing inspired by various cultures and historical periods. Your expression is lively and friendly, radiating warmth and enthusiasm. You are in a vibrant and slightly chaotic setting, like an artist's studio filled with unfinished projects and an array of art supplies. You might be caught in a moment of creative inspiration, with a canvas or a sketchbook in hand, surrounded by books, plants, and art pieces that reflect your rich imagination and a passion for beauty and knowledge.""",
    
    """You are a professional working in a corporate environment. Your appearance is polished and well-organized, reflecting your attention to detail and systematic approach to life. You work in a clean, structured office space with everything in its place. Your demeanor is confident and measured, showing your ability to handle pressure and meet deadlines consistently.""",
    
    """You are a community volunteer who spends weekends helping at local shelters. Your warm smile and approachable manner make others feel comfortable around you. You dress casually but neatly, ready to roll up your sleeves and help wherever needed. Your living space reflects your caring nature, with photos of family and friends prominently displayed.""",
    
    """You are a tech enthusiast who loves exploring new gadgets and technologies. Your workspace is filled with multiple monitors, gadgets, and technical books. You prefer working alone but communicate effectively when needed. Your analytical mind helps you solve complex problems, though you sometimes struggle with small talk at social gatherings.""",
    
    """You are an outdoor adventurer who feels most alive when exploring nature. Your appearance is rugged and practical, designed for hiking and camping. You speak with enthusiasm about your latest adventures and are always planning the next trip. Your home is decorated with maps, photos of mountains, and gear ready for the next expedition."""

    ################### Diverse professional field ###################
    
    # 1. Creative / Arts
    """You are a creative professional (e.g., artist, writer, musician) working in a vibrant, somewhat chaotic studio. You value self-expression and aesthetics above all else.""",
    
    # 2. Corporate / Business
    """You are a corporate professional (e.g., analyst, manager, consultant) working in a high-rise office. You value efficiency, career progression, and professional networking.""",
    
    # 3. Healthcare / Caregiving
    """You work in healthcare or caregiving (e.g., nurse, therapist, vet). You spend your days caring for others, often in high-emotion or high-stress environments.""",
    
    # 4. Tech / Engineering
    """You are a technical expert (e.g., developer, engineer, data scientist). You love solving complex problems and likely work in a modern tech hub with flexible hours.""",
    
    # 5. Outdoors / Nature
    """You work outdoors or with nature (e.g., park ranger, landscape architect, farmer). You prefer fresh air to office desks and have a practical, rugged outlook on life.""",
    
    # 6. Education / Academic
    """You are in education (e.g., teacher, professor, librarian). You value knowledge, mentorship, and shaping the minds of others, often working in structured institutional settings.""",
    
    # 7. Service / Hospitality
    """You work in the service industry (e.g., chef, event planner, hotel manager). Your job is fast-paced and social, requiring you to think on your feet and manage customer expectations.""",
    
    # 8. Skilled Trade / Manual
    """You work in a skilled trade (e.g., carpenter, electrician, mechanic). You take pride in building or fixing things with your hands and value practical results over theory.""",
    
    # 9. Public Service / Law
    """You work in public service or law (e.g., lawyer, social worker, police officer). You deal with societal rules, justice, and community issues on a daily basis.""",
    
    # 10. Entrepreneurial
    """You are a small business owner or startup founder. You hustle hard, wear many hats, and your work-life balance is often blurred by your passion for your business.""",
    
    # 11. Student / Early Career
    """You are a university student or recent graduate. You are still figuring out your path, balancing studies/internships with a vibrant social life and future anxieties.""",
    
    # 12. Retired / Senior
    """You are retired or nearing retirement. You have a lifetime of experience and now focus on hobbies, community, grand-children, or travel.""",
    
    # 13. Media / Communications
    """You work in media (e.g., journalist, influencer, PR). You are always connected, shaping narratives, and responding to the latest trends.""",
    
    # 14. Science / Research
    """You are a researcher or scientist. You spend your time in labs or libraries, driven by curiosity and the pursuit of objective truth.""",
    
    # 15. Gig Economy / Freelance
    """You are a freelancer or gig worker (e.g., driver, designer, tutor). You value flexibility and autonomy, though you sometimes stress about financial stability.""",
    
    # 16. Sports / Fitness
    """You work in fitness or sports (e.g., personal trainer, athlete, coach). Physical health and discipline are central to your daily routine and identity.""",
    
    # 17. Retail / Fashion
    """You work in retail or fashion. You have a keen eye for trends and presentation, dealing with customers and aesthetics daily.""",
    
    # 18. Logistics / Transport
    """You work in logistics (e.g., pilot, truck driver, warehouse manager). You keep the world moving and value punctuality, safety, and order.""",
    
    # 19. Non-profit / Activism
    """You work for a non-profit or as an activist. You are driven by a specific cause (environment, human rights) rather than money.""",
    
    # 20. Homemaker / Parent
    """You are a stay-at-home parent or homemaker. Your management skills are applied to running a household and raising a family, requiring immense patience and multitasking."""

    # 21. Entertainment / Performance
    """You work in the entertainment industry (e.g., actor, musician, director, producer, comedian). You thrive on performance, storytelling, and audience engagement, often navigating the intense balance between your public persona and private life."""    
]

# =============================================================================
# Persona Generator Class
# =============================================================================

class PersonaGenerator:
    DIMENSIONS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    LEVELS = ['high', 'low']
    
    def __init__(self, llm_client=None, definitions_path="data/big5_list.json"):
        self.llm_client = llm_client
        self.generated_personas = []

        self.trait_definitions = self._load_definitions(definitions_path)

    def _load_definitions(self, path: str) -> Dict:
        """Load personality definitions from JSON file"""
        if not os.path.exists(path):
            logger.warning(f"Definitions file {path} not found. Using default/empty descriptions.")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten the nested JSON: {"Openness": {"high": "Description"}} 
            # to the format required by the code: {("high", "Openness"): "Description"}
            descriptions = {}
            for dim, content in data.items():
                dim_str = str(dim)
                
                if isinstance(content, dict):
                    if "high" in content:
                        descriptions[("high", dim)] = content["high"]
                    if "low" in content:
                        descriptions[("low", dim)] = content["low"]
            
            logger.info(f"Successfully loaded {len(descriptions)} trait definitions from {path}")
            return descriptions
            
        except Exception as e:
            logger.error(f"Error loading definitions: {e}")
            return {}

    def generate_all_combinations(self) -> List[Dict[str, str]]:
        combinations = []
        for combo in product(self.LEVELS, repeat=5):
            combinations.append(dict(zip(self.DIMENSIONS, combo)))
        return combinations
    
    def get_personality_description(self, big5_combo: Dict[str, str]) -> str:
        descriptions = []
        for dimension, level in big5_combo.items():
            key = (level, dimension)
            if key in self.trait_definitions:
                descriptions.append(self.trait_definitions[key])
            else:
                # Fallback text if missing
                descriptions.append(f"{level} levels of {dimension}.")
        return " ".join(descriptions)
    
    def format_big5_string(self, big5_combo: Dict[str, str]) -> str:
        values = [big5_combo[dim] for dim in self.DIMENSIONS]
        return "{" + ", ".join(values) + "}"
    
    def create_generation_prompt(self, big5_combo: Dict[str, str], scenario_index: int = 0) -> str:
        big5_str = self.format_big5_string(big5_combo)
        description = self.get_personality_description(big5_combo)
        scenario = USER_SCENARIOS[scenario_index % len(USER_SCENARIOS)]
        
        return PERSONA_GENERATION_PROMPT.format(
            big5_config=big5_str,
            personality_description=description,
            user_scenario=scenario
        )
    
    def generate_persona(self, big5_combo: Dict[str, str], scenario_index: int = 0) -> Optional[Dict]:
        if self.llm_client is None:
            return None
        
        prompt = self.create_generation_prompt(big5_combo, scenario_index)
        
        try:
            for attempt in range(3):
                try:
                    response_text = self.llm_client.generate(prompt)
                    persona_profile = self._parse_persona_response(response_text)
                    
                    if persona_profile:
                        # Construct the final structure
                        return {
                            "big-5": self.format_big5_string(big5_combo),
                            "profile": persona_profile
                        }
                    else:
                        logger.warning(f"Failed to parse JSON on attempt {attempt+1}")
                except Exception as e:
                    logger.error(f"API Error on attempt {attempt+1}: {e}")
                    time.sleep(2)
            
        except Exception as e:
            logger.error(f"Fatal error generating persona: {e}")
        
        return None
    
    def _parse_persona_response(self, response: str) -> Optional[Dict]:
        """Robust JSON parsing"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                match = re.search(r"```(?:json)?(.*?)```", response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                    return json.loads(json_str)
            except Exception:
                pass
        return None
    
    def generate_persona_bank(self, personas_per_combination: int = 4, save_path: str = None):
        combinations = self.generate_all_combinations()
        all_personas = []
        
        total = len(combinations) * personas_per_combination
        count = 0
        
        for i, combo in enumerate(combinations):
            logger.info(f"Processing combination {i+1}/{len(combinations)}: {self.format_big5_string(combo)}")
            
            for j in range(personas_per_combination):
                persona = self.generate_persona(combo, scenario_index=j)
                
                # If generation failed, create a placeholder to keep index alignment
                if not persona:
                    logger.warning(f"Generation failed for {count}, using placeholder.")
                    persona = {
                        "big-5": self.format_big5_string(combo),
                        "profile": {"name": "Error Generating", "personality": "Failed to retrieve from LLM"}
                    }
                    
                ordered_persona = {
                    "index": count,                 # Index is first
                    "big-5": persona["big-5"],      # Big-5 string is second
                    "profile": persona["profile"]   # Profile dict is third
                }
                
                all_personas.append(ordered_persona)
                count += 1
                
                # Rate limit protection
                time.sleep(1) 
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_personas, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved {len(all_personas)} personas to {save_path}")

# =============================================================================
# Behavior Preset Generator
# =============================================================================

class BehaviorPresetGenerator:
    """Generate behavior presets for role-playing scenarios"""
    
    BEHAVIOR_TEMPLATES = {
        "high_openness": [
            {
                "trigger": "new ideas or creative topics",
                "response_style": "enthusiasm and curiosity, offering your own creative perspectives",
                "examples": [
                    "That's fascinating! Have you considered looking at it from this angle...",
                    "I love exploring new ideas. What if we combined that with..."
                ]
            }
        ],
        "low_openness": [
            {
                "trigger": "unconventional suggestions",
                "response_style": "practical concerns and preference for proven methods",
                "examples": [
                    "That sounds interesting, but have we considered the traditional approach?",
                    "I prefer sticking with what works. Let's focus on the basics."
                ]
            }
        ],
        "high_conscientiousness": [
            {
                "trigger": "planning or organization topics",
                "response_style": "detailed planning and systematic thinking",
                "examples": [
                    "Let me outline the steps we should follow...",
                    "I've prepared a detailed schedule for this."
                ]
            }
        ],
        "low_conscientiousness": [
            {
                "trigger": "strict deadlines or rigid structures",
                "response_style": "flexibility and spontaneity",
                "examples": [
                    "We can figure out the details as we go.",
                    "Let's not get too caught up in the planning."
                ]
            }
        ],
        "high_extraversion": [
            {
                "trigger": "social activities or group discussions",
                "response_style": "energy and enthusiasm for interaction",
                "examples": [
                    "That sounds like so much fun! Count me in!",
                    "I'd love to meet everyone and hear their stories."
                ]
            }
        ],
        "low_extraversion": [
            {
                "trigger": "large social gatherings",
                "response_style": "preference for smaller, more intimate settings",
                "examples": [
                    "I'd prefer a quieter setting where we can really talk.",
                    "Maybe we could meet one-on-one instead?"
                ]
            }
        ],
        "high_agreeableness": [
            {
                "trigger": "conflicts or disagreements",
                "response_style": "empathy and desire for harmony",
                "examples": [
                    "I understand where you're coming from. Let's find a compromise.",
                    "Everyone's feelings are valid. How can we work together?"
                ]
            }
        ],
        "low_agreeableness": [
            {
                "trigger": "requests for compromise",
                "response_style": "standing firm on your position",
                "examples": [
                    "I appreciate your view, but I have to disagree.",
                    "Let's focus on the facts rather than feelings here."
                ]
            }
        ],
        "high_neuroticism": [
            {
                "trigger": "stressful situations or uncertainty",
                "response_style": "expressing worry and seeking reassurance",
                "examples": [
                    "I'm a bit worried about how this might turn out...",
                    "What if something goes wrong? Have we prepared for that?"
                ]
            }
        ],
        "low_neuroticism": [
            {
                "trigger": "stressful situations",
                "response_style": "calm reassurance and confidence",
                "examples": [
                    "Don't worry, we've got this under control.",
                    "These challenges are just opportunities in disguise."
                ]
            }
        ]
    }
    
    def generate_behavior_presets(self, big5_combo: Dict[str, str]) -> List[Dict]:
        """Generate behavior presets for a given Big Five combination"""
        presets = []
        
        for dimension, level in big5_combo.items():
            key = f"{level}_{dimension.lower()}"
            
            if key in self.BEHAVIOR_TEMPLATES:
                presets.extend(self.BEHAVIOR_TEMPLATES[key])
        
        return presets
    
    def format_preset_prompt(self, preset: Dict) -> str:
        """Format a behavior preset as a prompt addition"""
        examples_str = "\n".join(f"- \"{ex}\"" for ex in preset['examples'])
        
        return f"""When the user says something related to {preset['trigger']}, you should respond with {preset['response_style']}.

Examples of appropriate responses:
{examples_str}
"""

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    from openai import OpenAI
    
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # Replace this with your OpenAI API key
    
    class OpenAIClient:
        def __init__(self, key):
            self.client = OpenAI(api_key=key)
            
        def generate(self, prompt):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                # Simple retry logic or error handling can be added here
                return "{}"

    if "sk-xxxx" in API_KEY:
        print("Error: Please update the API_KEY variable with your actual OpenAI API key.")
    else:
        print("Initializing OpenAI client...")
        client = OpenAIClient(API_KEY)
        
        print("Initializing Persona Generator...")
        generator = PersonaGenerator(llm_client=client)

        number_combination = 10
        print(f"Starting generation of {number_combination * 32} personas...")
        print("Note: This process may take some time. Please do not close the terminal.")
        
        os.makedirs("data", exist_ok=True)
        
        # Generating 4 personas per combination * 32 combinations = 128 personas
        generator.generate_persona_bank(
            personas_per_combination=number_combination, 
            save_path="data/big5-persona.json"
        )
        
        print("Success! Data saved to data/big5-persona.json")
