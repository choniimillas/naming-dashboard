# -*- coding: utf-8 -*-
"""
pipeline.py — Sistema de Naming Estocástico y Evaluación Multi-Agente.
Todo el procesamiento LLM corre sobre NVIDIA NIM (Llama 8B + Mixtral 8x7B).
"""
import os
import sys
import json
import time
import random
import socket
import logging
import requests
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Reconfigurar encoding de salida para Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configurar logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("NamingPipeline")

# Cargar variables de entorno y configuración
ENV_PATH = Path(r"c:\Users\miles\Desktop\Vibe coding\Asesoria\Empresa nombre\.env")
load_dotenv(ENV_PATH)

NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY")

if not NVIDIA_KEY or not SERPER_KEY:
    logger.critical("Faltan API keys en el archivo .env. Abortando.")
    sys.exit(1)

CONFIG_PATH = Path(r"C:\Users\miles\.gemini\antigravity-ide\brain\9d9f5b97-e100-4c4c-9676-37659f596045\config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    docs = list(yaml.safe_load_all(f))
    config = {}
    for doc in docs:
        if doc:
            config.update(doc)

# Parámetros del config
MONTE_CARLO = config["pipeline"]["monte_carlo"]
LLM_CFG = config["pipeline"]["llm"]
LEGAL_CFG = config["pipeline"]["legal"]
MORPHOLOGY_CONSTRAINTS = config["restricciones_morfologicas"]
SEMANTIC_ANGLES = config["distribuciones_generador"]["semantic_angles"]
MORPHOLOGY_FORMATS = config["distribuciones_generador"]["morphology_formats"]
SEED_LEXICON = config["semantica_y_semillas"]
COMPETIDORES = config["mercado_y_competencia"]["competidores_directos"]
MARCAS_CROSS = config["mercado_y_competencia"]["marcas_admiradas_cross_rubro"]

# Constantes de control
CHECKPOINT_PATH = Path("checkpoint.json")
RESULTS_PATH = Path("results.json")

# Definición de descripciones para los ángulos semánticos
ANGLE_DESCS = {
    "fabricar": "pride of building, workshops, hands-on engineering, welding, casting, molds, raw materials, 3D printing",
    "cliente_obtiene": "what the client gains: peace of mind, precision, remote control, real-time metrics, clarity",
    "instrumento": "meters, gauges, calibrators, dials, robust tools, measurement equipment, precision sensors",
    "hecho_de": "raw hardware components: relays, nodes, switches, copper, aluminum, signals, antennas",
    "terreno": "field deployment, ground work, base nodes, ports, rugged terrains, operations in the wild",
    "hace": "actions: tracking, measuring, reporting, transmitting, alert triggers, remote automation",
    "relacion": "connectivity, bridge between machine and dashboard, link, node network, connection",
    "origen_inventado": "abstract but solid engineering sound, roots in latin or germanic phonetics that sound structural"
}

# ============================================================
# API Wrapper con Reintentos y Rate Limiting
# ============================================================
def call_nvidia(prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Llama a la API de NVIDIA NIM con reintentos y backoff exponencial."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    }
    
    max_retries = 5
    delay = 3.0
    for attempt in range(max_retries):
        try:
            # Respetar rate limits
            time.sleep(60.0 / LLM_CFG["nvidia_rate_limit"])
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                logger.warning(f"NVIDIA API rate limit (429). Reintento {attempt+1}/{max_retries} en {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                logger.warning(f"NVIDIA API HTTP {r.status_code}. Reintento {attempt+1}/{max_retries}...")
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            logger.warning(f"Error de conexion con NVIDIA NIM: {e}. Reintento {attempt+1}/{max_retries}...")
            time.sleep(delay)
            delay *= 2
            
    raise RuntimeError(f"Fallo definitivo al llamar a NVIDIA NIM con el modelo {model}")

# ============================================================
# Utilidades de Texto y Filtros Scripted
# ============================================================
def LevenshteinDistance(s1: str, s2: str) -> int:
    """Calcula la distancia Levenshtein básica."""
    if len(s1) < len(s2):
        return LevenshteinDistance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def clean_name(name: str) -> str:
    """Normaliza un nombre de marca."""
    if not name:
        return ""
    name = "".join(c for c in name if c.isalnum())
    return name.upper().strip()

def check_syllables_es(word: str) -> int:
    """Estimacion simple de silabas en español basada en grupos vocalicos."""
    word = word.lower()
    vocals = "aeiouáéíóúü"
    count = 0
    in_vocal = False
    for char in word:
        if char in vocals:
            if not in_vocal:
                count += 1
                in_vocal = True
        else:
            in_vocal = False
    return max(1, count)

def pass_prefilter(name: str) -> bool:
    """Verifica si el nombre cumple con las restricciones morfologicas y de pronunciabilidad."""
    name = clean_name(name)
    if not name:
        return False
    
    # 1. Longitud básica
    if len(name) < 3 or len(name) > 10:  # Acotado a máximo 10 para evitar nombres muy largos
        return False
    
    # 2. Solo alfanuméricos
    if not name.isalnum():
        return False
        
    # 3. Control de Sílabas
    syllables = check_syllables_es(name)
    max_s = MORPHOLOGY_CONSTRAINTS["max_syllables"]
    if syllables > max_s:
        if syllables == 4 and MORPHOLOGY_CONSTRAINTS.get("allow_4_syllable_out_of_box", False):
            pass
        else:
            return False
            
    # 4. Chequeo estricto de Vocales para garantizar pronunciabilidad (evitar siglas secas)
    vocals = "AEIOU"
    vocal_count = sum(1 for c in name if c in vocals)
    if len(name) < 6:
        if vocal_count < 1:  # Nombres cortos necesitan al menos 1 vocal (ej: VOLT)
            return False
    else:
        if vocal_count < 2:  # Nombres largos (>=6) necesitan al menos 2 vocales (ej: KALTRO)
            return False

    # 5. Control de agrupaciones de consonantes impronunciables
    # Máximo 2 consonantes seguidas, excepto si forman parte de un cluster permitido
    name_lower = name.lower()
    consonants = "bcdfghjklmnpqrstvwxyz"
    valid_clusters = [
        "str", "thr", "mpr", "ldr", "ctr", "chr", "mpl", "ncl", "nst", 
        "gth", "rth", "scl", "sch", "phr", "scr", "ndr", "ntr", "ltr"
    ]
    
    consec_consonants = 0
    for i, char in enumerate(name_lower):
        if char in consonants:
            consec_consonants += 1
            if consec_consonants >= 3:
                # Extraer las 3 consonantes consecutivas finalizando en i
                triple = name_lower[i-2:i+1]
                if triple not in valid_clusters:
                    return False
        else:
            consec_consonants = 0

    # 6. Terminaciones prohibidas en español
    for ending in MORPHOLOGY_CONSTRAINTS["banned_endings_es"]:
        if name_lower.endswith(ending):
            return False
            
    return True

def get_pre_score(candidate: dict) -> float:
    """Calcula un pre-score rapido para ordenar candidatos, premiando legibilidad y equilibrio silábico."""
    name = candidate["name"]
    length = len(name)
    if length == 0:
        return 0.0
        
    # 1. Premiar longitud ideal de 5 a 7 letras
    if 5 <= length <= 7:
        len_score = 4.0
    elif length == 4 or length == 8:
        len_score = 2.5
    else:
        len_score = 1.0
        
    # 2. Premiar equilibrio de vocales (idealmente entre 35% y 50% de vocales)
    vocals = "AEIOU"
    vocal_count = sum(1 for c in name if c in vocals)
    vocal_ratio = vocal_count / length
    if 0.35 <= vocal_ratio <= 0.50:
        vocal_score = 4.0
    elif 0.30 <= vocal_ratio <= 0.60:
        vocal_score = 2.5
    else:
        vocal_score = 1.0
        
    # 3. Dar un pequeño bonus por consonantes que den fuerza (T, R, V, L, N) pero moderadamente
    strong_letters = set("TRVLN")
    strong_count = sum(1 for c in name if c in strong_letters)
    strong_ratio = strong_count / length
    strong_score = strong_ratio * 2.0
    
    score = len_score + vocal_score + strong_score
    return max(0.0, min(10.0, score))

def deduplicate(candidates: list[dict], threshold: float = 0.75) -> list[dict]:
    """Colapsa candidatos cuya similitud Levenshtein sea muy alta."""
    unique = []
    for candidate in sorted(candidates, key=lambda c: c.get("pre_score", 0.0), reverse=True):
        name = candidate["name"]
        is_duplicate = False
        for existing in unique:
            dist = LevenshteinDistance(name, existing["name"])
            max_len = max(len(name), len(existing["name"]))
            similarity = 1.0 - (dist / max_len)
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(candidate)
    return unique

# ============================================================
# Carga y Escritura de Checkpoints
# ============================================================
def save_checkpoint(step_name: str, data: dict):
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            checkpoint = {}
    else:
        checkpoint = {}
        
    checkpoint[step_name] = data
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logger.info(f"Guardado checkpoint para: {step_name}")

def load_checkpoint(step_name: str):
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        return checkpoint.get(step_name)
    except Exception:
        return None

# ============================================================
# FASE 2: Loops de Generación
# ============================================================
def select_random_weighted(distribution: dict) -> str:
    choices = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(choices, weights=weights)[0]

def sample_seeds(angle: str) -> list[str]:
    seeds_pool = SEED_LEXICON.get(angle, [])
    if angle in ["fabricar", "instrumento"]:
        seeds_pool += SEED_LEXICON.get("electrico", []) + SEED_LEXICON.get("movimiento", [])
    k = min(len(seeds_pool), random.randint(2, 4))
    return random.sample(seeds_pool, k=k) if seeds_pool else ["hardware", "tracking"]

def run_round_1() -> list[dict]:
    cached = load_checkpoint("ronda_1")
    if cached:
        logger.info("Cargando Ronda 1 desde el checkpoint.")
        return cached

    logger.info("Iniciando Ronda 1: Exploración Amplia (Llama 8B)...")
    candidates = []
    n_batches = 70
    for i in range(n_batches):
        angle = select_random_weighted(SEMANTIC_ANGLES)
        morphology = select_random_weighted(MORPHOLOGY_FORMATS)
        seeds = sample_seeds(angle)
        
        prompt = f"""You are a linguistic branding expert.
Generate exactly 10 unique brand name candidates for an industrial IoT and telemetry hardware company.
The names must sound robust and reliable, but they must be simple, friendly, easy to pronounce in both Spanish and English, and have a natural syllabic flow. 
Avoid harsh, dry abbreviations or complex clusters of consonants with no vowels (such as KRP, TK, RGT, KTR, etc.). We want names that sound like real words or smooth neologisms (e.g. Haasten, Teltonika, Valiot, Strider).

Semantic Angle: {angle} ({ANGLE_DESCS.get(angle, "")})
Morphology Formats: {morphology}
Seed words: {", ".join(seeds)}

Constraints:
- Syllable count: 2 to {MORPHOLOGY_CONSTRAINTS["max_syllables"]} (2 syllables is highly preferred for punchiness)
- Pronunciation: Must have a clear syllabic pattern (e.g. CVCV, CVCVC) with a healthy ratio of vowels. Easy to say.
- Characters: a-zA-Z0-9 only
- Banned Spanish endings: {", ".join(MORPHOLOGY_CONSTRAINTS["banned_endings_es"])}

Reply ONLY with a JSON object in this format, do not include markdown styling or any conversational text:
{{
  "candidates": [
    {{
      "name": "NAME",
      "etymology": "Short description of origin and concept",
      "syllables": 2,
      "morphology": "{morphology}",
      "angle": "{angle}"
    }}
  ]
}}
"""
        try:
            logger.info(f"Ronda 1: Batch {i+1}/{n_batches} (Ángulo: {angle}, Formato: {morphology})")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_model"], temperature=1.1)
            batch_data = json.loads(response_text)
            for c in batch_data.get("candidates", []):
                c["name"] = clean_name(c.get("name", ""))
                c["round"] = 1
                if pass_prefilter(c["name"]):
                    c["pre_score"] = get_pre_score(c)
                    candidates.append(c)
        except Exception as e:
            logger.error(f"Error procesando batch {i+1} en Ronda 1: {e}")
            
    save_checkpoint("ronda_1", candidates)
    return candidates

def run_round_2(r1_candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("ronda_2")
    if cached:
        logger.info("Cargando Ronda 2 desde el checkpoint.")
        return cached

    deduped_r1 = deduplicate(r1_candidates, threshold=0.75)
    exemplars = [c["name"] for c in sorted(deduped_r1, key=lambda x: x["pre_score"], reverse=True)[:30]]
    exemplars_str = ", ".join(exemplars)
    
    logger.info("Iniciando Ronda 2: Refinamiento Dirigido con Exemplars (Llama 8B)...")
    logger.info(f"Exemplars seleccionados de R1: {exemplars_str}")
    
    candidates = []
    n_batches = 70
    for i in range(n_batches):
        angle = select_random_weighted(SEMANTIC_ANGLES)
        morphology = select_random_weighted(MORPHOLOGY_FORMATS)
        seeds = sample_seeds(angle)
        
        prompt = f"""You are a linguistic branding expert.
Generate exactly 10 unique brand name candidates for an industrial IoT and telemetry hardware company.
The names must sound robust and reliable, but they must be simple, friendly, easy to pronounce in both Spanish and English, and have a natural syllabic flow. 
Avoid harsh, dry abbreviations or complex clusters of consonants with no vowels (such as KRP, TK, RGT, KTR, etc.). We want names that sound like real words or smooth neologisms (e.g. Haasten, Teltonika, Valiot, Strider).

Semantic Angle: {angle} ({ANGLE_DESCS.get(angle, "")})
Morphology Formats: {morphology}
Seed words: {", ".join(seeds)}

EXEMPLARS OF STYLE: Use these names as guidance for the TONO, ENERGY, and styling of what we want. Do NOT copy them, generate similar energy:
[{exemplars_str}]

Constraints:
- Syllable count: 2 to {MORPHOLOGY_CONSTRAINTS["max_syllables"]} (2 syllables is highly preferred for punchiness)
- Pronunciation: Must have a clear syllabic pattern (e.g. CVCV, CVCVC) with a healthy ratio of vowels. Easy to say.
- Characters: a-zA-Z0-9 only
- Banned Spanish endings: {", ".join(MORPHOLOGY_CONSTRAINTS["banned_endings_es"])}

Reply ONLY with a JSON object in this format, do not include markdown styling or any conversational text:
{{
  "candidates": [
    {{
      "name": "NAME",
      "etymology": "Short description of origin and concept",
      "syllables": 2,
      "morphology": "{morphology}",
      "angle": "{angle}"
    }}
  ]
}}
"""
        try:
            logger.info(f"Ronda 2: Batch {i+1}/{n_batches} (Ángulo: {angle}, Formato: {morphology})")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_model"], temperature=0.9)
            batch_data = json.loads(response_text)
            for c in batch_data.get("candidates", []):
                c["name"] = clean_name(c.get("name", ""))
                c["round"] = 2
                if pass_prefilter(c["name"]):
                    c["pre_score"] = get_pre_score(c)
                    candidates.append(c)
        except Exception as e:
            logger.error(f"Error procesando batch {i+1} en Ronda 2: {e}")
            
    save_checkpoint("ronda_2", candidates)
    return candidates

def run_round_3(top_candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("ronda_3")
    if cached:
        logger.info("Cargando Ronda 3 desde el checkpoint.")
        return cached

    logger.info("Iniciando Ronda 3: Mutación y Cruce Genético...")
    top_50 = sorted(top_candidates, key=lambda x: x["pre_score"], reverse=True)[:50]
    top_names = [c["name"] for c in top_50]
    
    mutations = []
    prefixes = ["TRAK", "PRO", "NEO", "KINE", "CORE", "VOLT", "FORGE", "KALT", "NODE"]
    suffixes = ["EX", "EN", "IX", "EK", "ON", "IC", "OX", "UM", "UX"]
    
    for c in top_50:
        name = c["name"]
        angle = c["angle"]
        
        p_name = random.choice(prefixes) + name
        if pass_prefilter(p_name):
            mutations.append({"name": p_name, "etymology": f"Prefix mutation of {name}", "angle": angle, "morphology": "mutation"})
            
        s_name = name + random.choice(suffixes)
        if pass_prefilter(s_name):
            mutations.append({"name": s_name, "etymology": f"Suffix mutation of {name}", "angle": angle, "morphology": "mutation"})
            
        if len(name) > 4:
            t_name = name[:4] + random.choice(suffixes)
            if pass_prefilter(t_name):
                mutations.append({"name": t_name, "etymology": f"Truncation mutation of {name}", "angle": angle, "morphology": "mutation"})

    for _ in range(50):
        name1 = random.choice(top_names)
        name2 = random.choice(top_names)
        if name1 != name2 and len(name1) > 3 and len(name2) > 3:
            split1 = len(name1) // 2
            split2 = len(name2) // 2
            cross_name1 = name1[:split1] + name2[split2:]
            cross_name2 = name2[:split2] + name1[split1:]
            if pass_prefilter(cross_name1):
                mutations.append({"name": cross_name1, "etymology": f"Cross blend of {name1} and {name2}", "angle": "mutation", "morphology": "mutation"})
            if pass_prefilter(cross_name2):
                mutations.append({"name": cross_name2, "etymology": f"Cross blend of {name2} and {name1}", "angle": "mutation", "morphology": "mutation"})

    deduped_mutations = deduplicate(mutations, threshold=0.85)
    logger.info(f"Candidatos mutados generados: {len(mutations)} -> {len(deduped_mutations)} únicos post-dedup.")
    
    passed_mutations = []
    batch_size = 30
    batches = [deduped_mutations[i:i + batch_size] for i in range(0, len(deduped_mutations), batch_size)]
    
    for idx, batch in enumerate(batches):
        names_list = [m["name"] for m in batch]
        prompt = f"""You are a branding specialist. We are running a genetic mutation on naming candidates for a rugged, industrial telemetry hardware company.
We generated several mutated options. Evaluate them and filter out the ones that sound like gibberish or are hard to pronounce. Only output the ones that sound like REAL, viable industrial engineering brands.

Candidates to evaluate:
{json.dumps(names_list)}

Reply ONLY with a JSON object in this format containing the names that passed:
{{
  "passed": ["NAME1", "NAME2"]
}}
"""
        try:
            logger.info(f"Ronda 3: Validando lote de mutaciones {idx+1}/{len(batches)}")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_model"], temperature=0.7)
            passed_names = json.loads(response_text).get("passed", [])
            passed_names = [clean_name(n) for n in passed_names]
            
            for m in batch:
                if m["name"] in passed_names:
                    m["round"] = 3
                    m["pre_score"] = get_pre_score(m)
                    passed_mutations.append(m)
        except Exception as e:
            logger.error(f"Error en Ronda 3 batch {idx+1}: {e}")
            
    save_checkpoint("ronda_3", passed_mutations)
    return passed_mutations

# ============================================================
# FASE 3: Evaluación Progresiva
# ============================================================

def run_evaluation_fonetista(candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("eval_fonetista")
    if cached:
        logger.info("Cargando Evaluacion Fonetista desde el checkpoint.")
        return cached

    logger.info("Iniciando Agente Fonetista (Llama 8B) para top 150 candidatos...")
    top_150 = sorted(candidates, key=lambda x: x["pre_score"], reverse=True)[:150]
    
    batch_size = 15
    batches = [top_150[i:i + batch_size] for i in range(0, len(top_150), batch_size)]
    
    evaluated = []
    for idx, batch in enumerate(batches):
        names_list = [c["name"] for c in batch]
        prompt = f"""You are a phonetician expert. Evaluate the following brand names for their suitability in Spanish (ES), English (EN), and Portuguese (PT) pronunciation, and their phonetic hardness/memorability.
Company type: Rugged industrial hardware & telemetry tracker.

Evaluate each name on:
1. pronounce_es: ease of pronunciation in Spanish (0-10)
2. pronounce_en: ease of pronunciation in English (0-10)
3. pronounce_pt: ease of pronunciation in Portuguese (0-10)
4. hardness: phonetical hardness (0-10, consonants like K, T, R, X increase hardness; soft vovels decrease it. We want rugged and solid, aim for high hardness)
5. memorability: how easy it is to remember (0-10)

Names to evaluate:
{json.dumps(names_list)}

Reply ONLY with a JSON object in this format:
{{
  "evaluations": {{
    "NAME": {{
      "pronounce_es": 8.5,
      "pronounce_en": 9.0,
      "pronounce_pt": 7.0,
      "hardness": 8.0,
      "memorability": 7.5
    }}
  }}
}}
"""
        try:
            logger.info(f"Fonetista: Evaluando lote {idx+1}/{len(batches)}")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_model"], temperature=0.3)
            evals = json.loads(response_text).get("evaluations", {})
            
            for c in batch:
                name = c["name"]
                ev = evals.get(name, {
                    "pronounce_es": 7.0, "pronounce_en": 7.0, "pronounce_pt": 6.5,
                    "hardness": 6.0, "memorability": 6.0
                })
                
                c["scores"] = c.get("scores", {})
                c["scores"].update(ev)
                
                es = ev.get("pronounce_es", 7.0)
                pt = ev.get("pronounce_pt", 6.5)
                en = ev.get("pronounce_en", 7.0)
                hardness = ev.get("hardness", 6.0)
                
                hardness_factor = 0.5 + (hardness / 20.0)
                phonetic_base = (es * 0.45 + pt * 0.35 + en * 0.20)
                c["scores"]["phonetic"] = round(phonetic_base * hardness_factor, 2)
                evaluated.append(c)
        except Exception as e:
            logger.error(f"Error en Fonetista lote {idx+1}: {e}")
            for c in batch:
                c["scores"] = c.get("scores", {})
                c["scores"].update({"pronounce_es": 7.0, "pronounce_en": 7.0, "pronounce_pt": 6.5, "hardness": 6.0, "memorability": 6.0, "phonetic": 6.5})
                evaluated.append(c)
                
    save_checkpoint("eval_fonetista", evaluated)
    return evaluated

def run_evaluation_estratega(candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("eval_estratega")
    if cached:
        logger.info("Cargando Evaluacion Estratega desde el checkpoint.")
        return cached

    logger.info("Iniciando Agente Estratega (Mixtral 8x7B) para top 80 candidatos...")
    top_80 = sorted(candidates, key=lambda x: x["scores"]["phonetic"] + x["pre_score"], reverse=True)[:80]
    
    batch_size = 10
    batches = [top_80[i:i + batch_size] for i in range(0, len(top_80), batch_size)]
    
    evaluated = []
    for idx, batch in enumerate(batches):
        names_list = [c["name"] for c in batch]
        prompt = f"""You are a Lead Brand Strategist. Evaluate the following brand name candidates for an industrial IoT telemetry company.
Company Posicionamiento:
- We build hardware (telemetry boards) and custom dashboard software (from board to dashboard).
- High reliability, hands-on Argentine engineering, rugged operations.
- Competitors: we-do.io, efficast.ai, bitronic, nexus-ingenieria, Teltonika.

Evaluate each name on:
1. relevance: how fitting the name is for industrial operations and hardware (0-10)
2. differentiation: how unique it stands out in a crowded market of generic SaaS or IoT terms (0-10)
3. vertical_fit: flexibility to expand to logistics, fleets, utility monitoring, or agriculture (0-10)
4. rationale: A short sentence in Spanish explaining the semantic fit and rationale.

Names to evaluate:
{json.dumps(names_list)}

Reply ONLY with a JSON object in this format:
{{
  "evaluations": {{
    "NAME": {{
      "relevance": 8.0,
      "differentiation": 8.5,
      "vertical_fit": 9.0,
      "rationale": "Breve explicación en español del nombre y su fit semántico."
    }}
  }}
}}
"""
        try:
            logger.info(f"Estratega: Evaluando lote {idx+1}/{len(batches)}")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_evaluator_model"], temperature=0.5)
            evals = json.loads(response_text).get("evaluations", {})
            
            for c in batch:
                name = c["name"]
                ev = evals.get(name, {
                    "relevance": 7.0, "differentiation": 7.0, "vertical_fit": 7.0,
                    "rationale": "Evaluación por defecto."
                })
                c["scores"].update(ev)
                c["etymology"] = ev.get("rationale", c.get("etymology", ""))
                evaluated.append(c)
        except Exception as e:
            logger.error(f"Error en Estratega lote {idx+1}: {e}")
            for c in batch:
                c["scores"].update({"relevance": 6.5, "differentiation": 6.5, "vertical_fit": 7.0, "rationale": "Evaluación por defecto."})
                evaluated.append(c)
                
    save_checkpoint("eval_estratega", evaluated)
    return evaluated

# ============================================================
# FASE 4: Agente Legal (Serper + DNS resolution)
# ============================================================
def check_domain_dns(domain: str) -> bool:
    try:
        socket.setdefaulttimeout(3.0)
        socket.gethostbyname(domain)
        return False
    except socket.gaierror:
        return True
    except Exception:
        return False

def call_serper(query: str) -> int:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "gl": "ar", "hl": "es"}
    try:
        time.sleep(0.5)
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            results = r.json().get("organic", [])
            return len(results)
    except Exception as e:
        logger.error(f"Error en Serper para query '{query}': {e}")
    return 0

def run_evaluation_legal(candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("eval_legal")
    if cached:
        logger.info("Cargando Evaluacion Legal desde el checkpoint.")
        return cached

    logger.info("Iniciando Agente Legal (DNS Check + Serper API) para los 80 candidatos...")
    
    evaluated = []
    total = len(candidates)
    for idx, c in enumerate(candidates):
        name = c["name"]
        logger.info(f"Legal: Procesando {idx+1}/{total} -> {name}")
        
        com_avail = check_domain_dns(f"{name.lower()}.com")
        com_ar_avail = check_domain_dns(f"{name.lower()}.com.ar")
        
        res_brand = call_serper(name)
        res_telemetry = call_serper(f"{name} telemetria")
        res_gps = call_serper(f"{name} GPS")
        
        total_hits = res_brand + res_telemetry + res_gps
        if total_hits == 0:
            google_scarcity = 10.0
        elif total_hits <= 5:
            google_scarcity = 9.5
        elif total_hits <= 15:
            google_scarcity = 9.0
        elif total_hits <= 30:
            google_scarcity = 8.0
        elif total_hits <= 100:
            google_scarcity = 7.0
        elif total_hits <= 500:
            google_scarcity = 5.5
        elif total_hits <= 2000:
            google_scarcity = 4.0
        else:
            google_scarcity = 2.0
            
        c["legal"] = {
            "domain_com": com_avail,
            "domain_com_ar": com_ar_avail,
            "google_hits_total": total_hits,
            "google_hits_brand": res_brand
        }
        
        domain_score = (5.0 if com_avail else 0.0) + (2.0 if com_ar_avail else 0.0)
        c["scores"]["legal"] = round(domain_score + (google_scarcity * 0.3), 2)
        c["scores"]["rarity"] = round(google_scarcity, 2)
        
        evaluated.append(c)
        
    save_checkpoint("eval_legal", evaluated)
    return evaluated

# ============================================================
# FASE 5: Juez y Scoring Final
# ============================================================
def run_evaluation_juez(candidates: list[dict]) -> list[dict]:
    cached = load_checkpoint("eval_juez")
    if cached:
        logger.info("Cargando Evaluacion Juez desde el checkpoint.")
        return cached

    logger.info("Iniciando Agente Juez (Mixtral 8x7B) para scoring final y test de taller...")
    
    batch_size = 10
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    
    evaluated = []
    for idx, batch in enumerate(batches):
        names_list = [c["name"] for c in batch]
        prompt = f"""You are a Senior Judge Evaluator for brand names.
We are choosing a name for a hands-on Argentine engineering team making telemetry hardware (trackers, sensors) and IoT dashboards.
The name must sound natural when used in the workshop and during daily operations.

Evaluate how the names sound in the following Spanish slang/operational sentences (Taller Stress Test):
1. "Traeme el equipo [NAME]"
2. "Le metí un [NAME] a la moto"
3. "¿Reportó el [NAME]?"

Puntúa de 0 a 10 el "taller_fit" (how natural, punchy, and operational it sounds in these contexts). Also, write the completed sentences.

Names to evaluate:
{json.dumps(names_list)}

Reply ONLY with a JSON object in this format:
{{
  "evaluations": {{
    "NAME": {{
      "taller_fit": 8.5,
      "llama_a_los_de": "Traeme el equipo NAME",
      "vertical_flota": "NAME Flota",
      "vertical_industrial": "NAME Industrial",
      "serigrafia": "NAME serigrafiado en placa"
    }}
  }}
}}
"""
        try:
            logger.info(f"Juez: Evaluando lote {idx+1}/{len(batches)}")
            response_text = call_nvidia(prompt, LLM_CFG["nvidia_evaluator_model"], temperature=0.3)
            evals = json.loads(response_text).get("evaluations", {})
            
            for c in batch:
                name = c["name"]
                ev = evals.get(name, {
                    "taller_fit": 6.5,
                    "llama_a_los_de": f"Traeme el equipo {name}",
                    "vertical_flota": f"{name} Flota",
                    "vertical_industrial": f"{name} Industrial",
                    "serigrafia": f"{name} serigrafiado"
                })
                
                c["scores"]["taller_fit"] = ev.get("taller_fit", 6.5)
                c["stress_test"] = {
                    "llama_a_los_de": ev.get("llama_a_los_de", ""),
                    "vertical_flota": ev.get("vertical_flota", ""),
                    "vertical_industrial": ev.get("vertical_industrial", ""),
                    "serigrafia": ev.get("serigrafia", "")
                }
                
                rel = c["scores"].get("relevance", 7.0)
                rarity = c["scores"].get("rarity", 5.0)
                phon = c["scores"].get("phonetic", 7.0)
                leg = c["scores"].get("legal", 5.0)
                v_fit = c["scores"].get("vertical_fit", 7.0)
                t_fit = c["scores"].get("taller_fit", 6.5)
                
                final_score = (
                    (0.22 * rel) + 
                    (0.18 * rarity) + 
                    (0.22 * phon) + 
                    (0.18 * leg) + 
                    (0.10 * v_fit) + 
                    (0.10 * t_fit)
                ) * 10.0
                
                c["scores"]["final"] = round(final_score, 1)
                
                if final_score >= 82.0:
                    c["tier"] = "A"
                elif final_score >= 70.0:
                    c["tier"] = "B"
                elif final_score >= 55.0:
                    c["tier"] = "C"
                else:
                    c["tier"] = "D"
                    
                evaluated.append(c)
        except Exception as e:
            logger.error(f"Error en Juez lote {idx+1}: {e}")
            for c in batch:
                c["scores"]["taller_fit"] = 6.0
                c["scores"]["final"] = 65.0
                c["tier"] = "C"
                evaluated.append(c)
                
    save_checkpoint("eval_juez", evaluated)
    return evaluated

# ============================================================
# Orquestador Principal
# ============================================================
def main():
    logger.info("=== INICIANDO PIPELINE DE NAMING ESTOCÁSTICO V3 ===")
    start_time = time.time()
    
    r1_cand = run_round_1()
    logger.info(f"Ronda 1 finalizada. Candidatos brutos: {len(r1_cand)}")
    
    r2_cand = run_round_2(r1_cand)
    logger.info(f"Ronda 2 finalizada. Candidatos brutos acumulados: {len(r1_cand) + len(r2_cand)}")
    
    combined_r1_r2 = r1_cand + r2_cand
    for c in combined_r1_r2:
        c["pre_score"] = get_pre_score(c)
    unique_gen = deduplicate(combined_r1_r2, threshold=0.75)
    logger.info(f"Fase de generación básica completada. Candidatos únicos: {len(unique_gen)}")
    
    r3_cand = run_round_3(unique_gen)
    logger.info(f"Ronda 3 finalizada. Candidatos mutados aprobados: {len(r3_cand)}")
    
    all_unique = deduplicate(unique_gen + r3_cand, threshold=0.75)
    logger.info(f"Pool final de candidatos únicos: {len(all_unique)}")
    
    top_phonetic = run_evaluation_fonetista(all_unique)
    top_strategic = run_evaluation_estratega(top_phonetic)
    top_legal = run_evaluation_legal(top_strategic)
    final_candidates = run_evaluation_juez(top_legal)
    
    final_candidates = sorted(final_candidates, key=lambda x: x["scores"]["final"], reverse=True)
    
    evol_data = {
        "r1": {
            "generated": len(r1_cand),
            "survived_prefilter": len(deduplicate(r1_cand, 0.75))
        },
        "r2": {
            "generated": len(r2_cand),
            "survived_prefilter": len(deduplicate(r2_cand, 0.75))
        },
        "r3": {
            "generated": len(r3_cand),
            "survived_prefilter": len(r3_cand)
        }
    }
    
    output = {
        "meta": {
            "run_id": f"run_{int(time.time())}",
            "total_generated": len(r1_cand) + len(r2_cand) + (len(unique_gen) * 2),
            "total_after_dedup": len(all_unique),
            "total_evaluated": len(final_candidates),
            "rounds": 3,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S-03:00"),
            "duration_minutes": round((time.time() - start_time) / 60.0, 1)
        },
        "evolution": evol_data,
        "thresholds": {
            "rarity_min": 5.0,
            "relevance_min": 6.0
        },
        "candidates": final_candidates
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"=== PIPELINE COMPLETADO EXITOSAMENTE EN {output['meta']['duration_minutes']} MINUTOS ===")
    logger.info(f"Resultados exportados a: {RESULTS_PATH.resolve()}")

if __name__ == "__main__":
    main()
