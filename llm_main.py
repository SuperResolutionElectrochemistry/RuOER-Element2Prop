from openai import OpenAI
import pandas as pd
import os
import time


client = OpenAI(
    api_key="",
    base_url=""
)

MODEL_NAME = "gpt-4o"
REQUEST_INTERVAL = 1.2  


FINAL_PROMPT = """
You are a scientific information extraction assistant specializing in Ru-based electrocatalysts for the oxygen evolution reaction (OER).

Definitions:
- Ru-based materials: materials containing at least two metallic elements, one of which must be Ru.
- Catalytic research: research focused on catalysts or catalytic processes.
- The definition of OER: The Oxygen Evolution Reaction (OER) is an electrocatalytic process that forms oxygen (O₂), and it is different from Oxygen Reduction Reaction(ORR) and Hydrogen Evolution Reaction(HER).

Task:

Based solely on the provided title and abstract, classify the article as either a catalysis study and an OER study, and extract the specific mechanisms or physical-chemical properties.

If a property is not explicitly mentioned, or you are uncertain, reply exactly "NULL".

Input:
Title: {title}
Abstract: {abstract}

Instructions for property extraction:
For each property listed below, check whether it is explicitly mentioned in the title or abstract.
The property may be expressed using equivalent terms or phrases listed.
If mentioned, extract the original sentence or phrase.
If not mentioned, reply "NULL".

Properties and equivalent expressions:

1) Ru–O covalency:
metal–oxygen covalency; Ru–O covalency; Ru–O hybridization; Ru–O bond covalency;
p–d covalency; Ru 4d–O 2p hybridization; covalent character

2) Metal dissolution free energy:
ΔG_diss; metal dissolution free energy; dissolution thermodynamics;
dissolution driving force; dissolution potential (E_diss); leaching thermodynamics

3) High-valence accessibility:
high oxidation state accessibility; valence evolution; oxidation state shift;
average oxidation state; Ru(V); Ru(VI)

4) Oxygen vacancy formation energy:
oxygen vacancy formation energy; vacancy formation energy;
defect formation energy; oxygen vacancy formation enthalpy; E_vac

5) Metal–oxygen bond strength:
metal–oxygen bond strength; M–O bond strength;
metal–oxygen bond energy; bond dissociation energy; BDE

6) Non-lattice oxygen participation tendency: 
non-lattice oxygen participation; non-LOM; negligible lattice oxygen involvement; 
suppressed lattice oxygen participation; AEM-dominated

7) Pourbaix stability window:
Pourbaix diagram stability; Pourbaix stability window;
electrochemical stability window; E–pH stability

8) Configurational entropy:
configurational entropy; mixing entropy; entropy stabilization;
high-entropy effect; S_config

9) e_g orbital occupancy:
e_g orbital occupancy; e_g filling; e_g electron count; e_g descriptor

10) Overpotential:
overpotential; η; η10; η100; onset overpotential

11) ΔG*O:
ΔG*O; ΔG(O); oxygen adsorption free energy;
*O adsorption energy; O* binding energy

12) ΔG*OH:
ΔG*OH; ΔG(OH); hydroxyl adsorption free energy;
*OH adsorption energy; OH* binding energy

13) Work function:
work function; surface work function; electronic work function;
Φ; vacuum level alignment

Output format (STRICT, one item per line, no extra text):

Detailed Research Field: [Catalytic Research or Other Research]
Specified Research Field: [Research on OER or Other Research]
Article research on Ru-based materials: [Yes or No]
Article Type: [Research or Review or NULL]
Elements: [Comma-separated metal symbols or NULL]
Ru–O covalency: [Extracted content or NULL]
Metal dissolution free energy: [Extracted content or NULL]
High-valence accessibility: [Extracted content or NULL]
Oxygen vacancy formation energy: [Extracted content or NULL]
Metal–oxygen bond strength: [Extracted content or NULL]
Non-lattice oxygen participation tendency: [Extracted content or NULL]
Pourbaix stability window: [Extracted content or NULL]
Configurational entropy: [Extracted content or NULL]
e_g orbital occupancy: [Extracted content or NULL]
Overpotential: [Extracted content or NULL]
ΔG*O: [Extracted content or NULL]
ΔG*OH: [Extracted content or NULL]
Work function: [Extracted content or NULL]
"""

FIELDS = [
    "Detailed Research Field",
    "Specified Research Field",
    "Article research on Ru-based materials",
    "Article Type",
    "Elements",
    "Ru–O covalency",
    "Metal dissolution free energy",
    "High-valence accessibility",
    "Oxygen vacancy formation energy",
    "Metal–oxygen bond strength",
    "Non-lattice oxygen participation tendency",
    "Pourbaix stability window",
    "Configurational entropy",
    "e_g orbital occupancy",
    "Overpotential",
    "ΔG*O",
    "ΔG*OH",
    "Work function"
]
FINAL_COLUMNS = ["Title", "Abstract"] + FIELDS

def call_gpt(title, abstract):
    prompt = FINAL_PROMPT.format(title=title, abstract=abstract)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise scientific information extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            timeout=60
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("API Error:", e)
        return None


def parse_output(text):
    parsed = {k: "NULL" for k in FIELDS}
    if text is None:
        return parsed

    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in parsed:
                parsed[key] = value
    return parsed


input_dir = ""
output_dir = ""

for file in os.listdir(input_dir):
    if not file.lower().endswith((".xls", ".xlsx")):
        continue

    file_path = os.path.join(input_dir, file)
    print(f"Processing: {file_path}")

    df_in = pd.read_excel(file_path)
    results = []

    for idx, row in df_in.iterrows():
        title = str(row["Article Title"])
        abstract = str(row["Abstract"])

        gpt_text = call_gpt(title, abstract)
        time.sleep(REQUEST_INTERVAL)

        parsed = parse_output(gpt_text)
        parsed["Title"] = title
        parsed["Abstract"] = abstract

        results.append(parsed)

    df_out = pd.DataFrame(results, columns=FINAL_COLUMNS)
    out_name = os.path.splitext(file)[0] + f"-{MODEL_NAME}.csv"
    out_path = os.path.join(output_dir, out_name)

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

print("All files processed.")

