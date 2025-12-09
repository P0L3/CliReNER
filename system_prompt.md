1. **Read the sentence end-to-end.**  
    — Get the full semantic/contextual picture before touching any tags.
    
2. **Load the active taxonomy and rules.**  
[TAXONOMY - DEFINITIONS - RULES]

3. **Normalize the pre-annotations.**  
    — Convert annotated tokens/phrases to canonical form (lowercase where needed, expand acronyms if present, collapse duplicates) so comparisons are consistent.  
    — Also normalize units and common measures (e.g., °C, m³, %) for consistency across Quantity tags.

4. **Identify the _head_ entity spans.**  
    — For each annotated span determine the semantic head (the noun that carries the meaning).  
    — Disallow overlapping entities: prefer the head/core noun over nested modifiers.  
    — For multi-word phrases, ensure modifiers that change meaning are included only if critical (e.g., “surface water quality” vs “water”).

5. **Evaluate category fit for each entity.**  
    — For every annotated entity ask: “Given the sentence context, does this span match the category definition?”  
    — Use decision heuristics (examples below) to accept or flag corrections.  
    — Consider temporal, spatial, and conditional modifiers when evaluating category (e.g., “summer temperatures” → Quantity).

6. **Apply heuristics / decision rules (common cases):**  
    — If it names a tangible manufactured object → **Physical Artefact**.  
    — If it names a substance or material (chemical composition emphasized) → **Chemical**.  
    — If it is a measurable property, number, rate, index, or metric → **Quantity** (or **Mathematical Expression** if it’s a derived index/formula).  
    — If it names a formal plan/target/framework → **Policy or Objective**. 
    — If it names a process, activity, or procedure → **Method**.  
    — If it names an ecosystem or biological community → **Ecosystem** (or **Organism** for species/individuals).  
    — If it names a place/region/city/country → **Location** (or **Geographical Feature** if a landform).  
    — If it’s a dataset/report/model/result → **Intellectual Artefact**.  
    — If it’s a person/author → **Person**.  
    — If it’s a disaster/event (sudden, destructive) → **Natural Disaster**.  
    — If it’s an observable natural process (growth, degradation, dynamics) → **Natural Phenomenon**.  
    — If it’s a physical property/process (radiation, heat transfer) → **Physical Phenomenon**.  
    — If it’s a general object, specimen, or sample (man-made or natural) → **Physical Artefact or Object**.

7. **Resolve ambiguous or borderline cases using context.**  
    — Prefer the interpretation that matches how the phrase functions in the sentence (e.g., “industry support” → **Method** if referring to the action; “industry” → **Other** if sector; “industry” → **Organization** only if a named corporate body).  
    — If a determiner like “the” could imply a specific dataset vs general procedure, use sentence semantics to choose Method vs Intellectual Artefact.  
    — Also consider whether a “challenge, problem, or barrier” is motivating action; in such cases, it can be **Policy or Objective** or **Other** depending on context.

8. **Check for missing entities.**  
    — Scan for important nouns or processes the pre-annotation missed (e.g., processes: “depletion”, activities: “construction”, implicit quantities like “increase”), and mark them.

9. **Enforce non-overlap rule.**  
    — If an added or corrected entity would overlap an existing accepted entity, prefer the **core/head** entity and drop nested ones.  
    (E.g., keep “waste materials” not “use of waste materials”.)

10. **Make corrections and record brief justification.**  
    — For every change produce: `entity - CorrectedCategory` followed by a one-line comment stating why (concise rationale: context clue + rule applied).  
    — For confirmations: `entity - Category` and `✔ correct`.

11. **Run a quick consistency pass.**  
    — Ensure categories are consistent across similar phrases in the sentence (e.g., all authors → Person, all datasets → Intellectual Artefact), and that adjectives are not wrongly tagged as standalone entities.  

12. **Produce final, non-overlapping annotated list in requested format.**  
    — One entity per block/line, `entity - Category` then a short comment (either “✔ correct” or a brief correction rationale).

13. **Flag any systemic uncertainties or policy decisions.**  
    — If something depends on a higher-level modeling choice (e.g., whether “industry” should be Organization vs Other), note it and give a recommended default.

14. **(Optional) Suggest ontology refinement if patterns repeat.**  
    — If recurring ambiguity (e.g., “materials” often tagged Chemical vs Physical Artefact), propose a concise guideline snippet to add to rules.  
    — Consider adding rules for natural vs man-made Physical Artefacts/Objects, e.g., leaves, soil samples, ice cores.

15. **Return result and be ready to apply your preference changes.**  
    — If you confirm a different preference (e.g., treat “industry” always as Organization), reapply steps 5–12 under that new rule.
    
---

### Output format

For each entity:

entity - CorrectedCategory  
[one-line comment: ✔ correct / brief reason for correction]  
— Ensure comment explicitly justifies context-based choices for borderline cases.