## Phase 1: Parsing Data
- Loaded **729** documents from `ANOT`.
- Loaded **11973** documents from `PRED_IBM`.

## Phase 2: Filtering and Aligning Spans
- Filtered to **722** common documents based on `ANOT`.
- Aligned a total of **8953** unique entity spans.

## Phase 3: Generating and Saving Mapping Matrix
- Plot saved to: `PLOTS/ANOT_vs_PRED_IBM_mapping_matrix.png`

### Label Relationship Analysis (Threshold: 66%)

**Potential Mappings (Likely Equivalent Concepts)**
- `Organism` (`ANOT`) **<-->** `climate-organisms` (`PRED_IBM`) (**88.7%** forward, **73.4%** backward)

**`ANOT` Labels that are Likely Subsets of `PRED_IBM` Labels**
- `Astronomical Object` (`ANOT`) **-->** `climate-nature` (`PRED_IBM`) (**100.0%**)
- `Body Part` (`ANOT`) **-->** `climate-nature` (`PRED_IBM`) (**75.0%**)
- `Body of Water` (`ANOT`) **-->** `climate-nature` (`PRED_IBM`) (**93.4%**)
- `Disease` (`ANOT`) **-->** `climate-hazards` (`PRED_IBM`) (**73.9%**)
- `Ecosystem` (`ANOT`) **-->** `climate-nature` (`PRED_IBM`) (**71.2%**)
- `Energy Source` (`ANOT`) **-->** `climate-mitigations` (`PRED_IBM`) (**80.0%**)
- `Geographical Feature` (`ANOT`) **-->** `climate-nature` (`PRED_IBM`) (**71.4%**)
- `Mathematical Expression` (`ANOT`) **-->** `climate-properties` (`PRED_IBM`) (**69.6%**)
- `Measuring Device` (`ANOT`) **-->** `climate-observations` (`PRED_IBM`) (**72.5%**)
- `Natural Disaster` (`ANOT`) **-->** `climate-hazards` (`PRED_IBM`) (**78.6%**)
- `Organization` (`ANOT`) **-->** `climate-organizations` (`PRED_IBM`) (**73.8%**)
- `Person` (`ANOT`) **-->** `climate-organizations` (`PRED_IBM`) (**78.0%**)
- `Quantity` (`ANOT`) **-->** `climate-properties` (`PRED_IBM`) (**86.7%**)
- `Satellite` (`ANOT`) **-->** `climate-observations` (`PRED_IBM`) (**89.7%**)

**`PRED_IBM` Labels that are Likely Subsets of `ANOT` Labels**
- `climate-greenhouse-gases` (`PRED_IBM`) **-->** `Chemical` (`ANOT`) (**92.4%**)

---
## Phase 1: Parsing Data
- Loaded **729** documents from `ANOT`.
- Loaded **11973** documents from `PRED_BIO`.

## Phase 2: Filtering and Aligning Spans
- Filtered to **722** common documents based on `ANOT`.
- Aligned a total of **9882** unique entity spans.

## Phase 3: Generating and Saving Mapping Matrix
- Plot saved to: `PLOTS/ANOT_vs_PRED_BIO_mapping_matrix.png`

### Label Relationship Analysis (Threshold: 66%)

**Potential Mappings (Likely Equivalent Concepts)**
- `Chemical` (`ANOT`) **<-->** `Matter` (`PRED_BIO`) (**83.0%** forward, **72.5%** backward)
- `Location` (`ANOT`) **<-->** `Location` (`PRED_BIO`) (**66.1%** forward, **71.2%** backward)

**`ANOT` Labels that are Likely Subsets of `PRED_BIO` Labels**
- `Disease` (`ANOT`) **-->** `Phenomena` (`PRED_BIO`) (**72.9%**)
- `Ecosystem` (`ANOT`) **-->** `Environment` (`PRED_BIO`) (**80.0%**)
- `Mathematical Expression` (`ANOT`) **-->** `Quality` (`PRED_BIO`) (**68.8%**)
- `Natural Disaster` (`ANOT`) **-->** `Phenomena` (`PRED_BIO`) (**100.0%**)
- `Natural Phenomenon` (`ANOT`) **-->** `Phenomena` (`PRED_BIO`) (**84.9%**)
- `Organism` (`ANOT`) **-->** `Organism` (`PRED_BIO`) (**87.4%**)
- `Organization` (`ANOT`) **-->** `Organism` (`PRED_BIO`) (**80.0%**)
- `Person` (`ANOT`) **-->** `Organism` (`PRED_BIO`) (**88.5%**)
- `Physical Phenomenon` (`ANOT`) **-->** `Phenomena` (`PRED_BIO`) (**73.8%**)
- `Quantity` (`ANOT`) **-->** `Quality` (`PRED_BIO`) (**86.4%**)
- `Time Period` (`ANOT`) **-->** `Phenomena` (`PRED_BIO`) (**68.2%**)

---
## Phase 1: Parsing Data
- Loaded **11973** documents from `PRED_BIO`.
- Loaded **11973** documents from `PRED_IBM`.

## Phase 2: Filtering and Aligning Spans
- Aligned a total of **145150** unique entity spans.

## Phase 3: Generating and Saving Mapping Matrix
- Plot saved to: `PLOTS/PRED_BIO_vs_PRED_IBM_mapping_matrix.png`

### Label Relationship Analysis (Threshold: 66%)

**`PRED_BIO` Labels that are Likely Subsets of `PRED_IBM` Labels**
- `Environment` (`PRED_BIO`) **-->** `climate-nature` (`PRED_IBM`) (**70.9%**)

**`PRED_IBM` Labels that are Likely Subsets of `PRED_BIO` Labels**
- `climate-greenhouse-gases` (`PRED_IBM`) **-->** `Matter` (`PRED_BIO`) (**90.2%**)
- `climate-hazards` (`PRED_IBM`) **-->** `Phenomena` (`PRED_BIO`) (**73.6%**)
- `climate-impacts` (`PRED_IBM`) **-->** `Phenomena` (`PRED_BIO`) (**73.5%**)
- `climate-organisms` (`PRED_IBM`) **-->** `Organism` (`PRED_BIO`) (**68.8%**)
- `climate-properties` (`PRED_IBM`) **-->** `Quality` (`PRED_BIO`) (**74.1%**)

---
