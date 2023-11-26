# Multi-level Graph Representations of Melanoma Whole Slide Images for Identifying Immune Subgroups
This is a repo for code for a GRAIL 2023 paper. In this repo there is code for generating the multi-level graph representations and run the GNN models which are described in the paper :-)

Stratifying melanoma patients into immune subgroups is important for understanding patient outcomes and treatment options. Current weakly supervised classification methods often involve dividing digitised whole slide images into patches, which leads to the loss of important contextual diagnostic information. Here, we propose using graph attention neural networks, which utilise graph representations of whole slide images, to introduce context to classifications. In addition, we present a novel hierarchical graph approach, which leverages histopathological features from multiple resolutions to improve on state-of-the-art (SOTA) multiple instance learning (MIL) methods. We achieve a mean test area under the curve metric of 0.80 for classifying low and high immune melanoma subtypes, using multi-level and 20x patch graph representations of whole slide images, compared to 0.77 when using SOTA MIL methods. Our experimental results comprehensively show how our whole slide image graph representation is a valuable improvement on the MIL paradigm and could help to determine early-stage prognostic markers and stratify melanoma patients for effective treatments.****
