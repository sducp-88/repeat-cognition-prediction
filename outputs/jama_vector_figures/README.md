# Editable Figure Sources

The PDF files are the preferred JAMA Network Open submission files. They preserve text, lines, markers, and plotted paths as vector objects. The SVG files are editable in Inkscape, Adobe Illustrator, Affinity Designer, or recent versions of Microsoft PowerPoint after conversion to shapes.

`Figure_Source_Data.xlsx` and the CSV files under `data` contain only the aggregate values displayed in the figures. They do not contain individual-level CHARLS, HRS, or ELSA data.

Regenerate all vector figures by running:

`python 40_jama_vector_figures.py`

Figure 1 uses flow counts recorded in the script and exported to `data/Figure_1_flow.csv`. Figure 2 reads the prespecified sensitivity and ELSA paired-difference aggregate CSV files. Figure 3 reads the HRS and ELSA aggregate calibration and decision-curve CSV files. Figure titles and full legends are intentionally kept in the manuscript rather than embedded in the artwork.
