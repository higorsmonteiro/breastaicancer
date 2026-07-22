# Breast cancer risk prediction from longitudinal mammography reports in a real-world health system
 
 
# Introduction
Here we provide the pipeline for: [Breast cancer risk prediction from longitudinal mammography reports in a real-world health system](to-be-released)

### Citation

> XXXXJOURNALXXXX. DOI: XXXX


# Abstract

Breast cancer screening protocols apply uniform strategies across populations with vastly different underlying risk profiles, limiting early detection in high-risk individuals and generating unnecessary burden in low-risk ones. Risk-based screening has been proposed as a strategy to tailor screening intensity to individual risk, but its implementation requires accurate, calibrated, and scalable risk models validated in real-world settings. Here we develop and validate a population-scale breast cancer risk model using longitudinal mammography reports and clinical data from approximately 4.6 million examinations of 1.8 million patients across the largest private health system in Brazil. Without requiring mammography images, the model achieves \textcolor{blue}{strong and clinically meaningful} discrimination for predicting breast cancer within one to five years, with a 5-year AUROC of 0.86 (95\% CI: 0.85--0.87) and consistent performance across shorter horizons. Flagging the top 3.4\% highest-risk exams identifies 50.1\% of future cancer cases, enabling earlier detection, while in the bottom 47\% risk percentile only 10.0\% develop cancer within five years, suggesting approximately half of all patients could safely transition to longer screening intervals.  An explainability analysis reveals distinct and clinically meaningful patterns driving predictions at both ends of the risk spectrum. These results establish the feasibility of interpretable, image-free, population-scale risk modeling as a practical pathway toward personalized breast cancer screening.

__Fig. X XXXX.__
*(a) Epidemiological curves of Covid-19 cases and deaths in Fortaleza, and major interventions implemented in the city, Jan 2020-July 2022. (b) Cases (gray) and (c) deaths (red) of Covid-19 and cumulative distribution of individuals vaccinated with 2 doses (we consider a single dose of the Janssen vaccine as equivalent to a two-dose regimen).* 



# Organization
We have organized this repo into three main folders:
- `code` - Code to perform all the tasks of the project. Codes to bundle all classes to perform a specific tasks.
    - `src` - Main classes to bundle functions regarding a specific task (e. g. performing the matching procedure).
    - `lib` - auxiliary functions used in the main classes.  
- `data` - Contains the final dataset (after linkage) with all deidentified information used to create the cohort and perform the survival analysis.
- `figures` - All figures from the main document for publication. 


# Correspondence
For any issues with anonymization or major issues with the functionality of the script please [create an issue](https://github.com/higorsmonteiro/breastaicancer/issues).


## License
The data collected and presented is licensed under the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/), and the underlying code used to format, analyze and display that content is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php). 


# Authors
- __Higor S Monteiro__: Department of Physics, Universidade Federal do Ceará | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [GitHub Profile](https://github.com/higorsmonteiro)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Profile](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
