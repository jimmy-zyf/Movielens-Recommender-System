# Personalization:Final Project

Team members: 
- Diyue Gu (dg3198@columbia.edu, github: Ivygdy)
- Jingyan Xu (jx2424p@columbia.edu, github: jx2424)
- Yifei Zhang (yz3925@columbia.edu, github: jimmy-zyf)
- Chelsea Cui (ac4788@columbia.edu, github: acui34)
- Yishi Wang (yw3619@columbia.edu, github: wangyis)

## Project Goal

The overall goal of this project is to build a improved version of the recommendation system in HW2, which recommends users the movies they potentially would like, by utilizing a hybrid architecture design. In this version, the recommendation systems switches the prediction model based on the user rating history. Based on the advantages each model, the system coud generate the customized best results for user. In this way, we can maintain a relatively active user population and keep promoting newly released movies to the right target audience.

The data used in this project are from the full version of MovieLens Latest Dataset, which was last updated 9/2018. While the dataset included 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users, we only used the ratings table and kept about 20,000 users and 1,000 items for calculation due to the limitation of computation power.


## File Structure

Our submission includes exported Google Collab notebook files (splitted due to extended running time) and intermediate files, as specified below:

```
Project Folder:

│   README.md
│   requirements.txt

# Notebook Files

│   Hybrid Model.ipynb
│   DataPreparation.ipynb
│   ContentBased.ipynb
│   DL.ipynb
│   MF.ipynb

# Python Model Modules (models saved in python for easy import)

│   dlModel.py
│   contentBased.py
│   modelBased.py

# Preprocessed Data (created by DataPreparation.ipynb)
│   train.csv
│   test_for_dp.csv
│   test_for_content.csv
│   test_for_mf.csv

# Cache Data (test results pre-saved for faster evaluation)

│   DL_recommend_df.csv
│   MFRecommendation.csv
│   content_based_recommend_df.csv
│   cb_evaluation_df.csv
│   mb_evaluation_df.csv
│   hybrid_evaluation_df.csv
│   dl_evaluation_df.csv
```

## Report Structure

The main report, including business objectives, model description, model comparison, and final conclusion, can be found in the `Hybrid Model.ipynb` file. Data preparation and sampling can be found in `DataPreparation.ipynb`. Single Model explanatoin and tuning can be found in notesbooks named by the models (`ContentBased.ipynb`, `DL.ipynb`, `MF.ipynb`).

 For the ease of rebuilding the environment we used, we also included Notebook links below:

- Main Report and Hybrid Model
  - Link: https://colab.research.google.com/drive/1LKuJoyNW42HE80G2rVTkdNI3AgojzeB3?usp=sharing

- Matrix Factorization Model and Baseline Model
  - Link: https://colab.research.google.com/drive/1U5yt_IuVVl5CJmZtzObqbaZSCMOWdl2l?usp=sharing

- Content-based Model
  - Link: https://colab.research.google.com/drive/1473nhLVUojLyzlXNbRBieIq6APGl7O3p?usp=sharing

- Deep Learning Model
  - Link: https://colab.research.google.com/drive/1LsxuLngPmVNLclJrl4Vzsub3UY12FhPK?usp=sharing

- Data Preparation
  - Link: https://colab.research.google.com/drive/1RWPTFawW680AS-Wr0lCGgKLV8rMjOATt?usp=sharing
