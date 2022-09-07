# Capstone-IS03_PT-SureBoTv2

This project is an extension of SureBotv1 (https://github.com/adrielkuek/IRS-PM-2021-01-16-ISY5001PT-SureBoT)

## SECTION 1: PROJECT TITLE

## SureBoTv2 - Combating Multimodel Misinformation and Disinformation

![surebot](https://user-images.githubusercontent.com/67159970/114734523-dfb7d280-9d76-11eb-9d84-a029fa6a5a29.gif)

## SECTION 2: OBJECTIVES

In this project, the team aims to explore multimodal understanding of social media content to tackle the issue of prevalent hate speech, violent content, and fake news (misinformation/disinformation). Current context of popular social media content is primarily propagated through short video clips (Tik Tok) or images with accompanying text captions, one such popular propagation medium that has become ubiquitous today are meme images. While memes originated purely as a tool to propagate ideas of humour and nonsensical information, recent light has seen increasing use of memes as propagation mediums of hate speech, racial discrimination, violent content and even misinformation

Such medium often utilize both imagery and textual content to masquerade ideas through sarcasm or non-obvious references which proves to be a significant challenge for automated understanding such as sentiment analysis. We believe that joint multimodal understanding is crucial in enabling a baseline framework towards semantic understanding and sentiment analysis. 

## SECTION 3: CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What) | Email (Optional)
| ---- | ---- | ---- | ---- |
| CHUA HAO ZI | A0229960W | 1. Market Research <br /> 2.  <br /> 3.  <br /> 4. Project Report Writing  | e0687368@u.nus.edu |
| YAP POW LOOK  | A0163450M | 1. Market Research <br /> 2.  <br /> 3.  <br /> 4. Project Report Writing | e0147014@u.nus.edu |
| Kenneth Lee | AXXXXXX | 1. Market Research <br /> 2. <br /> 3.  <br /> 4. Project Report Writing | exxxxxx@u.nus.edu |


## SECTION 4: MARKETING VIDEO


## SECTION 5: USER GUIDE

### Installation
Reccomend to use python 3.7 or higher. Requires Pytorch and Transformers from Huggingface

**Step 1: Get the repository**

Using `git clone 
```
git clone <Github Repo URL>`
```
**Step 2: Create a Conda Environment**

Create a new environment to sandbox your developmental workspace (with Anaconda Prompt)
```
conda create -n "YOUR_ENV_NAME" python=3.7
```
**Step 3: Install dependencies**

Enter folder using `cd` and install requirements using `pip`
```
cd Code
pip install -r requirements.txt
```
**Step 4: Download the Models**

**Download pretrained models**: https://drive.google.com/uc?id=1jxUd27-K51AkRX20yEHtban8Uvab7lkX

Unzip the file inside the directory **Capstone-IS03_PT-SureBoTv2/Code**

Alternatively you can run the script **DownloadModels.py** to download and unzip the models. Please replace **<MODEL-DOWNLOAD-URL>** with the above link in the script.

The folder structure will look like this:
```
pipeline_models/pretrained_models/BERT-Pair/
    	pytorch_model.bin
    	vocab.txt
    	bert_config.json
    	
pipeline_models/models/bart-large-cnn
	msmarco-distilroberta-base-v2
	pegasus-cnn_dailymail
	stsb-distilbert-base
	1layerbest.pth
	2layerbest.pth
	3layerbest.pth
	4layerbest.pth
	
pipeline_models/trained_models
	finalized_model.pkl
```
**Step 5: Download Spacy Model**

Download ``en_core_web_sm`` via the following command
```
python -m spacy download en_core_web_sm
```
### Usage
We provide 3 methods to implement the system architecture: Command-Line Interface, Localhost Server Bot & Telegram Bot Deployment on GCE


**For details for Localhost and GCE deployment, please refer to Report Appendix for step-by-step guide to setup**

For Command-Line Interface: Quick way to test out fact-checking functionality:

**Step 1: Run**
```
python SureBoT_v2.py
```
**Step 2: Input Query Claim**
```
SureBoT: Input a claim that you would like to fact-check!
```
**Step 3: Enter and Process Claim**

## SECTION 6: SYSTEM EXPLAINER VIDEO


## SECTION 7: PROJECT REPORT / PAPER

`Refer to project report at Github Folder: Project_Report`

- Executive Summary
- Problem Description & Background
- Project Objective
- Proposed Measurement Metrics
- System Overview
- Knowledge Modelling & Representation
- System Architecture
- System Implementation
- Assumptions
- System Performance
- Limitations & Improvements
- Conclusions
- Appendix of report: Project Proposal
- Appendix of report: Installation & User Guide
- Appendix of report: 1-2 pages individual project report per project member
- Appendix of report: Mapped System Functionalities against knowledge, techniques and skills of modular courses
- Appendix of report: Performance Survey



=======

